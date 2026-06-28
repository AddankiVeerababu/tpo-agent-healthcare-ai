import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import (
    CLUSTERING_MODEL_PATH,
    TIMESERIES_ANOMALY_MODEL_PATH,
    TIMESERIES_PREPROCESSING_PATH,
)
from app.utils.data_loader import (
    load_claims_data,
    load_claims_timeseries_data,
    get_feature_columns,
)


def build_claim_clustering_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "patient_age",
        "claim_amount",
        "prior_visits_30d",
        "length_of_stay_days",
        "member_risk_score",
    ]

    categorical_features = [
        "diagnosis",
        "procedure",
        "documentation_complete",
        "provider_type",
        "claim_type",
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def train_claim_clustering_model():
    """
    Train a simple KMeans model to group similar synthetic claim patterns.

    This demonstrates clustering and pattern recognition in the prototype.
    """
    print("Loading synthetic claims for clustering...")
    df = load_claims_data()

    X = df[get_feature_columns()]

    preprocessor = build_claim_clustering_preprocessor()

    clustering_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "kmeans",
                KMeans(
                    n_clusters=3,
                    random_state=42,
                    n_init=10,
                ),
            ),
        ]
    )

    clustering_pipeline.fit(X)

    cluster_ids = clustering_pipeline.predict(X)
    df["cluster_id"] = cluster_ids

    print("\nClaim Pattern Clusters")
    print("----------------------")
    cluster_summary = (
        df.groupby("cluster_id")
        .agg(
            claim_count=("case_id", "count"),
            avg_claim_amount=("claim_amount", "mean"),
            avg_prior_visits=("prior_visits_30d", "mean"),
            avg_member_risk=("member_risk_score", "mean"),
        )
        .round(2)
    )

    print(cluster_summary)

    joblib.dump(clustering_pipeline, CLUSTERING_MODEL_PATH)
    print(f"\nSaved clustering model to: {CLUSTERING_MODEL_PATH}")


def build_timeseries_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "claim_count",
        "total_claim_amount",
        "incomplete_documentation_count",
        "avg_member_risk_score",
    ]

    categorical_features = ["procedure"]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def train_timeseries_anomaly_model():
    """
    Train a lightweight time-series anomaly detector using daily synthetic claim activity.

    This is not a production-grade forecasting model. It demonstrates the first principle:
    detecting unusual spikes in claim volume, total amount, documentation gaps, and risk score.
    """
    print("\nLoading synthetic time-series claim activity...")
    df = load_claims_timeseries_data()

    features = [
        "procedure",
        "claim_count",
        "total_claim_amount",
        "incomplete_documentation_count",
        "avg_member_risk_score",
    ]

    X = df[features]

    preprocessor = build_timeseries_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.12,
        random_state=42,
    )

    model.fit(X_processed)

    predictions = model.predict(X_processed)
    scores = model.decision_function(X_processed)

    df["timeseries_anomaly"] = predictions
    df["anomaly_score"] = scores

    print("\nTime-Series Anomaly Candidates")
    print("------------------------------")
    anomaly_rows = df[df["timeseries_anomaly"] == -1][
        [
            "date",
            "procedure",
            "claim_count",
            "total_claim_amount",
            "incomplete_documentation_count",
            "avg_member_risk_score",
            "anomaly_score",
        ]
    ]

    print(anomaly_rows.to_string(index=False))

    joblib.dump(model, TIMESERIES_ANOMALY_MODEL_PATH)
    joblib.dump(preprocessor, TIMESERIES_PREPROCESSING_PATH)

    print(f"\nSaved time-series anomaly model to: {TIMESERIES_ANOMALY_MODEL_PATH}")
    print(f"Saved time-series preprocessing to: {TIMESERIES_PREPROCESSING_PATH}")


def main():
    print("Training advanced pattern-recognition models...")
    train_claim_clustering_model()
    train_timeseries_anomaly_model()
    print("\nAdvanced model training complete.")
    print("Synthetic data only. No PHI was used.")


if __name__ == "__main__":
    main()