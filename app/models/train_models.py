import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import (
    CLASSIFIER_PATH,
    ANOMALY_MODEL_PATH,
    PREPROCESSING_PATH,
)
from app.utils.data_loader import (
    load_claims_data,
    get_feature_columns,
    get_target_column,
)


def build_preprocessor() -> ColumnTransformer:
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

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_risk_classifier(df: pd.DataFrame):
    X = df[get_feature_columns()]
    y = df[get_target_column()]

    preprocessor = build_preprocessor()

    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nRisk Classification Model Results")
    print("--------------------------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(classification_report(y_test, predictions))

    joblib.dump(model, CLASSIFIER_PATH)
    print(f"Saved classifier to: {CLASSIFIER_PATH}")

    return model


def train_anomaly_model(df: pd.DataFrame):
    X = df[get_feature_columns()]

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    anomaly_model = IsolationForest(
        n_estimators=200,
        contamination=0.18,
        random_state=42,
    )

    anomaly_model.fit(X_processed)

    joblib.dump(anomaly_model, ANOMALY_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSING_PATH)

    print("\nAnomaly Detection Model")
    print("-----------------------")
    print(f"Saved anomaly model to: {ANOMALY_MODEL_PATH}")
    print(f"Saved preprocessing pipeline to: {PREPROCESSING_PATH}")

    return anomaly_model, preprocessor


def main():
    print("Loading synthetic healthcare claims data...")
    df = load_claims_data()

    print(f"Loaded {len(df)} synthetic claims.")
    print("Training risk classifier...")
    train_risk_classifier(df)

    print("Training anomaly detection model...")
    train_anomaly_model(df)

    print("\nTraining complete.")
    print("Synthetic data only. No PHI was used.")


if __name__ == "__main__":
    main()