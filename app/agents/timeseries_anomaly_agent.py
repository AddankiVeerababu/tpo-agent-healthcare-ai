import joblib
import pandas as pd

from app.config import (
    TIMESERIES_ANOMALY_MODEL_PATH,
    TIMESERIES_PREPROCESSING_PATH,
)
from app.utils.data_loader import load_claims_timeseries_data


def get_latest_procedure_activity(procedure: str) -> pd.DataFrame:
    """
    Pull the latest synthetic time-series row for the selected procedure.
    """
    df = load_claims_timeseries_data()

    procedure_rows = df[df["procedure"] == procedure].sort_values("date")

    if procedure_rows.empty:
        # Fallback to the latest row overall if procedure is not in time-series data
        return df.sort_values("date").tail(1)

    return procedure_rows.tail(1)


def timeseries_anomaly_agent(state: dict) -> dict:
    """
    Uses Isolation Forest on synthetic daily claim activity to detect operational spikes.

    This demonstrates time-series anomaly detection at the operations/payment monitoring level.
    """
    case_data = state["validated_case"]
    procedure = case_data["procedure"]

    model = joblib.load(TIMESERIES_ANOMALY_MODEL_PATH)
    preprocessor = joblib.load(TIMESERIES_PREPROCESSING_PATH)

    latest_activity = get_latest_procedure_activity(procedure)

    features = [
        "procedure",
        "claim_count",
        "total_claim_amount",
        "incomplete_documentation_count",
        "avg_member_risk_score",
    ]

    X = latest_activity[features]
    X_processed = preprocessor.transform(X)

    prediction = model.predict(X_processed)[0]
    score = float(model.decision_function(X_processed)[0])

    anomaly_detected = prediction == -1

    activity_record = latest_activity.iloc[0].to_dict()

    state["timeseries_anomaly_result"] = {
        "procedure": procedure,
        "activity_date": str(activity_record["date"].date()),
        "claim_count": int(activity_record["claim_count"]),
        "total_claim_amount": float(activity_record["total_claim_amount"]),
        "incomplete_documentation_count": int(
            activity_record["incomplete_documentation_count"]
        ),
        "avg_member_risk_score": float(activity_record["avg_member_risk_score"]),
        "timeseries_anomaly_detected": bool(anomaly_detected),
        "timeseries_anomaly_score": round(score, 4),
        "explanation": (
            "This module checks whether recent daily claim activity for the selected "
            "procedure looks unusual compared with the synthetic operational trend data. "
            "It is intended as an operations monitoring signal, not a final decision."
        ),
    }

    return state