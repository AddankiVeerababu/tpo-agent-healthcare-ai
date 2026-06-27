import joblib
import pandas as pd

from app.config import ANOMALY_MODEL_PATH, PREPROCESSING_PATH


def anomaly_detection_agent(state: dict) -> dict:
    """
    Uses Isolation Forest to identify unusual synthetic claim patterns.
    """
    case_data = state["validated_case"]

    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSING_PATH)

    input_df = pd.DataFrame([case_data])
    input_processed = preprocessor.transform(input_df)

    prediction = anomaly_model.predict(input_processed)[0]
    score = float(anomaly_model.decision_function(input_processed)[0])

    anomaly_detected = prediction == -1

    if anomaly_detected:
        explanation = (
            "The case appears unusual compared with the synthetic claims dataset. "
            "Potential contributors may include claim amount, utilization frequency, "
            "documentation status, provider setting, or member risk score."
        )
    else:
        explanation = (
            "The case does not appear highly unusual compared with the synthetic claims dataset."
        )

    state["anomaly_result"] = {
        "anomaly_detected": bool(anomaly_detected),
        "anomaly_score": round(score, 4),
        "explanation": explanation,
    }

    return state