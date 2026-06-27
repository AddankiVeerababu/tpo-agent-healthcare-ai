import joblib
import pandas as pd

from app.config import CLASSIFIER_PATH


def risk_classification_agent(state: dict) -> dict:
    """
    Uses the trained Scikit-learn classifier to predict risk level.
    """
    case_data = state["validated_case"]

    model = joblib.load(CLASSIFIER_PATH)

    input_df = pd.DataFrame([case_data])

    risk_level = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]
    confidence = float(max(probabilities))

    state["risk_prediction"] = {
        "risk_level": risk_level,
        "confidence": round(confidence, 4),
    }

    return state