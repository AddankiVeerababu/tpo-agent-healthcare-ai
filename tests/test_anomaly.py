import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import joblib
import pandas as pd

from app.config import ANOMALY_MODEL_PATH, PREPROCESSING_PATH


def test_anomaly_model_exists():
    assert ANOMALY_MODEL_PATH.exists(), "anomaly_model.pkl should exist after training"


def test_preprocessing_pipeline_exists():
    assert PREPROCESSING_PATH.exists(), "preprocessing.pkl should exist after training"


def test_anomaly_model_returns_valid_output():
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSING_PATH)

    sample_case = pd.DataFrame(
        [
            {
                "patient_age": 80,
                "diagnosis": "Pressure injury",
                "procedure": "Wound care",
                "claim_amount": 3600,
                "prior_visits_30d": 11,
                "documentation_complete": "No",
                "provider_type": "Skilled nursing facility",
                "length_of_stay_days": 3,
                "member_risk_score": 0.97,
                "claim_type": "Payment",
            }
        ]
    )

    processed_case = preprocessor.transform(sample_case)
    prediction = anomaly_model.predict(processed_case)[0]

    assert prediction in [-1, 1]