import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import joblib
import pandas as pd

from app.config import CLASSIFIER_PATH


def test_classifier_model_exists():
    assert CLASSIFIER_PATH.exists(), "classifier.pkl should exist after training"


def test_classifier_predicts_valid_risk_level():
    model = joblib.load(CLASSIFIER_PATH)

    sample_case = pd.DataFrame(
        [
            {
                "patient_age": 67,
                "diagnosis": "Type 2 diabetes with neuropathy",
                "procedure": "Wound care",
                "claim_amount": 1250,
                "prior_visits_30d": 6,
                "documentation_complete": "No",
                "provider_type": "Outpatient clinic",
                "length_of_stay_days": 0,
                "member_risk_score": 0.82,
                "claim_type": "Payment",
            }
        ]
    )

    prediction = model.predict(sample_case)[0]

    assert prediction in ["Low", "Medium", "High"]