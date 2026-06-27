import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pytest
from pydantic import ValidationError

from app.schemas.decision_schema import CaseInput, FinalDecision


def test_case_input_accepts_valid_case():
    case = CaseInput(
        patient_age=67,
        diagnosis="Type 2 diabetes with neuropathy",
        procedure="Wound care",
        claim_amount=1250,
        prior_visits_30d=6,
        documentation_complete="No",
        provider_type="Outpatient clinic",
        length_of_stay_days=0,
        member_risk_score=0.82,
        claim_type="Payment",
    )

    assert case.patient_age == 67
    assert case.documentation_complete == "No"


def test_case_input_rejects_invalid_age():
    with pytest.raises(ValidationError):
        CaseInput(
            patient_age=140,
            diagnosis="Type 2 diabetes",
            procedure="Wound care",
            claim_amount=1250,
            prior_visits_30d=6,
            documentation_complete="No",
            provider_type="Outpatient clinic",
            length_of_stay_days=0,
            member_risk_score=0.82,
            claim_type="Payment",
        )


def test_final_decision_schema_accepts_valid_output():
    decision = FinalDecision(
        decision="Request Documentation",
        risk_level="High",
        anomaly_detected=False,
        policy_evidence=[
            {
                "source": "wound_care_policy.txt",
                "text": "Synthetic wound care policy evidence.",
                "relevance_score": 0.91,
            }
        ],
        shap_explanation={
            "top_features": ["documentation complete", "claim amount"],
            "explanation_text": "Missing documentation and claim amount influenced the prediction.",
        },
        reasoning="The case requires documentation review.",
        governance={
            "flags": ["Synthetic data only", "Human review required"],
            "human_review_required": True,
            "safety_statement": "Decision support only.",
        },
        recommended_action="Request additional documentation.",
    )

    assert decision.decision == "Request Documentation"
    assert decision.governance.human_review_required is True