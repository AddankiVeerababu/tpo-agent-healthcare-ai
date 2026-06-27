from typing import List, Literal, Optional
from pydantic import BaseModel, Field


RiskLevel = Literal["Low", "Medium", "High"]
DecisionLabel = Literal["Approve", "Manual Review", "Request Documentation"]


class CaseInput(BaseModel):
    patient_age: int = Field(..., ge=0, le=120)
    diagnosis: str = Field(..., min_length=2)
    procedure: str = Field(..., min_length=2)
    claim_amount: float = Field(..., ge=0)
    prior_visits_30d: int = Field(..., ge=0)
    documentation_complete: Literal["Yes", "No"]
    provider_type: str = Field(..., min_length=2)
    length_of_stay_days: int = Field(0, ge=0)
    member_risk_score: float = Field(..., ge=0, le=1)
    claim_type: Literal["Treatment", "Payment", "Operations"]


class PolicyEvidence(BaseModel):
    source: str
    text: str
    relevance_score: Optional[float] = None


class ModelPrediction(BaseModel):
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0, le=1)


class AnomalyResult(BaseModel):
    anomaly_detected: bool
    anomaly_score: float
    explanation: str


class ShapExplanation(BaseModel):
    top_features: List[str]
    explanation_text: str


class GovernanceReview(BaseModel):
    flags: List[str]
    human_review_required: bool
    safety_statement: str


class FinalDecision(BaseModel):
    decision: DecisionLabel
    risk_level: RiskLevel
    anomaly_detected: bool
    policy_evidence: List[PolicyEvidence]
    shap_explanation: ShapExplanation
    reasoning: str
    governance: GovernanceReview
    recommended_action: str