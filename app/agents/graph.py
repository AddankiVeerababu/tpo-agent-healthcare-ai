import sys
from pathlib import Path
from typing import TypedDict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from langgraph.graph import StateGraph, END

from app.agents.case_intake_agent import case_intake_agent
from app.agents.policy_retrieval_agent import policy_retrieval_agent
from app.agents.risk_classification_agent import risk_classification_agent
from app.agents.clustering_agent import clustering_agent
from app.agents.anomaly_detection_agent import anomaly_detection_agent
from app.agents.timeseries_anomaly_agent import timeseries_anomaly_agent
from app.agents.shap_explanation_agent import shap_explanation_agent
from app.agents.llm_reasoning_agent import llm_reasoning_agent
from app.agents.governance_agent import governance_agent


class TPOAgentState(TypedDict, total=False):
    case: dict
    validated_case: dict
    intake_status: str
    policy_evidence: list[dict]
    risk_prediction: dict
    cluster_result: dict
    anomaly_result: dict
    timeseries_anomaly_result: dict
    shap_explanation: dict
    llm_reasoning: str
    final_decision: dict


def build_tpo_graph():
    workflow = StateGraph(TPOAgentState)

    workflow.add_node("case_intake_agent", case_intake_agent)
    workflow.add_node("policy_retrieval_agent", policy_retrieval_agent)
    workflow.add_node("risk_classification_agent", risk_classification_agent)
    workflow.add_node("clustering_agent", clustering_agent)
    workflow.add_node("anomaly_detection_agent", anomaly_detection_agent)
    workflow.add_node("timeseries_anomaly_agent", timeseries_anomaly_agent)
    workflow.add_node("shap_explanation_agent", shap_explanation_agent)
    workflow.add_node("llm_reasoning_agent", llm_reasoning_agent)
    workflow.add_node("governance_agent", governance_agent)

    workflow.set_entry_point("case_intake_agent")

    workflow.add_edge("case_intake_agent", "policy_retrieval_agent")
    workflow.add_edge("policy_retrieval_agent", "risk_classification_agent")
    workflow.add_edge("risk_classification_agent", "clustering_agent")
    workflow.add_edge("clustering_agent", "anomaly_detection_agent")
    workflow.add_edge("anomaly_detection_agent", "timeseries_anomaly_agent")
    workflow.add_edge("timeseries_anomaly_agent", "shap_explanation_agent")
    workflow.add_edge("shap_explanation_agent", "llm_reasoning_agent")
    workflow.add_edge("llm_reasoning_agent", "governance_agent")
    workflow.add_edge("governance_agent", END)

    return workflow.compile()


def run_tpo_workflow(case_data: dict) -> dict[str, Any]:
    graph = build_tpo_graph()
    initial_state = {"case": case_data}
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    sample_case = {
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

    result = run_tpo_workflow(sample_case)

    print("\nFinal TPO Agent Decision")
    print("------------------------")
    print(result["final_decision"])

    print("\nCluster Result")
    print("--------------")
    print(result["cluster_result"])

    print("\nTime-Series Anomaly Result")
    print("--------------------------")
    print(result["timeseries_anomaly_result"])