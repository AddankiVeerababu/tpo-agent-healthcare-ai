from app.schemas.decision_schema import FinalDecision


def determine_decision(state: dict) -> tuple[str, str]:
    case_data = state["validated_case"]
    risk_level = state["risk_prediction"]["risk_level"]
    anomaly_detected = state["anomaly_result"]["anomaly_detected"]

    documentation_complete = case_data["documentation_complete"]

    if documentation_complete == "No":
        return (
            "Request Documentation",
            "Request additional documentation before final payment or operational decision.",
        )

    if risk_level == "High" or anomaly_detected:
        return (
            "Manual Review",
            "Route the case to a qualified human reviewer before final determination.",
        )

    if risk_level == "Medium":
        return (
            "Manual Review",
            "Perform targeted review because the case has moderate risk signals.",
        )

    return (
        "Approve",
        "Proceed with approval in this synthetic demo because risk signals are low.",
    )


def governance_agent(state: dict) -> dict:
    """
    Applies responsible AI and human-in-the-loop governance rules.
    """
    case_data = state["validated_case"]
    risk_level = state["risk_prediction"]["risk_level"]
    anomaly_detected = state["anomaly_result"]["anomaly_detected"]

    decision, recommended_action = determine_decision(state)

    flags = [
        "Synthetic data only",
        "No PHI used",
        "Recommendation only",
        "No autonomous denial",
        "Policy evidence required",
    ]

    human_review_required = False

    if decision in ["Manual Review", "Request Documentation"]:
        human_review_required = True
        flags.append("Human review required")

    if case_data["documentation_complete"] == "No":
        flags.append("Incomplete documentation")

    if risk_level == "High":
        flags.append("High risk prediction")

    if anomaly_detected:
        flags.append("Anomaly detected")

    safety_statement = (
        "This output is an AI-generated decision-support recommendation for a synthetic "
        "healthcare informatics demo. It must not be used for real clinical, billing, "
        "coverage, or payment decisions. Final decisions require qualified human review."
    )

    final_decision = FinalDecision(
        decision=decision,
        risk_level=risk_level,
        anomaly_detected=anomaly_detected,
        policy_evidence=state["policy_evidence"],
        shap_explanation=state["shap_explanation"],
        reasoning=state["llm_reasoning"],
        governance={
            "flags": flags,
            "human_review_required": human_review_required,
            "safety_statement": safety_statement,
        },
        recommended_action=recommended_action,
    )

    state["final_decision"] = final_decision.model_dump()

    return state