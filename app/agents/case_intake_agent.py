from app.schemas.decision_schema import CaseInput


def case_intake_agent(state: dict) -> dict:
    """
    Validates and standardizes the incoming synthetic healthcare case.

    This agent ensures that the case input follows the expected schema
    before any ML, RAG, or LLM step runs.
    """
    raw_case = state.get("case", {})

    validated_case = CaseInput(**raw_case)

    state["validated_case"] = validated_case.model_dump()
    state["intake_status"] = "validated"

    return state