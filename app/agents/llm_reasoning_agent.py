import json
import ollama

from app.config import OLLAMA_MODEL_NAME


def build_reasoning_prompt(state: dict) -> str:
    case_data = state["validated_case"]
    risk_prediction = state["risk_prediction"]
    anomaly_result = state["anomaly_result"]
    shap_explanation = state["shap_explanation"]
    policy_evidence = state["policy_evidence"]

    policy_text = "\n\n".join(
        [
            f"Source: {item['source']}\nPolicy Text: {item['text']}"
            for item in policy_evidence
        ]
    )

    prompt = f"""
You are a healthcare AI decision-support assistant for a synthetic Treatment, Payment, and Operations demo.

Important rules:
- Use synthetic data only.
- Do not generate real medical advice.
- Do not diagnose patients.
- Do not make autonomous denial decisions.
- Use the provided case facts and policy evidence only.
- If documentation is incomplete, recommend requesting documentation.
- If risk is high or anomaly is detected, recommend human review.
- Keep the explanation concise, professional, and policy-aware.

Synthetic Case:
{json.dumps(case_data, indent=2)}

Risk Prediction:
{json.dumps(risk_prediction, indent=2)}

Anomaly Result:
{json.dumps(anomaly_result, indent=2)}

Explainability Summary:
{json.dumps(shap_explanation, indent=2)}

Retrieved Synthetic Policy Evidence:
{policy_text}

Write a clear reasoning paragraph explaining:
1. Why the case was assigned this risk level.
2. Why the recommended action is appropriate.
3. Why human review may be needed.
4. That this is decision support only.
"""
    return prompt


def llm_reasoning_agent(state: dict) -> dict:
    """
    Uses Ollama local LLM to generate policy-aware reasoning.

    If Ollama is unavailable, the agent falls back to a safe deterministic explanation.
    """
    prompt = build_reasoning_prompt(state)

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious healthcare AI assistant for synthetic "
                        "Treatment, Payment, and Operations decision support."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        reasoning = response["message"]["content"].strip()

    except Exception as error:
        reasoning = (
            "The case was evaluated using synthetic policy evidence, risk classification, "
            "anomaly detection, and explainability signals. Because the case may involve "
            "incomplete documentation, elevated utilization, or higher predicted risk, it "
            "should be treated as decision support only and routed according to governance "
            "rules. Ollama reasoning was unavailable, so this fallback explanation was used. "
            f"Technical note: {str(error)}"
        )

    state["llm_reasoning"] = reasoning

    return state