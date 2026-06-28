import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st


st.set_page_config(
    page_title="TPO Agent Healthcare AI",
    page_icon="🏥",
    layout="wide",
)


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    color: #f9fafb;
}

.subtitle {
    font-size: 1rem;
    color: #cbd5e1;
    margin-bottom: 1.2rem;
}

.info-banner {
    background-color: #172554;
    border-left: 5px solid #38bdf8;
    padding: 1rem;
    border-radius: 0.5rem;
    color: #e0f2fe;
    margin-bottom: 1.5rem;
}

.workflow-card {
    background-color: #020617;
    color: #f9fafb;
    padding: 1rem;
    border-radius: 0.8rem;
    margin-bottom: 1rem;
    border: 1px solid #334155;
    font-size: 0.95rem;
}

.card {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 1.2rem;
    border-radius: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.35);
    margin-bottom: 1rem;
    color: #f9fafb;
}

.card-title {
    font-size: 0.85rem;
    color: #cbd5e1;
    margin-bottom: 0.35rem;
}

.card-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: #ffffff;
}

.summary-box {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 1.2rem;
    border-radius: 0.8rem;
    margin-bottom: 1rem;
    color: #f9fafb;
    line-height: 1.55;
}

.decision-approve {
    background-color: #052e16;
    color: #86efac;
    border: 1px solid #22c55e;
    padding: 1rem;
    border-radius: 0.8rem;
    font-size: 1.45rem;
    font-weight: 800;
    text-align: center;
}

.decision-review {
    background-color: #451a03;
    color: #fde68a;
    border: 1px solid #f59e0b;
    padding: 1rem;
    border-radius: 0.8rem;
    font-size: 1.45rem;
    font-weight: 800;
    text-align: center;
}

.decision-docs {
    background-color: #450a0a;
    color: #fecaca;
    border: 1px solid #ef4444;
    padding: 1rem;
    border-radius: 0.8rem;
    font-size: 1.45rem;
    font-weight: 800;
    text-align: center;
}

.flag {
    display: inline-block;
    background-color: #1f2937;
    color: #f9fafb;
    padding: 0.35rem 0.6rem;
    border-radius: 999px;
    margin: 0.2rem;
    font-size: 0.85rem;
    border: 1px solid #4b5563;
}

.chat-help {
    background-color: #082f49;
    border: 1px solid #0ea5e9;
    color: #e0f2fe;
    padding: 1rem;
    border-radius: 0.8rem;
    margin-bottom: 1rem;
}

div[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 1rem;
    border-radius: 0.8rem;
}

div[data-testid="stMetric"] label {
    color: #cbd5e1 !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

button[kind="primary"] {
    background-color: #2563eb;
    border-radius: 0.6rem;
    border: none;
    font-weight: 700;
}

button[kind="primary"]:hover {
    background-color: #1d4ed8;
}

.stTabs [data-baseweb="tab"] {
    color: #e5e7eb;
}

.stTabs [aria-selected="true"] {
    color: #60a5fa;
}
</style>
"""


def apply_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def initialize_session_state():
    if "final_state" not in st.session_state:
        st.session_state.final_state = None

    if "final_decision" not in st.session_state:
        st.session_state.final_decision = None

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def render_header():
    st.markdown(
        """
        <div class="main-title">TPO Agent: Governed Agentic AI for Healthcare Decision Support</div>
        <div class="subtitle">
        Agentic AI + RAG + Risk Classification + Anomaly Detection + Explainability + Local LLM Reasoning + Governance
        </div>
        <div class="info-banner">
        <b>Privacy and safety notice:</b> This demo uses synthetic healthcare data only.
        It does not use PHI and must not be used for real clinical, billing, coverage, payment, or operational decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_pipeline():
    st.markdown("### Agentic Workflow")
    st.markdown(
        """
        <div class="workflow-card">
        Case Intake Agent → Policy Retrieval Agent → Risk Classification Agent → 
        Anomaly Detection Agent → Explainability Agent → LLM Reasoning Agent → 
        Governance Review Agent → Final Decision Dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_inputs():
    st.sidebar.title("Synthetic Case Intake")
    st.sidebar.caption("Enter a synthetic Treatment, Payment, or Operations case.")

    with st.sidebar.expander("Patient and Clinical Context", expanded=True):
        patient_age = st.number_input(
            "Patient age",
            min_value=0,
            max_value=120,
            value=67,
        )

        diagnosis = st.text_input(
            "Diagnosis",
            value="Type 2 diabetes with neuropathy",
        )

        procedure = st.selectbox(
            "Procedure",
            [
                "Wound care",
                "Advanced imaging",
                "Chest imaging",
                "Physical therapy",
                "Laboratory panel",
                "Office visit",
                "Joint injection",
            ],
        )

        provider_type = st.selectbox(
            "Provider type",
            [
                "Outpatient clinic",
                "Wound care center",
                "Hospital outpatient",
                "Primary care",
                "Urgent care",
                "Specialist clinic",
                "Rehabilitation clinic",
                "Skilled nursing facility",
            ],
        )

    with st.sidebar.expander("Payment, Utilization, and Risk Signals", expanded=True):
        claim_amount = st.number_input(
            "Claim amount",
            min_value=0.0,
            value=1250.0,
            step=50.0,
        )

        prior_visits_30d = st.number_input(
            "Prior visits in last 30 days",
            min_value=0,
            value=6,
        )

        documentation_complete = st.selectbox(
            "Documentation complete",
            ["No", "Yes"],
        )

        length_of_stay_days = st.number_input(
            "Length of stay days",
            min_value=0,
            value=0,
        )

        member_risk_score = st.slider(
            "Synthetic member risk score",
            min_value=0.0,
            max_value=1.0,
            value=0.82,
            step=0.01,
        )

        claim_type = st.selectbox(
            "TPO workflow type",
            ["Payment", "Treatment", "Operations"],
        )

    return {
        "patient_age": patient_age,
        "diagnosis": diagnosis,
        "procedure": procedure,
        "claim_amount": claim_amount,
        "prior_visits_30d": prior_visits_30d,
        "documentation_complete": documentation_complete,
        "provider_type": provider_type,
        "length_of_stay_days": length_of_stay_days,
        "member_risk_score": member_risk_score,
        "claim_type": claim_type,
    }


def render_case_summary(case_data: dict):
    st.markdown("### Case Summary")

    col1, col2, col3, col4 = st.columns(4)

    cards = [
        ("Procedure", case_data["procedure"]),
        ("Claim Amount", f"${case_data['claim_amount']:,.0f}"),
        ("Prior Visits", str(case_data["prior_visits_30d"])),
        ("Documentation", case_data["documentation_complete"]),
    ]

    for col, (title, value) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">{title}</div>
                    <div class="card-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("View full synthetic case input"):
        st.table(
            {
                "Field": list(case_data.keys()),
                "Value": [str(value) for value in case_data.values()],
            }
        )


def decision_class(decision: str) -> str:
    if decision == "Approve":
        return "decision-approve"
    if decision == "Manual Review":
        return "decision-review"
    return "decision-docs"


def render_executive_decision(final_decision: dict):
    decision = final_decision["decision"]
    risk_level = final_decision["risk_level"]
    anomaly_detected = final_decision["anomaly_detected"]
    human_review_required = final_decision["governance"]["human_review_required"]

    st.markdown("### Decision Summary")

    decision_col, risk_col, anomaly_col, review_col = st.columns([2, 1, 1, 1])

    with decision_col:
        st.markdown(
            f"""
            <div class="{decision_class(decision)}">
                {decision}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with risk_col:
        st.metric("Risk Level", risk_level)

    with anomaly_col:
        st.metric("Anomaly", "Yes" if anomaly_detected else "No")

    with review_col:
        st.metric("Human Review", "Yes" if human_review_required else "No")

    st.markdown(
        f"""
        <div class="summary-box">
            <b>Recommended Action:</b><br>
            {final_decision["recommended_action"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_policy_evidence(final_decision: dict):
    st.markdown("### Retrieved Policy Evidence")

    for idx, evidence in enumerate(final_decision["policy_evidence"], start=1):
        source = evidence["source"]
        relevance = evidence["relevance_score"]
        text = evidence["text"]

        with st.expander(f"Policy Evidence {idx}: {source} | Relevance {relevance}"):
            st.write(text)


def render_reasoning(final_decision: dict):
    st.markdown("### Policy-Aware LLM Reasoning")
    st.markdown(
        f"""
        <div class="summary-box">
            {final_decision["reasoning"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explainability(final_decision: dict, final_state: dict):
    st.markdown("### Explainability Summary")

    explanation = final_decision["shap_explanation"]["explanation_text"]
    top_features = final_decision["shap_explanation"]["top_features"]

    st.markdown(
        f"""
        <div class="summary-box">
            {explanation}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Top Contributing Features")

    cols = st.columns(min(len(top_features), 5))

    for col, feature in zip(cols, top_features):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-value" style="font-size:1rem;">{feature}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    try:
        from app.utils.visualization import anomaly_gauge

        st.markdown("### Anomaly Signal")
        anomaly_score = final_state["anomaly_result"]["anomaly_score"]
        anomaly_detected = final_state["anomaly_result"]["anomaly_detected"]

        st.plotly_chart(
            anomaly_gauge(anomaly_detected, anomaly_score),
            use_container_width=True,
        )

        st.info(final_state["anomaly_result"]["explanation"])

    except Exception as error:
        st.warning("Anomaly gauge could not be displayed.")
        st.write(str(error))


def render_governance(final_decision: dict):
    st.markdown("### Governance and Responsible AI")

    flags = final_decision["governance"]["flags"]
    safety_statement = final_decision["governance"]["safety_statement"]

    flag_html = "".join([f'<span class="flag">✅ {flag}</span>' for flag in flags])

    st.markdown(
        f"""
        <div class="summary-box">
            {flag_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(safety_statement)


def render_dataset_monitoring():
    st.markdown("### Synthetic Dataset Monitoring")

    try:
        from app.utils.data_loader import load_claims_data
        from app.utils.visualization import (
            risk_distribution_chart,
            claim_amount_by_procedure_chart,
        )

        df = load_claims_data()

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.plotly_chart(risk_distribution_chart(df), use_container_width=True)

        with chart_col2:
            st.plotly_chart(claim_amount_by_procedure_chart(df), use_container_width=True)

        with st.expander("View synthetic claims dataset"):
            st.dataframe(df, use_container_width=True)

    except Exception as error:
        st.error("Dataset monitoring could not be loaded.")
        st.write(str(error))


def build_case_chat_context(final_state: dict) -> str:
    case_data = final_state.get("validated_case", {})
    final_decision = final_state.get("final_decision", {})
    risk_prediction = final_state.get("risk_prediction", {})
    anomaly_result = final_state.get("anomaly_result", {})
    shap_explanation = final_state.get("shap_explanation", {})
    policy_evidence = final_state.get("policy_evidence", [])

    policy_text = "\n\n".join(
        [
            f"Source: {item.get('source')}\nText: {item.get('text')}"
            for item in policy_evidence
        ]
    )

    return f"""
Synthetic Case:
{json.dumps(case_data, indent=2)}

Risk Prediction:
{json.dumps(risk_prediction, indent=2)}

Anomaly Result:
{json.dumps(anomaly_result, indent=2)}

Explainability:
{json.dumps(shap_explanation, indent=2)}

Final Decision:
{json.dumps(final_decision, indent=2)}

Retrieved Synthetic Policy Evidence:
{policy_text}
"""


def answer_case_question(question: str, final_state: dict) -> str:
    from app.config import OLLAMA_MODEL_NAME

    context = build_case_chat_context(final_state)

    prompt = f"""
You are a careful healthcare AI case-review assistant for a synthetic TPO demo.

Rules:
- Answer only using the provided synthetic case context.
- Do not provide real medical advice.
- Do not diagnose patients.
- Do not make final clinical, payment, billing, or coverage decisions.
- Do not invent facts that are not in the case context.
- If the answer is uncertain, say what information is missing.
- Keep the answer clear, concise, and useful for a human reviewer.

Case Context:
{context}

User Question:
{question}

Answer:
"""

    try:
        import ollama

        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions about a synthetic healthcare TPO case. "
                        "You are careful, policy-aware, and governance-aware."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response["message"]["content"].strip()

    except Exception as error:
        return (
            "I could not reach Ollama, so here is a safe fallback answer: "
            "Based on the current synthetic case, the recommendation should be interpreted "
            "as decision support only. Review the risk level, anomaly result, retrieved policy "
            "evidence, documentation completeness, and governance flags before taking action. "
            f"Technical note: {str(error)}"
        )


def render_case_chat_assistant():
    st.markdown("### Case Q&A Assistant")

    if st.session_state.final_state is None:
        st.info("Run the governed agentic AI workflow first. Then you can ask questions about the case.")
        return

    st.markdown(
        """
        <div class="chat-help">
        Ask questions about this synthetic case, the policy evidence, the risk prediction,
        anomaly result, explainability summary, governance flags, or recommended action.
        </div>
        """,
        unsafe_allow_html=True,
    )

    example_questions = [
        "Why did this case require documentation?",
        "What policy evidence supports the decision?",
        "What factors contributed most to the high risk prediction?",
        "Is this an autonomous denial?",
        "What should the human reviewer check next?",
    ]

    selected_question = st.selectbox(
        "Example questions",
        [""] + example_questions,
    )

    user_question = st.chat_input("Ask a question about the current synthetic case")

    if selected_question and selected_question != "":
        user_question = selected_question

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_question:
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Answering based on the current case and retrieved policy evidence..."):
                answer = answer_case_question(
                    user_question,
                    st.session_state.final_state,
                )
                st.write(answer)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


def run_workflow(case_data: dict):
    from app.agents.graph import run_tpo_workflow

    return run_tpo_workflow(case_data)


def main():
    initialize_session_state()
    apply_css()
    render_header()

    case_data = render_sidebar_inputs()

    render_agent_pipeline()
    render_case_summary(case_data)

    st.markdown("---")

    run_button = st.button(
        "Run Governed Agentic AI Workflow",
        type="primary",
        use_container_width=True,
    )

    if run_button:
        st.session_state.chat_messages = []

        with st.spinner(
            "Running agentic workflow: intake, RAG retrieval, risk prediction, anomaly detection, reasoning, and governance..."
        ):
            final_state = run_workflow(case_data)
            final_decision = final_state["final_decision"]

        st.session_state.final_state = final_state
        st.session_state.final_decision = final_decision

    if st.session_state.final_decision is not None:
        st.markdown("---")
        render_executive_decision(st.session_state.final_decision)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "Policy Evidence",
                "LLM Reasoning",
                "Explainability",
                "Governance",
                "Case Q&A Assistant",
                "Technical Details",
            ]
        )

        with tab1:
            render_policy_evidence(st.session_state.final_decision)

        with tab2:
            render_reasoning(st.session_state.final_decision)

        with tab3:
            render_explainability(
                st.session_state.final_decision,
                st.session_state.final_state,
            )

        with tab4:
            render_governance(st.session_state.final_decision)

        with tab5:
            render_case_chat_assistant()

        with tab6:
            st.markdown("### Technical Agent State")
            st.caption("This section is included for transparency and debugging.")
            st.json(st.session_state.final_state)

    else:
        st.info("Click the workflow button to generate a decision and enable case-based Q&A.")

    st.markdown("---")
    render_dataset_monitoring()


if __name__ == "__main__":
    main()