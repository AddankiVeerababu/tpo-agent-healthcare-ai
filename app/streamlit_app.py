import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st


st.set_page_config(
    page_title="TPO Agent",
    page_icon="🏥",
    layout="wide",
)


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1360px;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}

.app-title {
    font-size: 2.25rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.2rem;
}

.app-subtitle {
    color: #cbd5e1;
    font-size: 1rem;
    margin-bottom: 1rem;
}

.notice {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-left: 5px solid #38bdf8;
    padding: 0.9rem 1rem;
    border-radius: 0.7rem;
    color: #e0f2fe;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}

.workflow {
    background-color: #111827;
    border: 1px solid #374151;
    border-radius: 0.8rem;
    padding: 0.9rem 1rem;
    color: #e5e7eb;
    font-size: 0.92rem;
    margin-bottom: 1rem;
}

.section-card {
    background-color: #111827;
    border: 1px solid #374151;
    border-radius: 0.85rem;
    padding: 1.1rem;
    color: #f8fafc;
    margin-bottom: 1rem;
}

.kpi-card {
    background-color: #111827;
    border: 1px solid #374151;
    border-radius: 0.85rem;
    padding: 1rem;
    color: #f8fafc;
    min-height: 96px;
}

.kpi-label {
    color: #94a3b8;
    font-size: 0.82rem;
    margin-bottom: 0.4rem;
}

.kpi-value {
    color: #f8fafc;
    font-size: 1.25rem;
    font-weight: 800;
    line-height: 1.25;
}

.decision-docs {
    background-color: #450a0a;
    color: #fecaca;
    border: 1px solid #ef4444;
    padding: 1rem;
    border-radius: 0.85rem;
    font-size: 1.45rem;
    font-weight: 800;
    text-align: center;
}

.decision-review {
    background-color: #451a03;
    color: #fde68a;
    border: 1px solid #f59e0b;
    padding: 1rem;
    border-radius: 0.85rem;
    font-size: 1.45rem;
    font-weight: 800;
    text-align: center;
}

.decision-approve {
    background-color: #052e16;
    color: #86efac;
    border: 1px solid #22c55e;
    padding: 1rem;
    border-radius: 0.85rem;
    font-size: 1.45rem;
    font-weight: 800;
    text-align: center;
}

.flag {
    display: inline-block;
    background-color: #1f2937;
    color: #f8fafc;
    padding: 0.38rem 0.65rem;
    border-radius: 999px;
    margin: 0.22rem;
    font-size: 0.84rem;
    border: 1px solid #4b5563;
}

.chat-note {
    background-color: #082f49;
    border: 1px solid #0284c7;
    color: #e0f2fe;
    padding: 0.9rem 1rem;
    border-radius: 0.85rem;
    margin-bottom: 1rem;
}

div[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 1rem;
    border-radius: 0.85rem;
}

div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

button[kind="primary"] {
    background-color: #2563eb;
    border-radius: 0.7rem;
    border: none;
    font-weight: 700;
}

button[kind="primary"]:hover {
    background-color: #1d4ed8;
}

.stTabs [data-baseweb="tab"] {
    color: #e5e7eb;
    font-size: 0.94rem;
}

.stTabs [aria-selected="true"] {
    color: #60a5fa;
}
</style>
"""


def apply_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def initialize_session_state():
    defaults = {
        "final_state": None,
        "final_decision": None,
        "chat_messages": [],
        "case_data": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    st.markdown(
        """
        <div class="app-title">TPO Agent</div>
        <div class="app-subtitle">
        Governed AI case review for synthetic healthcare Treatment, Payment, and Operations workflows.
        </div>
        <div class="notice">
        This is a synthetic proof of concept. It uses fake healthcare cases and fictional policy text.
        It is not designed for real clinical, billing, coverage, or payment decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow():
    st.markdown(
        """
        <div class="workflow">
        Case Intake → Policy Retrieval → Risk Prediction → Clustering → Claim Anomaly Check → 
        Time-Series Signal → Explanation → Local LLM Reasoning → Governance Review
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_case_input():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Clinical Context")

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

    with col2:
        st.markdown("#### Provider and Utilization")

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

        prior_visits_30d = st.number_input(
            "Prior visits in last 30 days",
            min_value=0,
            value=6,
        )

        length_of_stay_days = st.number_input(
            "Length of stay days",
            min_value=0,
            value=0,
        )

    with col3:
        st.markdown("#### Payment and Risk")

        claim_amount = st.number_input(
            "Claim amount",
            min_value=0.0,
            value=1250.0,
            step=50.0,
        )

        documentation_complete = st.selectbox(
            "Documentation complete",
            ["No", "Yes"],
        )

        member_risk_score = st.slider(
            "Synthetic member risk score",
            min_value=0.0,
            max_value=1.0,
            value=0.82,
            step=0.01,
        )

        claim_type = st.selectbox(
            "Workflow type",
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


def run_workflow(case_data: dict):
    from app.agents.graph import run_tpo_workflow

    return run_tpo_workflow(case_data)


def render_case_kpis(case_data: dict):
    st.markdown("#### Case Snapshot")

    col1, col2, col3, col4 = st.columns(4)

    items = [
        ("Procedure", case_data["procedure"]),
        ("Claim Amount", f"${case_data['claim_amount']:,.0f}"),
        ("Prior Visits", str(case_data["prior_visits_30d"])),
        ("Documentation", case_data["documentation_complete"]),
    ]

    for col, (label, value) in zip([col1, col2, col3, col4], items):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def decision_class(decision: str) -> str:
    if decision == "Approve":
        return "decision-approve"
    if decision == "Manual Review":
        return "decision-review"
    return "decision-docs"


def render_decision_page():
    final_decision = st.session_state.final_decision
    final_state = st.session_state.final_state

    if final_decision is None:
        st.info("Run a case review first from the Case Intake tab.")
        return

    decision = final_decision["decision"]
    risk_level = final_decision["risk_level"]
    claim_anomaly = "Yes" if final_decision["anomaly_detected"] else "No"
    human_review = (
        "Yes" if final_decision["governance"]["human_review_required"] else "No"
    )

    st.markdown("### Decision Summary")

    col1, col2, col3, col4 = st.columns([2.2, 1, 1, 1])

    with col1:
        st.markdown(
            f"""
            <div class="{decision_class(decision)}">
            {decision}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Risk Level", risk_level)

    with col3:
        st.metric("Claim Anomaly", claim_anomaly)

    with col4:
        st.metric("Human Review", human_review)

    st.markdown(
        f"""
        <div class="section-card">
        <b>Recommended next step</b><br><br>
        {final_decision["recommended_action"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Governance Flags")

    flags = final_decision["governance"]["flags"]
    flag_html = "".join([f'<span class="flag">{flag}</span>' for flag in flags])

    st.markdown(
        f"""
        <div class="section-card">
        {flag_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(final_decision["governance"]["safety_statement"])

    with st.expander("Technical trace"):
        st.json(final_state)


def render_evidence_page():
    final_decision = st.session_state.final_decision

    if final_decision is None:
        st.info("Run a case review first from the Case Intake tab.")
        return

    st.markdown("### Retrieved Policy Evidence")
    st.caption("These are fictional policy documents used only for the synthetic demo.")

    for index, item in enumerate(final_decision["policy_evidence"], start=1):
        with st.expander(
            f"{index}. {item['source']} | relevance: {item['relevance_score']}",
            expanded=index == 1,
        ):
            st.write(item["text"])

    st.markdown("### Case Reasoning")

    st.markdown(
        f"""
        <div class="section-card">
        {final_decision["reasoning"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pattern_page():
    final_state = st.session_state.final_state

    if final_state is None:
        st.info("Run a case review first from the Case Intake tab.")
        return

    cluster = final_state.get("cluster_result", {})
    claim_anomaly = final_state.get("anomaly_result", {})
    ts = final_state.get("timeseries_anomaly_result", {})

    st.markdown("### Pattern Recognition")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="section-card">
            <b>Claim Pattern Cluster</b><br><br>
            Cluster ID: {cluster.get("cluster_id", "N/A")}<br>
            Pattern: {cluster.get("cluster_description", "N/A")}<br><br>
            {cluster.get("explanation", "")}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        ts_anomaly = "Yes" if ts.get("timeseries_anomaly_detected", False) else "No"
        ts_amount = ts.get("total_claim_amount", 0)

        st.markdown(
            f"""
            <div class="section-card">
            <b>Operational Time-Series Signal</b><br><br>
            Procedure: {ts.get("procedure", "N/A")}<br>
            Activity date: {ts.get("activity_date", "N/A")}<br>
            Claim count: {ts.get("claim_count", "N/A")}<br>
            Total amount: ${ts_amount:,.0f}<br>
            Incomplete documentation count: {ts.get("incomplete_documentation_count", "N/A")}<br>
            Time-series anomaly: {ts_anomaly}<br><br>
            {ts.get("explanation", "")}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Explainability")

    shap_info = final_state.get("shap_explanation", {})
    st.markdown(
        f"""
        <div class="section-card">
        {shap_info.get("explanation_text", "")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = shap_info.get("top_features", [])

    if features:
        cols = st.columns(min(5, len(features)))
        for col, feature in zip(cols, features):
            with col:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                    <div class="kpi-label">Model signal</div>
                    <div class="kpi-value" style="font-size:1rem;">{feature}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.info(claim_anomaly.get("explanation", ""))

    try:
        from app.utils.visualization import anomaly_gauge

        st.plotly_chart(
            anomaly_gauge(
                claim_anomaly.get("anomaly_detected", False),
                claim_anomaly.get("anomaly_score", 0),
            ),
            use_container_width=True,
        )
    except Exception:
        pass


def build_case_chat_context(final_state: dict) -> str:
    case_data = final_state.get("validated_case", {})
    risk_prediction = final_state.get("risk_prediction", {})
    cluster_result = final_state.get("cluster_result", {})
    anomaly_result = final_state.get("anomaly_result", {})
    timeseries_result = final_state.get("timeseries_anomaly_result", {})
    shap_result = final_state.get("shap_explanation", {})
    final_decision = final_state.get("final_decision", {})
    policy_evidence = final_state.get("policy_evidence", [])

    policy_text = "\n\n".join(
        [
            f"Source: {item.get('source')}\nText: {item.get('text')}"
            for item in policy_evidence
        ]
    )

    return f"""
Current Synthetic Case:
{json.dumps(case_data, indent=2)}

Risk Prediction:
{json.dumps(risk_prediction, indent=2)}

Claim Pattern Cluster:
{json.dumps(cluster_result, indent=2)}

Claim-Level Anomaly Result:
{json.dumps(anomaly_result, indent=2)}

Time-Series Operations Signal:
{json.dumps(timeseries_result, indent=2)}

Explainability:
{json.dumps(shap_result, indent=2)}

Final Recommendation:
{json.dumps(final_decision, indent=2)}

Policy Evidence:
{policy_text}
"""


def answer_case_question(question: str, final_state: dict) -> str:
    from app.config import OLLAMA_MODEL_NAME

    context = build_case_chat_context(final_state)

    prompt = f"""
You are helping a reviewer understand one synthetic healthcare case.

Use only the context below.
Do not give medical advice.
Do not make final payment, billing, coverage, or clinical decisions.
Do not invent facts.
If information is missing, say what is missing.
Avoid raw JSON in the final answer.
Keep the answer direct and practical.

Context:
{context}

Question:
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
                    "content": "You explain synthetic healthcare case-review outputs in plain language.",
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
            "I could not reach the local Ollama model. Review the risk level, policy evidence, "
            "documentation status, pattern signals, and governance flags before taking action. "
            f"Technical note: {str(error)}"
        )


def render_assistant_page():
    if st.session_state.final_state is None:
        st.info("Run a case review first from the Case Intake tab.")
        return

    st.markdown("### Reviewer Assistant")

    st.markdown(
        """
        <div class="section-card">
        Ask questions about the current synthetic case. The assistant answers using the case output,
        retrieved policy evidence, model signals, and governance flags.
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick_questions = [
        "",
        "Why was this decision recommended?",
        "What policy evidence supports the decision?",
        "What cluster does this case belong to?",
        "Is there a time-series anomaly for this procedure?",
        "What should the human reviewer check next?",
        "Is this an autonomous denial?",
    ]

    selected_question = st.selectbox("Choose a common reviewer question", quick_questions)
    typed_question = st.chat_input("Ask a question about this case")

    question = typed_question or selected_question

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if question:
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Checking the case..."):
                answer = answer_case_question(question, st.session_state.final_state)
                st.write(answer)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


def render_data_monitor_page():
    st.markdown("### Synthetic Data Monitor")

    try:
        from app.utils.data_loader import load_claims_data, load_claims_timeseries_data
        from app.utils.visualization import (
            risk_distribution_chart,
            claim_amount_by_procedure_chart,
        )
        import plotly.express as px

        claims_df = load_claims_data()
        ts_df = load_claims_timeseries_data()

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(risk_distribution_chart(claims_df), use_container_width=True)

        with col2:
            st.plotly_chart(
                claim_amount_by_procedure_chart(claims_df),
                use_container_width=True,
            )

        st.markdown("### Daily Synthetic Claim Activity")

        line_chart = px.line(
            ts_df,
            x="date",
            y="total_claim_amount",
            color="procedure",
            markers=True,
            title="Synthetic Daily Claim Amount by Procedure",
        )
        st.plotly_chart(line_chart, use_container_width=True)

        with st.expander("View synthetic claims"):
            st.dataframe(claims_df, use_container_width=True)

        with st.expander("View synthetic time-series activity"):
            st.dataframe(ts_df, use_container_width=True)

    except Exception as error:
        st.error("Data monitor could not be loaded.")
        st.write(str(error))


def main():
    initialize_session_state()
    apply_css()

    render_header()
    render_workflow()

    main_tabs = st.tabs(
        [
            "1. Case Intake",
            "2. Decision",
            "3. Evidence",
            "4. Pattern Signals",
            "5. Reviewer Assistant",
            "6. Data Monitor",
        ]
    )

    with main_tabs[0]:
        st.markdown("### New Case Review")
        case_data = get_case_input()
        st.session_state.case_data = case_data

        render_case_kpis(case_data)

        st.markdown("---")

        if st.button("Run Case Review", type="primary", use_container_width=True):
            st.session_state.chat_messages = []

            with st.spinner("Reviewing the case..."):
                final_state = run_workflow(case_data)
                final_decision = final_state["final_decision"]

            st.session_state.final_state = final_state
            st.session_state.final_decision = final_decision

            st.success("Case review completed. Open the Decision tab to review the result.")

    with main_tabs[1]:
        render_decision_page()

    with main_tabs[2]:
        render_evidence_page()

    with main_tabs[3]:
        render_pattern_page()

    with main_tabs[4]:
        render_assistant_page()

    with main_tabs[5]:
        render_data_monitor_page()


if __name__ == "__main__":
    main()