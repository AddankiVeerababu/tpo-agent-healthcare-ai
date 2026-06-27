import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st


st.set_page_config(
    page_title="TPO Agent Healthcare AI",
    page_icon="TPO",
    layout="wide",
)


def render_header():
    st.title("TPO Agent: Governed Agentic AI for Healthcare Decision Support")
    st.caption(
        "Agentic AI + RAG + ML Classification + Anomaly Detection + Explainability + Governance"
    )

    st.info(
        "This application uses synthetic healthcare data only. "
        "It does not use PHI and must not be used for real clinical, billing, coverage, or payment decisions."
    )


def render_sidebar_inputs():
    st.sidebar.header("Synthetic Case Intake")

    patient_age = st.sidebar.number_input(
        "Patient age",
        min_value=0,
        max_value=120,
        value=67,
    )

    diagnosis = st.sidebar.text_input(
        "Diagnosis",
        value="Type 2 diabetes with neuropathy",
    )

    procedure = st.sidebar.selectbox(
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

    claim_amount = st.sidebar.number_input(
        "Claim amount",
        min_value=0.0,
        value=1250.0,
        step=50.0,
    )

    prior_visits_30d = st.sidebar.number_input(
        "Prior visits in last 30 days",
        min_value=0,
        value=6,
    )

    documentation_complete = st.sidebar.selectbox(
        "Documentation complete",
        ["No", "Yes"],
    )

    provider_type = st.sidebar.selectbox(
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

    length_of_stay_days = st.sidebar.number_input(
        "Length of stay days",
        min_value=0,
        value=0,
    )

    member_risk_score = st.sidebar.slider(
        "Synthetic member risk score",
        min_value=0.0,
        max_value=1.0,
        value=0.82,
        step=0.01,
    )

    claim_type = st.sidebar.selectbox(
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


def render_decision_badge(decision: str):
    if decision == "Approve":
        st.success(f"Decision: {decision}")
    elif decision == "Manual Review":
        st.warning(f"Decision: {decision}")
    else:
        st.error(f"Decision: {decision}")


def render_final_decision(final_decision: dict, final_state: dict):
    decision = final_decision["decision"]
    risk_level = final_decision["risk_level"]
    anomaly_detected = final_decision["anomaly_detected"]

    render_decision_badge(decision)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Risk Level", risk_level)

    with col2:
        st.metric("Anomaly Detected", "Yes" if anomaly_detected else "No")

    with col3:
        human_review = final_decision["governance"]["human_review_required"]
        st.metric("Human Review Required", "Yes" if human_review else "No")

    st.subheader("Recommended Action")
    st.write(final_decision["recommended_action"])

    st.subheader("Policy-Aware LLM Reasoning")
    st.write(final_decision["reasoning"])

    st.subheader("Explainability Summary")
    st.write(final_decision["shap_explanation"]["explanation_text"])

    with st.expander("Top Explainability Features"):
        for feature in final_decision["shap_explanation"]["top_features"]:
            st.write(f"- {feature}")

    st.subheader("Retrieved Policy Evidence")

    for evidence in final_decision["policy_evidence"]:
        title = f"{evidence['source']} | relevance score: {evidence['relevance_score']}"
        with st.expander(title):
            st.write(evidence["text"])

    st.subheader("Governance and Responsible AI Flags")

    for flag in final_decision["governance"]["flags"]:
        st.write(f"✅ {flag}")

    st.warning(final_decision["governance"]["safety_statement"])

    with st.expander("Technical Agent State"):
        st.write(final_state)


def render_dataset_monitoring():
    st.subheader("Synthetic Dataset Monitoring")

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

        with st.expander("View Synthetic Claims Dataset"):
            st.dataframe(df, use_container_width=True)

    except Exception as error:
        st.error("Dataset monitoring could not be loaded.")
        st.write(str(error))


def main():
    render_header()

    case_data = render_sidebar_inputs()

    st.subheader("Current Synthetic Case")
    st.write(case_data)

    run_button = st.button("Run Governed Agentic AI Workflow", type="primary")

    if run_button:
        with st.spinner(
            "Running Case Intake → RAG Retrieval → ML Risk Classification → "
            "Anomaly Detection → Explainability → LLM Reasoning → Governance..."
        ):
            from app.agents.graph import run_tpo_workflow

            final_state = run_tpo_workflow(case_data)
            final_decision = final_state["final_decision"]

        st.divider()
        render_final_decision(final_decision, final_state)

        try:
            from app.utils.visualization import anomaly_gauge

            st.divider()
            anomaly_score = final_state["anomaly_result"]["anomaly_score"]
            anomaly_detected = final_state["anomaly_result"]["anomaly_detected"]

            st.plotly_chart(
                anomaly_gauge(anomaly_detected, anomaly_score),
                use_container_width=True,
            )
        except Exception as error:
            st.warning("Anomaly gauge could not be displayed.")
            st.write(str(error))

    st.divider()
    render_dataset_monitoring()


if __name__ == "__main__":
    main()