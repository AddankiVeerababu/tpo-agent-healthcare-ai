import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def risk_distribution_chart(df: pd.DataFrame):
    counts = df["risk_level"].value_counts().reset_index()
    counts.columns = ["risk_level", "count"]

    fig = px.bar(
        counts,
        x="risk_level",
        y="count",
        title="Synthetic Claims by Risk Level",
        text="count",
    )
    fig.update_layout(xaxis_title="Risk Level", yaxis_title="Number of Claims")
    return fig


def claim_amount_by_procedure_chart(df: pd.DataFrame):
    fig = px.box(
        df,
        x="procedure",
        y="claim_amount",
        color="risk_level",
        title="Claim Amount Distribution by Procedure",
    )
    fig.update_layout(xaxis_title="Procedure", yaxis_title="Claim Amount")
    return fig


def anomaly_gauge(anomaly_detected: bool, anomaly_score: float):
    value = abs(float(anomaly_score))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "Anomaly Signal Strength"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "darkred" if anomaly_detected else "green"},
                "steps": [
                    {"range": [0, 0.35], "color": "lightgreen"},
                    {"range": [0.35, 0.70], "color": "khaki"},
                    {"range": [0.70, 1], "color": "salmon"},
                ],
            },
        )
    )
    return fig