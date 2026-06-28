import joblib
import pandas as pd

from app.config import CLUSTERING_MODEL_PATH


def describe_cluster(cluster_id: int) -> str:
    """
    Human-readable interpretation of claim pattern clusters.

    These labels are based on the synthetic dataset behavior and are used
    only for explanation in this proof of concept.
    """
    cluster_descriptions = {
        0: "Routine or lower-complexity claim pattern",
        1: "Moderate treatment/utilization claim pattern",
        2: "Higher-cost or higher-review claim pattern",
    }

    return cluster_descriptions.get(
        int(cluster_id),
        "Unlabeled synthetic claim pattern",
    )


def clustering_agent(state: dict) -> dict:
    """
    Uses KMeans clustering to identify which synthetic claim pattern group
    the current case most closely resembles.
    """
    case_data = state["validated_case"]

    model = joblib.load(CLUSTERING_MODEL_PATH)

    input_df = pd.DataFrame([case_data])

    cluster_id = int(model.predict(input_df)[0])

    state["cluster_result"] = {
        "cluster_id": cluster_id,
        "cluster_description": describe_cluster(cluster_id),
        "explanation": (
            "The clustering model groups the case with similar synthetic claims "
            "based on claim amount, utilization, documentation status, diagnosis, "
            "procedure, provider type, and member risk score."
        ),
    }

    return state