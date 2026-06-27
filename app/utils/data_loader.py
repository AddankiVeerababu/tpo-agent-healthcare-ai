import pandas as pd

from app.config import DATA_DIR


def load_claims_data() -> pd.DataFrame:
    """
    Load the synthetic healthcare claims dataset.

    This project uses synthetic data only.
    No real patient data or PHI is used.
    """
    data_path = DATA_DIR / "synthetic_claims.csv"
    return pd.read_csv(data_path)


def get_feature_columns() -> list[str]:
    """
    Features used by the ML models.
    """
    return [
        "patient_age",
        "diagnosis",
        "procedure",
        "claim_amount",
        "prior_visits_30d",
        "documentation_complete",
        "provider_type",
        "length_of_stay_days",
        "member_risk_score",
        "claim_type",
    ]


def get_target_column() -> str:
    """
    Target label for risk classification.
    """
    return "risk_level"


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into X and y.
    """
    X = df[get_feature_columns()]
    y = df[get_target_column()]
    return X, y