from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
POLICY_DOCS_DIR = DATA_DIR / "policy_docs"

MODEL_DIR = BASE_DIR / "models"
CLASSIFIER_PATH = MODEL_DIR / "classifier.pkl"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_model.pkl"
PREPROCESSING_PATH = MODEL_DIR / "preprocessing.pkl"

CHROMA_DB_DIR = BASE_DIR.parent / "chroma_db"
CHROMA_COLLECTION_NAME = "tpo_policy_documents"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3.2:3b"

RISK_LABELS = ["Low", "Medium", "High"]
DECISION_LABELS = ["Approve", "Manual Review", "Request Documentation"]
CLUSTERING_MODEL_PATH = MODEL_DIR / "clustering_model.pkl"
TIMESERIES_ANOMALY_MODEL_PATH = MODEL_DIR / "timeseries_anomaly_model.pkl"
TIMESERIES_PREPROCESSING_PATH = MODEL_DIR / "timeseries_preprocessing.pkl"