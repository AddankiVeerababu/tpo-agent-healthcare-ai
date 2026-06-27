import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


class PolicyRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.collection = self.client.get_collection(name=CHROMA_COLLECTION_NAME)

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode([query]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved_docs = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, metadata, distance in zip(documents, metadatas, distances):
            relevance_score = 1 / (1 + float(distance))

            retrieved_docs.append(
                {
                    "source": metadata.get("source", "unknown_policy"),
                    "text": doc,
                    "relevance_score": round(relevance_score, 4),
                }
            )

        return retrieved_docs


def build_case_query(case_data: dict) -> str:
    return (
        f"Diagnosis: {case_data.get('diagnosis')}. "
        f"Procedure: {case_data.get('procedure')}. "
        f"Claim amount: {case_data.get('claim_amount')}. "
        f"Prior visits in last 30 days: {case_data.get('prior_visits_30d')}. "
        f"Documentation complete: {case_data.get('documentation_complete')}. "
        f"Provider type: {case_data.get('provider_type')}. "
        f"Claim type: {case_data.get('claim_type')}."
    )


if __name__ == "__main__":
    retriever = PolicyRetriever()

    sample_case = {
        "diagnosis": "Type 2 diabetes with neuropathy",
        "procedure": "Wound care",
        "claim_amount": 1250,
        "prior_visits_30d": 6,
        "documentation_complete": "No",
        "provider_type": "Outpatient clinic",
        "claim_type": "Payment",
    }

    query = build_case_query(sample_case)
    docs = retriever.retrieve(query, top_k=3)

    print("\nRetrieved Policy Evidence")
    print("-------------------------")
    for doc in docs:
        print(f"\nSource: {doc['source']}")
        print(f"Relevance: {doc['relevance_score']}")
        print(doc["text"][:500])