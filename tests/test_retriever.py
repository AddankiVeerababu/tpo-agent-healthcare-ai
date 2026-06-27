import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.rag.retriever import PolicyRetriever, build_case_query


def test_build_case_query_contains_key_terms():
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

    assert "Wound care" in query
    assert "Payment" in query
    assert "Documentation complete" in query


def test_policy_retriever_returns_results():
    retriever = PolicyRetriever()

    docs = retriever.retrieve(
        "wound care missing documentation high prior visits payment review",
        top_k=2,
    )

    assert len(docs) > 0
    assert "source" in docs[0]
    assert "text" in docs[0]
    assert "relevance_score" in docs[0]