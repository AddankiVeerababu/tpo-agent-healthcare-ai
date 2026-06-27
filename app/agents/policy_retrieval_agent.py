from app.rag.retriever import PolicyRetriever, build_case_query


def policy_retrieval_agent(state: dict) -> dict:
    """
    Retrieves synthetic policy evidence using ChromaDB and SentenceTransformers.
    """
    case_data = state["validated_case"]

    query = build_case_query(case_data)

    retriever = PolicyRetriever()
    policy_evidence = retriever.retrieve(query=query, top_k=3)

    state["policy_evidence"] = policy_evidence

    return state