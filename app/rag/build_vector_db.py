import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import (
    POLICY_DOCS_DIR,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


def load_policy_documents():
    documents = []
    metadatas = []
    ids = []

    policy_files = sorted(POLICY_DOCS_DIR.glob("*.txt"))

    for idx, file_path in enumerate(policy_files):
        text = file_path.read_text(encoding="utf-8")

        documents.append(text)
        metadatas.append(
            {
                "source": file_path.name,
                "document_type": "synthetic_policy",
                "privacy": "synthetic_only_no_phi",
            }
        )
        ids.append(f"policy-{idx + 1}")

    return documents, metadatas, ids


def build_vector_db():
    print("Loading synthetic policy documents...")
    documents, metadatas, ids = load_policy_documents()

    if not documents:
        raise ValueError("No policy documents found in app/data/policy_docs")

    print(f"Loaded {len(documents)} policy documents.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Creating embeddings...")
    embeddings = embedding_model.encode(documents).tolist()

    print(f"Connecting to ChromaDB at: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    existing_collections = [collection.name for collection in client.list_collections()]
    if CHROMA_COLLECTION_NAME in existing_collections:
        print(f"Deleting existing collection: {CHROMA_COLLECTION_NAME}")
        client.delete_collection(CHROMA_COLLECTION_NAME)

    collection = client.create_collection(name=CHROMA_COLLECTION_NAME)

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"Created ChromaDB collection: {CHROMA_COLLECTION_NAME}")
    print("Vector database build complete.")
    print("Synthetic policy documents only. No PHI was used.")


if __name__ == "__main__":
    build_vector_db()