"""
ChromaDB vector store setup and operations.

Provides a thin wrapper around ChromaDB for:
- Creating/loading a persistent collection
- Adding chunked documents with embeddings
- Querying by text similarity
"""

import chromadb
from chromadb.config import Settings

from shared.embeddings import get_embeddings_model


_client: chromadb.ClientAPI | None = None
COLLECTION_NAME = "github_readmes"
PERSIST_DIR = ".chroma"


def get_client() -> chromadb.ClientAPI:
    """Get or create a persistent ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=PERSIST_DIR)
    return _client


def get_or_create_collection(name: str = COLLECTION_NAME) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(chunks: list[dict], collection_name: str = COLLECTION_NAME) -> int:
    """
    Embed and add document chunks to ChromaDB.

    Args:
        chunks: List of dicts with 'id', 'text', and 'metadata'.
        collection_name: Target collection name.

    Returns:
        Number of chunks added.
    """
    if not chunks:
        return 0

    collection = get_or_create_collection(collection_name)
    embeddings_model = get_embeddings_model()

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Generate embeddings via OpenAI
    embeddings = embeddings_model.embed_documents(texts)

    # Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def query_similar(
    query_text: str,
    n_results: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """
    Query ChromaDB for similar chunks.

    Args:
        query_text: The search query.
        n_results: Number of results to return.
        collection_name: Collection to search.

    Returns:
        List of dicts with 'id', 'text', 'metadata', and 'distance'.
    """
    collection = get_or_create_collection(collection_name)
    embeddings_model = get_embeddings_model()

    query_embedding = embeddings_model.embed_query(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return output


def list_all_documents(collection_name: str = COLLECTION_NAME) -> list[str]:
    """Return all unique source document IDs in the collection."""
    collection = get_or_create_collection(collection_name)
    all_items = collection.get(include=["metadatas"])

    doc_ids = set()
    for meta in all_items["metadatas"]:
        if "source_doc_id" in meta:
            doc_ids.add(meta["source_doc_id"])

    return sorted(doc_ids)


def get_document_chunks(doc_id: str, collection_name: str = COLLECTION_NAME) -> list[dict]:
    """Retrieve all chunks belonging to a specific source document."""
    collection = get_or_create_collection(collection_name)

    results = collection.get(
        where={"source_doc_id": doc_id},
        include=["documents", "metadatas"],
    )

    chunks = []
    for i in range(len(results["ids"])):
        chunks.append({
            "id": results["ids"][i],
            "text": results["documents"][i],
            "metadata": results["metadatas"][i],
        })

    # Sort by chunk index
    chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
    return chunks
