"""
Chunking and embedding utilities.

This module handles:
1. Loading documents (markdown/text)
2. Splitting them into chunks
3. Generating embeddings via OpenAI text-embedding-3-small
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


def get_embeddings_model() -> OpenAIEmbeddings:
    """Return the OpenAI embeddings model (uses OPENAI_API_KEY from env)."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_text_splitter(chunk_size: int = 500, chunk_overlap: int = 100) -> RecursiveCharacterTextSplitter:
    """
    Return a text splitter tuned for README-sized documents.

    Args:
        chunk_size: Max characters per chunk (default 500 — small for READMEs).
        chunk_overlap: Overlap between chunks to preserve context.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split a list of documents into chunks.

    Args:
        documents: List of dicts with keys 'id', 'text', and 'metadata'.

    Returns:
        List of chunk dicts with 'id', 'text', and 'metadata' (including source doc id).
    """
    splitter = get_text_splitter()
    chunks = []

    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "id": f"{doc['id']}_chunk_{i}",
                "text": chunk_text,
                "metadata": {
                    **doc.get("metadata", {}),
                    "source_doc_id": doc["id"],
                    "chunk_index": i,
                },
            })

    return chunks
