"""
Classic RAG Pipeline.

Flow:
    User query → embed → retrieve top-k chunks from ChromaDB → inject as context → single LLM call → answer

This is the "traditional" approach: one retrieval pass, one generation pass, no iteration.
"""

import time
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI

from shared.vector_store import query_similar


@dataclass
class RAGResult:
    """Result from the RAG pipeline, including trace info for comparison."""
    question: str
    answer: str
    retrieved_chunks: list[dict]
    llm_calls: int = 1
    total_tokens: int = 0
    latency_seconds: float = 0.0
    steps: list[str] = field(default_factory=list)


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("repo_name", "unknown")
        context_parts.append(
            f"[Source {i}: {source}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


RAG_SYSTEM_PROMPT = """You are a helpful assistant answering questions about a developer's GitHub portfolio.
You have been given relevant excerpts from their repository READMEs.
Answer the question based ONLY on the provided context. If the context doesn't contain enough information, say so.
Be specific and cite which repository/project your answer comes from when possible."""


def run_rag_pipeline(
    question: str,
    n_results: int = 5,
    model: str = "gpt-4o",
) -> RAGResult:
    """
    Execute the classic RAG pipeline.

    Args:
        question: User's question.
        n_results: Number of chunks to retrieve.
        model: OpenAI model name.

    Returns:
        RAGResult with answer, retrieved chunks, and performance metrics.
    """
    start = time.time()
    steps = []

    # Step 1: Retrieve
    steps.append(f"RETRIEVE: Searching for top {n_results} chunks matching: '{question}'")
    chunks = query_similar(question, n_results=n_results)
    steps.append(f"RETRIEVED: Got {len(chunks)} chunks from repos: {[c['metadata'].get('repo_name', '?') for c in chunks]}")

    # Step 2: Build context
    context = build_context(chunks)
    steps.append(f"CONTEXT: Built context from {len(chunks)} chunks ({len(context)} chars)")

    # Step 3: Generate
    steps.append("GENERATE: Sending single prompt to LLM")
    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke([
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ])

    answer = response.content
    total_tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
    steps.append(f"ANSWER: Generated response ({total_tokens} tokens)")

    elapsed = time.time() - start

    return RAGResult(
        question=question,
        answer=answer,
        retrieved_chunks=chunks,
        llm_calls=1,
        total_tokens=total_tokens,
        latency_seconds=round(elapsed, 2),
        steps=steps,
    )
