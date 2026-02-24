"""
Classic RAG Pipeline.

Flow:
    User query → embed → retrieve top-k chunks from ChromaDB → inject as context → single LLM call → answer

This is the "traditional" approach: one retrieval pass, one generation pass, no iteration.
"""

import time
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

from shared.vector_store import query_similar
from shared.metrics import estimate_cost, compute_confidence
from shared.guardrails import (
    RAGAnswer,
    should_gate,
    GATED_ANSWER,
    confidence_from_schema,
)


@dataclass
class RAGResult:
    """Result from the RAG pipeline, including trace info for comparison."""
    question: str
    answer: str
    retrieved_chunks: list[dict]
    llm_calls: int = 1
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    confidence: float = 0.0   # retrieval confidence: avg cosine similarity of top chunks
    latency_seconds: float = 0.0
    steps: list[str] = field(default_factory=list)
    uncertainty_note: str | None = None   # from Pydantic guardrails schema


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

    # Compute retrieval confidence before generation
    confidence = compute_confidence([c["distance"] for c in chunks])

    # Guardrail gate: if retrieval quality is too low, skip LLM call entirely
    if should_gate(confidence):
        steps.append("GATED: Retrieval confidence too low — skipping generation")
        elapsed = time.time() - start
        return RAGResult(
            question=question,
            answer=GATED_ANSWER,
            retrieved_chunks=chunks,
            llm_calls=0,
            confidence=confidence,
            latency_seconds=round(elapsed, 2),
            steps=steps,
            uncertainty_note="Retrieval confidence below threshold — answer suppressed.",
        )

    # Step 3: Generate with structured output (Pydantic guardrails)
    steps.append("GENERATE: Sending prompt to LLM with structured output schema")
    llm = ChatOpenAI(model=model, temperature=0)
    structured_llm = llm.with_structured_output(RAGAnswer)

    structured_prompt = (
        RAG_SYSTEM_PROMPT
        + "\n\nYou MUST return a structured response with: answer, sources list, "
        + "confidence label (high/medium/low/insufficient-context), and optional uncertainty_note."
    )

    answer = ""
    uncertainty_note: str | None = None

    with get_openai_callback() as cb:
        try:
            schema_response: RAGAnswer = structured_llm.invoke([
                {"role": "system", "content": structured_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ])
            answer = schema_response.answer
            # Blend retrieval confidence with schema's self-reported confidence
            schema_conf = confidence_from_schema(schema_response.confidence)
            if schema_conf > 0.1:
                confidence = max(confidence, schema_conf)
            uncertainty_note = schema_response.uncertainty_note
        except Exception:
            # Structured output failed — fall back to plain text
            fallback = llm.invoke([
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ])
            answer = fallback.content

    prompt_tokens = cb.prompt_tokens
    completion_tokens = cb.completion_tokens
    total_tokens = cb.total_tokens
    cost_usd = estimate_cost(model, prompt_tokens, completion_tokens)
    steps.append(f"ANSWER: Generated structured response ({total_tokens} tokens, ${cost_usd:.4f})")

    elapsed = time.time() - start

    return RAGResult(
        question=question,
        answer=answer,
        retrieved_chunks=chunks,
        llm_calls=1,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        confidence=confidence,
        latency_seconds=round(elapsed, 2),
        steps=steps,
        uncertainty_note=uncertainty_note,
    )
