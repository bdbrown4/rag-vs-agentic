"""
Guardrails — Pydantic output schemas and confidence gating.

Enforces structured, validated outputs from both pipelines and prevents
returning an answer when the system isn't confident enough to be useful.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Output Schemas ─────────────────────────────────────────────────────────────

class RAGAnswer(BaseModel):
    """Validated output from the RAG pipeline."""

    answer: str = Field(
        ...,
        description="Direct answer to the question based ONLY on the provided context.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Repository or file names the answer draws from.",
    )
    confidence: str = Field(
        ...,
        description=(
            "One of: 'high' (context clearly answers it), "
            "'medium' (partial info), "
            "'low' (context barely relevant), "
            "'insufficient-context' (cannot answer from context)."
        ),
    )
    uncertainty_note: str | None = Field(
        None,
        description="If confidence is low or insufficient-context, explain what information is missing.",
    )


class AgenticAnswer(BaseModel):
    """Validated output from the Agentic pipeline."""

    answer: str = Field(
        ...,
        description="Complete answer to the question based on all retrieved information.",
    )
    reasoning_summary: str = Field(
        ...,
        description="1-2 sentence summary of the reasoning steps and tools used.",
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Names of tools called during retrieval.",
    )
    confidence: str = Field(
        ...,
        description=(
            "One of: 'high', 'medium', 'low', 'insufficient-context'."
        ),
    )
    uncertainty_note: str | None = Field(
        None,
        description="If confidence is low, explain why.",
    )


# ── Confidence Gating ─────────────────────────────────────────────────────────

#: Below this retrieval distance the system won't even attempt LLM generation.
CONFIDENCE_GATE_THRESHOLD = 0.25


def should_gate(retrieval_confidence: float) -> bool:
    """Return True if retrieval quality is so poor we should skip LLM generation."""
    return retrieval_confidence < CONFIDENCE_GATE_THRESHOLD


GATED_ANSWER = (
    "⚠️ **Insufficient context** — the knowledge base did not contain enough "
    "relevant information to answer this question reliably. "
    "Try rephrasing or asking about a specific project."
)


def confidence_from_schema(schema_confidence: str) -> float:
    """Map schema confidence label to a 0-1 float."""
    return {"high": 0.9, "medium": 0.6, "low": 0.35, "insufficient-context": 0.1}.get(
        schema_confidence, 0.5
    )
