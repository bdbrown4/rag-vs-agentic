"""
Tracing — records every pipeline run for observability.

Priority:
  1. LangSmith (if LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2 are set in env/secrets)
  2. Local JSONL fallback → data/traces.jsonl

LangSmith setup:
  Add to Streamlit secrets (or .env):
    LANGCHAIN_API_KEY = "ls__..."
    LANGCHAIN_TRACING_V2 = "true"
    LANGCHAIN_PROJECT = "rag-vs-agentic"   # optional
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_TRACES_PATH = Path(__file__).parent.parent / "data" / "traces.jsonl"


def _langsmith_enabled() -> bool:
    """True if LangSmith tracing env vars are set and the SDK is installed."""
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() != "true":
        return False
    if not os.getenv("LANGCHAIN_API_KEY", ""):
        return False
    try:
        import langsmith  # noqa: F401
        return True
    except ImportError:
        return False


def setup_tracing() -> str:
    """
    Call once at app startup.  Returns a status string for admin UI display.

    When LangSmith vars are set, LangChain auto-reports every LLM/tool call —
    no further instrumentation needed here.  We just configure env vars and
    confirm the SDK is present.
    """
    if _langsmith_enabled():
        project = os.getenv("LANGCHAIN_PROJECT", "rag-vs-agentic")
        return f"✅ LangSmith tracing active (project: '{project}')"
    return "📝 Local trace log (set LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2=true for LangSmith)"


def record_trace(
    *,
    pipeline: str,          # "rag" | "agentic" | "ab_test_rag" | "ab_test_agentic"
    question: str,
    answer: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_seconds: float = 0.0,
    tool_calls: list[dict] | None = None,
    confidence: str = "unknown",
    ab_group: str | None = None,
    extra: dict | None = None,
) -> None:
    """
    Write a trace record to the local JSONL file.

    When LangSmith is active, LangChain already captures LLM/tool details; we
    still write here for the local admin panel aggregation.
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pipeline": pipeline,
        "question": question[:300],
        "answer_preview": answer[:200],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": round(cost_usd, 6),
        "latency_s": latency_seconds,
        "tool_calls": [t.get("tool") for t in (tool_calls or [])],
        "confidence": confidence,
        "ab_group": ab_group,
        **(extra or {}),
    }
    _TRACES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TRACES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_traces(limit: int = 50) -> list[dict]:
    """Return the most recent ``limit`` trace records (newest first)."""
    if not _TRACES_PATH.exists():
        return []
    lines = _TRACES_PATH.read_text(encoding="utf-8").splitlines()
    records = []
    for line in reversed(lines):
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if len(records) >= limit:
            break
    return records


def clear_traces() -> None:
    """Wipe the local trace log."""
    if _TRACES_PATH.exists():
        _TRACES_PATH.write_text("", encoding="utf-8")


def trace_summary() -> dict:
    """Aggregate stats for admin panel display."""
    records = load_traces(limit=1000)
    if not records:
        return {"count": 0}

    import statistics

    def _agg(subset):
        if not subset:
            return {}
        costs = [r.get("cost_usd", 0) for r in subset]
        latencies = [r.get("latency_s", 0) for r in subset]
        return {
            "count": len(subset),
            "avg_cost": round(sum(costs) / len(costs), 5),
            "avg_latency": round(statistics.mean(latencies), 2),
            "total_cost": round(sum(costs), 4),
        }

    by_pipeline: dict[str, list] = {}
    for r in records:
        by_pipeline.setdefault(r.get("pipeline", "unknown"), []).append(r)

    return {
        "count": len(records),
        "by_pipeline": {k: _agg(v) for k, v in by_pipeline.items()},
        "langsmith": _langsmith_enabled(),
    }
