"""
Shared utilities: cost estimation, retrieval confidence, and query audit logging.

Cost:    estimate_cost(model, prompt_tokens, completion_tokens) → USD float
Confidence: compute_confidence(distances) → 0-1 float
Logging:    log_query(entry) / load_query_log() → JSONL file in data/
"""

import json
from datetime import datetime, timezone
from pathlib import Path

# ── OpenAI pricing per 1M tokens (Feb 2026) ──────────────────────────────────
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":           {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":      {"input": 10.00, "output": 30.00},
    "o1":               {"input": 15.00, "output": 60.00},
    "o1-mini":          {"input": 1.10,  "output": 4.40},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "text-embedding-3-large": {"input": 0.13, "output": 0.00},
}
_DEFAULT_PRICE = _PRICING["gpt-4o"]

QUERY_LOG_PATH = Path(__file__).parent.parent / "data" / "query_log.jsonl"


# ── Cost estimation ───────────────────────────────────────────────────────────

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated USD cost for a given model and token counts."""
    p = _PRICING.get(model, _DEFAULT_PRICE)
    cost = (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000
    return round(cost, 6)


def format_cost(cost_usd: float) -> str:
    """Format a USD cost for display (e.g. '$0.0042' or '<$0.0001')."""
    if cost_usd == 0:
        return "$0.00"
    if cost_usd < 0.0001:
        return "<$0.0001"
    return f"${cost_usd:.4f}"


# ── Retrieval confidence ──────────────────────────────────────────────────────

def compute_confidence(distances: list[float]) -> float:
    """
    Convert cosine distances from ChromaDB into a 0-1 confidence score.

    ChromaDB cosine distance: 0.0 = identical, 1.0 = orthogonal, 2.0 = opposite.
    We convert to similarity (1 - distance) and average across retrieved chunks.
    Higher = more confident the retrieved context is relevant to the query.
    """
    if not distances:
        return 0.0
    avg_similarity = 1.0 - (sum(distances) / len(distances))
    return round(max(0.0, min(1.0, avg_similarity)), 3)


def confidence_label(confidence: float) -> tuple[str, str]:
    """Return (emoji, label) for a confidence score."""
    if confidence >= 0.75:
        return "🟢", f"High ({confidence:.0%})"
    if confidence >= 0.55:
        return "🟡", f"Medium ({confidence:.0%})"
    return "🔴", f"Low ({confidence:.0%})"


# ── Query audit log ───────────────────────────────────────────────────────────

def log_query(entry: dict) -> None:
    """Append a query log entry as a JSONL line to data/query_log.jsonl."""
    QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {**entry, "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(QUERY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_query_log(max_entries: int = 200) -> list[dict]:
    """Load the most recent query log entries, newest first."""
    if not QUERY_LOG_PATH.exists():
        return []
    try:
        raw = QUERY_LOG_PATH.read_text(encoding="utf-8").strip()
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        entries = []
        for line in lines[-max_entries:]:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
        return list(reversed(entries))  # newest first
    except Exception:
        return []


def clear_query_log() -> None:
    """Delete the query log file."""
    if QUERY_LOG_PATH.exists():
        QUERY_LOG_PATH.unlink()
