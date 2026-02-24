"""
RAGAS evaluation runner for RAG vs Agentic pipelines.

Measures:
  - faithfulness          : factual consistency of the answer with the retrieved context
  - answer_relevancy      : how relevant the answer is to the question
  - context_precision     : proportion of retrieved context that was actually useful

Usage (CLI):
    python -m evaluate.ragas_eval [--tier simple|multi-hop|ambiguous|all] [--limit N] [--output results.csv]

Usage (import):
    from evaluate.ragas_eval import run_ragas_eval
    df = run_ragas_eval(tier="simple", limit=5)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv()

# ── Lazy imports (heavy) ──────────────────────────────────────────────────────

def _import_ragas():
    """Import RAGAS lazily to avoid slowing down other modules."""
    try:
        from ragas import evaluate as ragas_evaluate
        from datasets import Dataset
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        # RAGAS >=0.2 requires explicitly-wired LLM and embeddings wrappers.
        # Without them ResponseRelevancy falls back to its own OpenAIEmbeddings
        # stub that is missing .embed_query().
        ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        ragas_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

        try:
            from ragas.metrics import LLMContextPrecisionWithoutReference
            ctx_prec = LLMContextPrecisionWithoutReference(llm=ragas_llm)
        except ImportError:
            from ragas.metrics import ContextPrecision
            ctx_prec = ContextPrecision(llm=ragas_llm)

        metrics = [
            Faithfulness(llm=ragas_llm),
            ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            ctx_prec,
        ]
        return ragas_evaluate, metrics, Dataset
    except ImportError as exc:
        raise ImportError(
            "RAGAS is not installed. Run: pip install ragas>=0.2.0 datasets>=2.0.0"
        ) from exc


# ── Data loading ──────────────────────────────────────────────────────────────

def load_questions(path: str | Path, tier: str = "all", limit: int | None = None) -> list[dict]:
    """Load evaluation questions filtered by tier."""
    with open(path, "r") as f:
        data = json.load(f)
    questions = data["questions"]
    if tier != "all":
        questions = [q for q in questions if q["tier"] == tier]
    if limit:
        questions = questions[:limit]
    return questions


# ── Pipeline runners ──────────────────────────────────────────────────────────

def _run_rag(question: str, n_results: int = 5) -> dict:
    """Run RAG pipeline and return RAGAS-compatible record."""
    from rag.pipeline import run_rag_pipeline
    result = run_rag_pipeline(question, n_results=n_results)
    contexts = [chunk["text"] for chunk in result.retrieved_chunks]
    return {
        "question": question,
        "answer": result.answer,
        "contexts": contexts,
        # perf fields
        "llm_calls": result.llm_calls,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
        "confidence": result.confidence,
        "latency_seconds": result.latency_seconds,
    }


def _run_agentic(question: str, max_iterations: int = 8) -> dict:
    """Run Agentic pipeline and return RAGAS-compatible record.

    For the agentic case 'contexts' is the list of tool-call output previews —
    the information the agent actually used to form its answer.
    """
    from agentic.agent import run_agentic_pipeline
    result = run_agentic_pipeline(question, max_iterations=max_iterations, verbose=False)
    # Use tool output previews as the context that informed the answer
    contexts = [tc.get("output_preview", "") for tc in result.tool_calls] or ["(no tools called)"]
    return {
        "question": question,
        "answer": result.answer,
        "contexts": contexts,
        # perf fields
        "llm_calls": result.llm_calls,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
        "latency_seconds": result.latency_seconds,
    }


# ── RAGAS scoring ─────────────────────────────────────────────────────────────

def _get(row: "pd.Series", *keys: str, default: float = 0.0) -> float:
    """Return the first key found in a pandas Series, supporting multiple name variants."""
    for k in keys:
        if k in row.index and row[k] == row[k]:  # also skips NaN
            return float(row[k])
    return default


def _score_pipeline(records: list[dict]) -> pd.DataFrame:
    """Score a list of pipeline records with RAGAS metrics."""
    ragas_evaluate, metrics, Dataset = _import_ragas()

    dataset = Dataset.from_dict({
        "question":  [r["question"] for r in records],
        "answer":    [r["answer"] for r in records],
        "contexts":  [r["contexts"] for r in records],
    })

    scores = ragas_evaluate(dataset, metrics=metrics)

    # RAGAS >=0.2 returns an EvaluationResult; <=0.1 returned a dict-like.
    if hasattr(scores, "to_pandas"):
        score_df = scores.to_pandas()
    else:
        score_df = pd.DataFrame(scores)
    return score_df


def _score_single(rec: dict) -> dict:
    """Score a single pipeline record; returns a flat dict of metric scores."""
    ragas_evaluate, metrics, Dataset = _import_ragas()
    dataset = Dataset.from_dict({
        "question": [rec["question"]],
        "answer":   [rec["answer"]],
        "contexts": [rec["contexts"]],
    })
    scores = ragas_evaluate(dataset, metrics=metrics)
    if hasattr(scores, "to_pandas"):
        row = scores.to_pandas().iloc[0]
    else:
        row = pd.DataFrame(scores).iloc[0]
    return row  # pandas Series — use _get() to read values


def run_ragas_eval_streaming(
    questions_path: str | Path = Path(__file__).parent / "questions.json",
    tier: str = "all",
    limit: int | None = None,
    rag_n_results: int = 5,
    agent_max_iterations: int = 8,
):
    """
    Generator version of run_ragas_eval.

    Yields one result dict per question as soon as it is scored, so callers
    can display live progress (e.g. Streamlit st.dataframe updates).

    Each yielded dict has the same keys as a row from run_ragas_eval(), plus
    a ``"_total"`` key on the first yield indicating how many questions will run.
    """
    questions = load_questions(questions_path, tier=tier, limit=limit)
    total = len(questions)

    for idx, q in enumerate(questions):
        # ── Run RAG ──────────────────────────────────────────────────────────
        try:
            rag_rec = _run_rag(q["question"], n_results=rag_n_results)
        except Exception as exc:
            rag_rec = {
                "question": q["question"], "answer": f"ERROR: {exc}",
                "contexts": [], "llm_calls": 0, "total_tokens": 0,
                "cost_usd": 0.0, "confidence": 0.0, "latency_seconds": 0.0,
            }

        # ── Run Agentic ───────────────────────────────────────────────────────
        try:
            agent_rec = _run_agentic(q["question"], max_iterations=agent_max_iterations)
        except Exception as exc:
            agent_rec = {
                "question": q["question"], "answer": f"ERROR: {exc}",
                "contexts": ["(error)"], "llm_calls": 0, "total_tokens": 0,
                "cost_usd": 0.0, "latency_seconds": 0.0,
            }

        # ── RAGAS score (1-item datasets — fast, immediate) ───────────────────
        try:
            rag_scores  = _score_single(rag_rec)
            agent_scores = _score_single(agent_rec)
        except Exception as exc:
            # Scoring failed — yield zeros so the UI still updates
            import traceback
            print(f"RAGAS scoring error Q{q['id']}: {traceback.format_exc()}")
            empty = pd.Series(dtype=float)
            rag_scores = agent_scores = empty

        row = {
            "question_id": q["id"],
            "tier":        q["tier"],
            "question":    q["question"],
            # RAG
            "rag_faithfulness":      round(_get(rag_scores,   "faithfulness"), 3),
            "rag_answer_relevancy":  round(_get(rag_scores,   "answer_relevancy", "response_relevancy"), 3),
            "rag_context_precision": round(_get(rag_scores,   "llm_context_precision_without_reference", "context_precision"), 3),
            "rag_tokens":            rag_rec["total_tokens"],
            "rag_cost_usd":          round(rag_rec["cost_usd"], 6),
            "rag_confidence":        round(rag_rec.get("confidence", 0.0), 3),
            "rag_latency_s":         rag_rec["latency_seconds"],
            # Agentic
            "agent_faithfulness":      round(_get(agent_scores, "faithfulness"), 3),
            "agent_answer_relevancy":  round(_get(agent_scores, "answer_relevancy", "response_relevancy"), 3),
            "agent_context_precision": round(_get(agent_scores, "llm_context_precision_without_reference", "context_precision"), 3),
            "agent_tokens":            agent_rec["total_tokens"],
            "agent_cost_usd":          round(agent_rec["cost_usd"], 6),
            "agent_latency_s":         agent_rec["latency_seconds"],
            # Metadata for progress tracking
            "_idx":   idx,
            "_total": total,
        }
        yield row


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_ragas_eval(
    questions_path: str | Path = Path(__file__).parent / "questions.json",
    tier: str = "all",
    limit: int | None = None,
    rag_n_results: int = 5,
    agent_max_iterations: int = 8,
) -> pd.DataFrame:
    """
    Run RAGAS evaluation on both pipelines and return a merged DataFrame.
    Delegates to run_ragas_eval_streaming and collects all rows.
    """
    rows = []
    for row in run_ragas_eval_streaming(
        questions_path=questions_path,
        tier=tier,
        limit=limit,
        rag_n_results=rag_n_results,
        agent_max_iterations=agent_max_iterations,
    ):
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        rows.append(clean)
        print(f"  Q{row['question_id']} scored ({row['_idx']+1}/{row['_total']})")
    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation — RAG vs Agentic")
    parser.add_argument("--questions", default=str(Path(__file__).parent / "questions.json"))
    parser.add_argument("--tier", default="all", choices=["all", "simple", "multi-hop", "ambiguous"])
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--output", default=None, help="Save CSV to this path")
    args = parser.parse_args()

    df = run_ragas_eval(
        questions_path=args.questions,
        tier=args.tier,
        limit=args.limit,
    )

    # Print summary
    print("\n\n=== RAGAS Evaluation Results ===\n")
    numeric_cols = [c for c in df.columns if any(
        c.endswith(s) for s in ["faithfulness", "relevancy", "precision", "cost_usd", "latency_s", "tokens"]
    )]
    print(df[["question_id", "tier"] + numeric_cols].to_string(index=False))

    # Averages
    print("\n=== Averages ===")
    for col in ["rag_faithfulness", "rag_answer_relevancy", "rag_context_precision",
                "agent_faithfulness", "agent_answer_relevancy", "agent_context_precision"]:
        if col in df.columns:
            print(f"  {col}: {df[col].mean():.3f}")

    total_cost = df["rag_cost_usd"].sum() + df["agent_cost_usd"].sum()
    print(f"\n  Total evaluation cost: ${total_cost:.4f}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
