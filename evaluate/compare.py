"""
Comparison runner: executes both RAG and Agentic pipelines on the same questions
and produces a side-by-side comparison report.

Usage:
    python -m evaluate.compare [--questions evaluate/questions.json] [--tier simple|multi-hop|ambiguous|all]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rag.pipeline import run_rag_pipeline
from agentic.agent import run_agentic_pipeline


def load_questions(path: str, tier: str = "all") -> list[dict]:
    """Load questions, optionally filtered by tier."""
    with open(path, "r") as f:
        data = json.load(f)

    questions = data["questions"]
    if tier != "all":
        questions = [q for q in questions if q["tier"] == tier]

    return questions


def run_comparison(questions: list[dict], verbose: bool = False) -> list[dict]:
    """Run both pipelines on each question and collect results."""
    results = []

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q{q['id']} [{q['tier'].upper()}]: {q['question']}")
        print(f"{'='*60}")

        # --- RAG ---
        print("\n--- RAG Pipeline ---")
        try:
            rag_result = run_rag_pipeline(q["question"])
            rag_data = {
                "answer": rag_result.answer,
                "llm_calls": rag_result.llm_calls,
                "total_tokens": rag_result.total_tokens,
                "latency_seconds": rag_result.latency_seconds,
                "steps": rag_result.steps,
                "chunks_used": len(rag_result.retrieved_chunks),
            }
            print(f"  Answer: {rag_result.answer[:150]}...")
            print(f"  LLM Calls: {rag_result.llm_calls} | Tokens: {rag_result.total_tokens} | Time: {rag_result.latency_seconds}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            rag_data = {"error": str(e)}

        # --- Agentic ---
        print("\n--- Agentic Pipeline ---")
        try:
            agent_result = run_agentic_pipeline(q["question"], verbose=verbose)
            agent_data = {
                "answer": agent_result.answer,
                "llm_calls": agent_result.llm_calls,
                "total_tokens": agent_result.total_tokens,
                "latency_seconds": agent_result.latency_seconds,
                "steps": agent_result.steps,
                "tool_calls": agent_result.tool_calls,
            }
            print(f"  Answer: {agent_result.answer[:150]}...")
            print(f"  LLM Calls: {agent_result.llm_calls} | Tool Calls: {len(agent_result.tool_calls)} | Time: {agent_result.latency_seconds}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            agent_data = {"error": str(e)}

        results.append({
            "question_id": q["id"],
            "tier": q["tier"],
            "question": q["question"],
            "rag": rag_data,
            "agentic": agent_data,
        })

    return results


def print_summary(results: list[dict]):
    """Print a summary comparison table."""
    print(f"\n\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Tier':<12} {'Q#':<4} {'RAG Calls':<12} {'Agent Calls':<14} {'RAG Time':<12} {'Agent Time':<12}")
    print("-" * 66)

    for r in results:
        rag = r.get("rag", {})
        agent = r.get("agentic", {})

        rag_calls = rag.get("llm_calls", "ERR")
        agent_calls = agent.get("llm_calls", "ERR")
        rag_time = f"{rag.get('latency_seconds', 0):.1f}s"
        agent_time = f"{agent.get('latency_seconds', 0):.1f}s"

        print(f"{r['tier']:<12} Q{r['question_id']:<3} {str(rag_calls):<12} {str(agent_calls):<14} {rag_time:<12} {agent_time:<12}")

    # Totals
    rag_total_time = sum(r.get("rag", {}).get("latency_seconds", 0) for r in results)
    agent_total_time = sum(r.get("agentic", {}).get("latency_seconds", 0) for r in results)
    rag_total_calls = sum(r.get("rag", {}).get("llm_calls", 0) for r in results)
    agent_total_calls = sum(r.get("agentic", {}).get("llm_calls", 0) for r in results)

    print("-" * 66)
    print(f"{'TOTAL':<12} {'':4} {str(rag_total_calls):<12} {str(agent_total_calls):<14} {rag_total_time:.1f}s{'':<7} {agent_total_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Compare RAG vs Agentic retrieval.")
    parser.add_argument("--questions", default="evaluate/questions.json", help="Path to questions file.")
    parser.add_argument("--tier", default="all", choices=["simple", "multi-hop", "ambiguous", "all"])
    parser.add_argument("--verbose", action="store_true", help="Show agent reasoning trace.")
    parser.add_argument("--output", default=None, help="Save results to JSON file.")
    args = parser.parse_args()

    questions = load_questions(args.questions, args.tier)
    print(f"Running comparison on {len(questions)} questions (tier: {args.tier})...\n")

    results = run_comparison(questions, verbose=args.verbose)
    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
