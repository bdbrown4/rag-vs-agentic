"""
📊 Eval Dashboard — RAGAS-scored comparison of RAG vs Agentic pipelines.

Streamlit multi-page app: accessible from the sidebar navigation when
running `streamlit run app.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make project root importable when running as a page.
# Idempotent — avoids corrupting sys.modules in Streamlit's shared MPA process.
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

st.set_page_config(page_title="Eval Dashboard", layout="wide")
st.title("📊 RAGAS Evaluation Dashboard")
st.markdown(
    """
    This dashboard runs a structured evaluation of the **Classic RAG** and **Agentic** pipelines
    side-by-side using the [RAGAS](https://docs.ragas.io) framework — an LLM-assisted evaluation
    suite used in production AI systems. Each question is scored on three dimensions that together
    tell you whether a pipeline is trustworthy, useful, and efficient.

    > **How to use:** Select a question tier and how many questions to run, then click **▶ Run Evaluation**.
    > Results appear question-by-question as they complete.
    """
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Eval Settings")
    tier = st.selectbox(
        "Question tier",
        ["all", "simple", "multi-hop", "ambiguous"],
        help=(
            "**simple** — single-fact lookups\n\n"
            "**multi-hop** — require combining facts across multiple docs\n\n"
            "**ambiguous** — open-ended synthesis questions\n\n"
            "**all** — run every tier"
        ),
    )
    limit = st.slider(
        "Max questions",
        min_value=1, max_value=30, value=5,
        help="Each question calls both pipelines + RAGAS scoring. Keep low to control cost.",
    )
    st.caption(
        f"Running **{limit}** question(s) from tier **'{tier}'**. "
        "Each question costs roughly $0.01–0.05 in OpenAI tokens."
    )
    st.divider()
    run_eval = st.button("▶ Run Evaluation", type="primary", width="stretch")

# ── Session state init ────────────────────────────────────────────────────────
if "eval_df" not in st.session_state:
    st.session_state["eval_df"] = None

# ── Helpers ───────────────────────────────────────────────────────────────────
_SCORE_COLS = [
    "rag_faithfulness", "rag_answer_relevancy", "rag_context_precision",
    "agent_faithfulness", "agent_answer_relevancy", "agent_context_precision",
]
_TABLE_COLS = [
    "question_id", "tier", "question",
    "rag_faithfulness", "rag_answer_relevancy", "rag_context_precision",
    "rag_cost_usd", "rag_latency_s",
    "agent_faithfulness", "agent_answer_relevancy", "agent_context_precision",
    "agent_cost_usd", "agent_latency_s",
]
_FMT = {
    "rag_faithfulness":        "{:.3f}",
    "rag_answer_relevancy":    "{:.3f}",
    "rag_context_precision":   "{:.3f}",
    "rag_cost_usd":            "${:.5f}",
    "rag_latency_s":           "{:.2f}s",
    "agent_faithfulness":      "{:.3f}",
    "agent_answer_relevancy":  "{:.3f}",
    "agent_context_precision": "{:.3f}",
    "agent_cost_usd":          "${:.5f}",
    "agent_latency_s":         "{:.2f}s",
}


def _avg(df: pd.DataFrame, col: str) -> float:
    return round(df[col].mean(), 3) if col in df.columns and len(df) else 0.0


def _delta(df: pd.DataFrame, rag_col: str, agent_col: str) -> str | None:
    if rag_col in df.columns and agent_col in df.columns and len(df):
        d = round(df[agent_col].mean() - df[rag_col].mean(), 3)
        return f"{d:+.3f} agent vs RAG"
    return None


def _render_summary_metrics(df: pd.DataFrame):
    st.markdown(
        """
        #### 📐 RAGAS Score Averages
        These three metrics are scored by an LLM judge — the industry-standard way to evaluate
        retrieval systems without hand-labeled ground truth:

        | Metric | What it measures |
        |---|---|
        | **Faithfulness** | Does the answer contain only facts supported by the retrieved context? High = no hallucination. |
        | **Answer Relevancy** | How directly does the answer address the question? High = focused, on-topic. |
        | **Context Precision** | How much of the retrieved context was actually useful? High = efficient retrieval. |

        Scores are 0–1. A well-performing pipeline should score **> 0.7** across all three.
        The delta on agent metrics shows Agent minus RAG — positive means the agent did better.
        """
    )
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("RAG Faithfulness",    _avg(df, "rag_faithfulness"),        help="0–1: factual consistency with retrieved context")
    c2.metric("RAG Relevancy",       _avg(df, "rag_answer_relevancy"),    help="0–1: how on-topic the answer is")
    c3.metric("RAG Ctx Precision",   _avg(df, "rag_context_precision"),   help="0–1: signal-to-noise in retrieved chunks")
    c4.metric("Agent Faithfulness",  _avg(df, "agent_faithfulness"),      delta=_delta(df, "rag_faithfulness",      "agent_faithfulness"))
    c5.metric("Agent Relevancy",     _avg(df, "agent_answer_relevancy"),  delta=_delta(df, "rag_answer_relevancy",  "agent_answer_relevancy"))
    c6.metric("Agent Ctx Precision", _avg(df, "agent_context_precision"), delta=_delta(df, "rag_context_precision", "agent_context_precision"))

    st.divider()
    st.markdown(
        """
        #### 💰 Cost Summary
        Every comparison call costs real tokens. The agentic pipeline typically costs more because
        it makes multiple LLM calls per question — one per reasoning step. This section lets you
        weigh quality gains against the extra spend.
        """
    )
    rag_cost   = df["rag_cost_usd"].sum()   if "rag_cost_usd"   in df.columns else 0.0
    agent_cost = df["agent_cost_usd"].sum() if "agent_cost_usd" in df.columns else 0.0
    d1, d2, d3 = st.columns(3)
    d1.metric("RAG Total Cost",   f"${rag_cost:.4f}",   help="Cumulative OpenAI spend for RAG across all questions")
    d2.metric("Agent Total Cost", f"${agent_cost:.4f}", help="Cumulative OpenAI spend for Agentic across all questions")
    d3.metric("Total Eval Cost",  f"${rag_cost + agent_cost:.4f}")


def _render_scores_table(df: pd.DataFrame):
    st.markdown(
        """
        #### 📋 Per-Question Scores
        Each row is one question. Compare RAG and Agentic side-by-side — latency and cost columns
        reveal the efficiency trade-off behind each score. Expand **Question Details** below to read
        the actual answers that produced these numbers.
        """
    )
    avail = [c for c in _TABLE_COLS if c in df.columns]
    st.dataframe(
        df[avail].style.format({k: v for k, v in _FMT.items() if k in avail}),
        width="stretch",
        hide_index=True,
    )


def _render_question_details(df: pd.DataFrame):
    st.markdown(
        """
        #### 💬 Question Details
        Scores only make sense in context of the actual answers. Expand any question to read what
        each pipeline said — and judge for yourself whether the RAGAS scores match your intuition.
        """
    )
    tier_icon = {"simple": "🟢", "multi-hop": "🟡", "ambiguous": "🔴"}
    for _, row in df.iterrows():
        icon  = tier_icon.get(row.get("tier", ""), "⚪")
        label = f"{icon} Q{row['question_id']} [{row.get('tier','?')}] — {row['question'][:80]}{'…' if len(row['question']) > 80 else ''}"
        with st.expander(label):
            st.markdown(f"**Question:** {row['question']}")
            st.divider()
            col_rag, col_agent = st.columns(2)

            with col_rag:
                st.markdown("##### 🗂️ Classic RAG")
                st.markdown(row.get("rag_answer", "_No answer recorded_"))
                st.caption(
                    f"Faithfulness **{row.get('rag_faithfulness', 0):.3f}** · "
                    f"Relevancy **{row.get('rag_answer_relevancy', 0):.3f}** · "
                    f"Ctx Precision **{row.get('rag_context_precision', 0):.3f}**  \n"
                    f"Tokens: {row.get('rag_tokens', '—')} · "
                    f"Cost: ${row.get('rag_cost_usd', 0):.5f} · "
                    f"Latency: {row.get('rag_latency_s', 0):.2f}s · "
                    f"Confidence: {row.get('rag_confidence', 0):.2f}"
                )

            with col_agent:
                st.markdown("##### 🤖 Agentic")
                st.markdown(row.get("agent_answer", "_No answer recorded_"))
                st.caption(
                    f"Faithfulness **{row.get('agent_faithfulness', 0):.3f}** · "
                    f"Relevancy **{row.get('agent_answer_relevancy', 0):.3f}** · "
                    f"Ctx Precision **{row.get('agent_context_precision', 0):.3f}**  \n"
                    f"Tokens: {row.get('agent_tokens', '—')} · "
                    f"Cost: ${row.get('agent_cost_usd', 0):.5f} · "
                    f"Latency: {row.get('agent_latency_s', 0):.2f}s"
                )


def _render_scatter(df: pd.DataFrame):
    st.markdown(
        """
        #### 📈 Quality vs Cost Trade-off
        Each dot is one pipeline's answer to one question. **Upper-left** = high quality, low cost
        (the sweet spot). This chart makes it immediately clear whether the extra spend on the
        agentic pipeline buys meaningfully better answers — or whether RAG is good enough.
        Dot shape indicates question tier (🟢 simple / 🟡 multi-hop / 🔴 ambiguous).
        """
    )
    try:
        import altair as alt
        scatter_rows = []
        for _, row in df.iterrows():
            rag_q   = (row.get("rag_faithfulness", 0)   + row.get("rag_answer_relevancy", 0))   / 2
            agent_q = (row.get("agent_faithfulness", 0) + row.get("agent_answer_relevancy", 0)) / 2
            q_short = row["question"][:55] + ("…" if len(row["question"]) > 55 else "")
            scatter_rows.append({"pipeline": "RAG",     "question": q_short, "tier": row.get("tier", ""), "quality_avg": round(rag_q,   3), "cost_usd": row.get("rag_cost_usd",   0), "latency_s": row.get("rag_latency_s",   0)})
            scatter_rows.append({"pipeline": "Agentic", "question": q_short, "tier": row.get("tier", ""), "quality_avg": round(agent_q, 3), "cost_usd": row.get("agent_cost_usd", 0), "latency_s": row.get("agent_latency_s", 0)})
        chart = (
            alt.Chart(pd.DataFrame(scatter_rows))
            .mark_circle(size=130, opacity=0.85)
            .encode(
                x=alt.X("cost_usd:Q", title="Cost per question (USD)", scale=alt.Scale(zero=True)),
                y=alt.Y("quality_avg:Q", title="Avg Quality (Faithfulness + Relevancy) / 2", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("pipeline:N", scale=alt.Scale(domain=["RAG", "Agentic"], range=["#4C78A8", "#F58518"])),
                shape=alt.Shape("tier:N"),
                tooltip=["pipeline", "tier", "question", "quality_avg", "cost_usd", "latency_s"],
            )
            .properties(height=380)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    except ImportError:
        st.caption("Install `altair` to see the scatter chart.")


def _render_tier_breakdown(df: pd.DataFrame):
    st.markdown(
        """
        #### 🏷️ Scores by Question Tier
        Not all questions are equally hard. **Simple** questions should be easy for both pipelines.
        **Multi-hop** questions require combining facts across documents — this is where the agentic
        pipeline's iterative search typically pays off. **Ambiguous** questions require synthesis;
        expect lower scores since RAGAS has less grounding to judge against.

        A meaningful gap between RAG and Agent on *multi-hop* is the key signal of agentic value.
        """
    )
    tier_cols = [c for c in _SCORE_COLS if c in df.columns]
    if tier_cols and "tier" in df.columns and df["tier"].nunique() > 0:
        tier_agg = df.groupby("tier")[tier_cols].mean().round(3)
        st.dataframe(tier_agg, width="stretch")
    else:
        st.caption("Run questions across multiple tiers to see a breakdown here.")


def _render_full_results(df: pd.DataFrame):
    _render_summary_metrics(df)
    st.divider()
    _render_scores_table(df)
    st.divider()
    _render_question_details(df)
    st.divider()
    _render_scatter(df)
    st.divider()
    _render_tier_breakdown(df)
    st.divider()
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download full results as CSV",
        data=csv,
        file_name="ragas_eval_results.csv",
        mime="text/csv",
    )


# ── Run evaluation ────────────────────────────────────────────────────────────
if run_eval:
    try:
        from evaluate.ragas_eval import run_ragas_eval_streaming
    except ImportError as exc:
        st.error(
            f"Could not import RAGAS evaluator: {exc}\n\n"
            "Make sure you have installed: `pip install ragas>=0.2.0 datasets>=2.0.0`"
        )
        st.stop()

    questions_path = Path(__file__).parent.parent / "evaluate" / "questions.json"

    progress_bar = st.progress(0, text="Initializing…")
    status_text  = st.empty()
    live_slot    = st.empty()   # grows with live results

    rows: list[dict] = []

    try:
        gen = run_ragas_eval_streaming(
            questions_path=questions_path,
            tier=tier,
            limit=limit,
        )
        for row in gen:
            idx, total = row["_idx"], row["_total"]

            # Clear old results only after the first row arrives (run is confirmed working)
            if idx == 0:
                st.session_state["eval_df"] = None

            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            rows.append(clean)

            tier_label = f"tier: {tier}" if tier != "all" else "all tiers"
            progress_bar.progress(
                (idx + 1) / total,
                text=f"✅ Q{row['question_id']} [{row['tier']}] — {idx+1}/{total} ({tier_label})",
            )
            status_text.markdown(
                f"**Last scored:** *{row['question'][:90]}*  \n"
                f"RAG faithfulness **{row['rag_faithfulness']:.2f}** · "
                f"Agent faithfulness **{row['agent_faithfulness']:.2f}**"
            )

            with live_slot.container():
                partial_df = pd.DataFrame(rows)
                st.caption(f"Live results — {len(rows)} of {total} questions scored so far")
                _render_scores_table(partial_df)
                st.divider()
                _render_question_details(partial_df)

        # Completed — clean up live widgets and do a full re-render
        progress_bar.progress(1.0, text=f"✅ Complete — {len(rows)} questions scored")
        status_text.empty()
        live_slot.empty()
        st.session_state["eval_df"] = pd.DataFrame(rows)
        st.rerun()

    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Evaluation error: {exc}")
        if rows:
            st.session_state["eval_df"] = pd.DataFrame(rows)
            live_slot.empty()
            st.warning(f"Showing partial results — {len(rows)} questions completed before the error.")
            _render_full_results(st.session_state["eval_df"])
        st.stop()

# ── Static display (from session_state — survives reruns) ────────────────────
df = st.session_state.get("eval_df")

if df is None or len(df) == 0:
    st.info(
        "👈 Configure settings in the sidebar and click **▶ Run Evaluation** to start.\n\n"
        "Results will appear question-by-question as they complete."
    )
    st.stop()

_render_full_results(df)
