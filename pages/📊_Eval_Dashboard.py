"""
📊 Eval Dashboard — RAGAS-scored comparison of RAG vs Agentic pipelines.

Streamlit multi-page app: add this file to pages/ at the project root.
Accessible from the sidebar navigation when running `streamlit run app.py`.
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

st.set_page_config(page_title="Eval Dashboard", layout="wide")
st.title("📊 RAGAS Evaluation Dashboard")
st.caption(
    "Score both pipelines on faithfulness, answer relevancy, and context precision "
    "using the [RAGAS](https://docs.ragas.io) framework."
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Eval Settings")
    tier = st.selectbox("Question tier", ["all", "simple", "multi-hop", "ambiguous"])
    limit = st.slider("Max questions", min_value=1, max_value=30, value=5)
    st.caption(
        "Each question runs both pipelines and calls OpenAI for RAGAS scoring — "
        "set a low limit to control cost during experimentation."
    )
    run_eval = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

# ── Cached result store (persist within session) ──────────────────────────────
if "eval_df" not in st.session_state:
    st.session_state["eval_df"] = None

# ── Run evaluation ────────────────────────────────────────────────────────────
if run_eval:
    try:
        from evaluate.ragas_eval import run_ragas_eval
    except ImportError as exc:
        st.error(
            f"Could not import RAGAS evaluator: {exc}\n\n"
            "Make sure you have installed: `pip install ragas>=0.2.0 datasets>=2.0.0`"
        )
        st.stop()

    questions_path = Path(__file__).parent.parent / "evaluate" / "questions.json"

    with st.status("Running evaluation…", expanded=True) as status:
        try:
            st.write(f"Evaluating **{limit}** '{tier}' questions on both pipelines…")
            df = run_ragas_eval(
                questions_path=questions_path,
                tier=tier,
                limit=limit,
            )
            st.session_state["eval_df"] = df
            status.update(label=f"✅ Evaluation complete — {len(df)} questions scored", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="❌ Evaluation failed", state="error", expanded=True)
            st.error(str(exc))
            st.stop()

# ── Display results ───────────────────────────────────────────────────────────
df = st.session_state.get("eval_df")

if df is None:
    st.info("Configure settings in the sidebar and click **▶ Run Evaluation** to start.")
    st.stop()

import pandas as pd  # noqa: E402  (imported after availability confirmed)

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("Summary Averages")
col1, col2, col3, col4, col5, col6 = st.columns(6)

def _avg(col: str) -> float:
    return round(df[col].mean(), 3) if col in df.columns else 0.0

col1.metric("RAG Faithfulness",        _avg("rag_faithfulness"))
col2.metric("RAG Answer Relevancy",    _avg("rag_answer_relevancy"))
col3.metric("RAG Context Precision",   _avg("rag_context_precision"))
col4.metric("Agent Faithfulness",      _avg("agent_faithfulness"))
col5.metric("Agent Answer Relevancy",  _avg("agent_answer_relevancy"))
col6.metric("Agent Context Precision", _avg("agent_context_precision"))

# Cost summary
rag_total_cost   = df["rag_cost_usd"].sum()   if "rag_cost_usd"   in df.columns else 0.0
agent_total_cost = df["agent_cost_usd"].sum() if "agent_cost_usd" in df.columns else 0.0
c1, c2, c3 = st.columns(3)
c1.metric("RAG Total Cost",    f"${rag_total_cost:.4f}")
c2.metric("Agent Total Cost",  f"${agent_total_cost:.4f}")
c3.metric("Total Eval Cost",   f"${rag_total_cost + agent_total_cost:.4f}")

st.divider()

# ── Per-question results table ────────────────────────────────────────────────
st.subheader("Per-Question Results")

display_cols = [
    "question_id", "tier", "question",
    "rag_faithfulness", "rag_answer_relevancy", "rag_context_precision",
    "rag_cost_usd", "rag_latency_s",
    "agent_faithfulness", "agent_answer_relevancy", "agent_context_precision",
    "agent_cost_usd", "agent_latency_s",
]
available = [c for c in display_cols if c in df.columns]

st.dataframe(
    df[available].style.format({
        "rag_faithfulness":        "{:.3f}",
        "rag_answer_relevancy":    "{:.3f}",
        "rag_context_precision":   "{:.3f}",
        "rag_cost_usd":            "${:.5f}",
        "rag_latency_s":           "{:.1f}s",
        "agent_faithfulness":      "{:.3f}",
        "agent_answer_relevancy":  "{:.3f}",
        "agent_context_precision": "{:.3f}",
        "agent_cost_usd":          "${:.5f}",
        "agent_latency_s":         "{:.1f}s",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Quality vs Cost scatter ───────────────────────────────────────────────────
st.subheader("Quality vs Cost Trade-off")

try:
    import altair as alt

    # Build long-form data for scatter
    scatter_rows = []
    for _, row in df.iterrows():
        rag_quality   = (row.get("rag_faithfulness", 0) + row.get("rag_answer_relevancy", 0)) / 2
        agent_quality = (row.get("agent_faithfulness", 0) + row.get("agent_answer_relevancy", 0)) / 2
        scatter_rows.append({
            "pipeline": "RAG",
            "question": row["question"][:50] + "…",
            "tier": row["tier"],
            "quality_avg": round(rag_quality, 3),
            "cost_usd": row.get("rag_cost_usd", 0),
            "latency_s": row.get("rag_latency_s", 0),
        })
        scatter_rows.append({
            "pipeline": "Agentic",
            "question": row["question"][:50] + "…",
            "tier": row["tier"],
            "quality_avg": round(agent_quality, 3),
            "cost_usd": row.get("agent_cost_usd", 0),
            "latency_s": row.get("agent_latency_s", 0),
        })

    scatter_df = pd.DataFrame(scatter_rows)

    chart = (
        alt.Chart(scatter_df)
        .mark_circle(size=120)
        .encode(
            x=alt.X("cost_usd:Q", title="Cost (USD)", scale=alt.Scale(zero=True)),
            y=alt.Y("quality_avg:Q", title="Avg Quality (Faithfulness + Relevancy) / 2", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("pipeline:N", scale=alt.Scale(domain=["RAG", "Agentic"], range=["#4C78A8", "#F58518"])),
            shape=alt.Shape("tier:N"),
            tooltip=["pipeline", "tier", "question", "quality_avg", "cost_usd", "latency_s"],
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

except ImportError:
    st.caption("Install `altair` to see the scatter chart.")
    st.dataframe(scatter_df if "scatter_df" in dir() else pd.DataFrame(), use_container_width=True)

st.divider()

# ── Score bars by tier ────────────────────────────────────────────────────────
st.subheader("Scores by Question Tier")

tier_agg = (
    df.groupby("tier")[[
        "rag_faithfulness", "rag_answer_relevancy",
        "agent_faithfulness", "agent_answer_relevancy",
    ]].mean().round(3)
)
st.dataframe(tier_agg, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.divider()
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download full results as CSV",
    data=csv,
    file_name="ragas_eval_results.csv",
    mime="text/csv",
)
