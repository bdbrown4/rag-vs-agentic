"""
Model Cost vs Quality Trade-off
================================
One of the most important engineering decisions in any AI product is:
"Which model do I use?"

Bigger models (GPT-4o) give better answers but cost 15–60× more than smaller ones (GPT-4o-mini).
This page runs the SAME question through multiple models, scores each answer with RAGAS metrics,
and shows you whether the quality difference justifies the price difference.

It also explains what *fine-tuning* is and when you'd use it — a common interview topic.
"""

import time
import streamlit as st

st.set_page_config(
    page_title="Model Comparison — RAG vs Agentic",
    page_icon="🧪",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧪 Model Comparison")
    st.caption("Run the same question through multiple OpenAI models and compare quality vs cost")

    st.markdown("---")
    st.markdown("### Model Pricing Guide")
    st.markdown(
        """
| Model | Input | Output |
|-------|-------|--------|
| gpt-4o | $2.50/M | $10.00/M |
| gpt-4o-mini | $0.15/M | $0.60/M |
| gpt-3.5-turbo | $0.50/M | $1.50/M |

*Prices per million **tokens** (≈ ¾ of a word each)*

**Rule of thumb:**
gpt-4o-mini is **~17× cheaper** than gpt-4o on input tokens.

The question this page answers: *is gpt-4o's quality worth that premium?*
        """
    )

    st.markdown("---")
    st.markdown("### What is Fine-Tuning?")
    st.markdown(
        """
Imagine teaching a very smart but general intern (GPT-4o) versus a specialist who spent
6 months learning your exact domain (a fine-tuned GPT-4o-mini).

**Fine-tuning** trains a smaller, cheaper model on *your* question-answer pairs so it learns
your specific vocabulary, style, and content.

**Goal:** get gpt-4o *quality* at gpt-4o-mini *price*.

This page shows you the current gap — giving you the data to decide if fine-tuning would
be worth the engineering effort.
        """
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🧪 Model Cost vs Quality Trade-off")
st.markdown(
    """
**What you're about to do:** Ask the same portfolio question to multiple OpenAI models.
Each answer is automatically scored for **faithfulness** (did it hallucinate?) and
**relevancy** (did it actually answer the question?).

The result is a data-driven decision: *which model gives the best bang for the buck?*

> **For the non-technical interviewer:** This is like running the same task past three
> different employees (a junior, a mid-level, and a senior) and comparing output quality
> to salary. The cheapest employee isn't always worse — sometimes they're 90% as good at
> 15% of the cost.
"""
)

st.divider()

# ── Configuration ─────────────────────────────────────────────────────────────

st.subheader("⚙️ Configure Comparison")

PRESET_QUESTIONS = [
    "What programming languages does Luke know?",
    "Describe Luke's experience with mobile development.",
    "What web frameworks has Luke used in his projects?",
    "Has Luke worked with cloud services or APIs?",
    "What is the most technically complex project in Luke's portfolio?",
]

col_q, col_m = st.columns([2, 1])

with col_q:
    use_custom = st.checkbox("Enter a custom question")
    if use_custom:
        question = st.text_input("Your question:", placeholder="Ask something about Luke's portfolio…")
    else:
        question = st.selectbox("Choose a preset question:", PRESET_QUESTIONS)

with col_m:
    st.markdown("**Models to compare:**")
    use_4o      = st.checkbox("GPT-4o (flagship, most capable)", value=True)
    use_mini    = st.checkbox("GPT-4o-mini (fast, cheapest)", value=True)
    use_35turbo = st.checkbox("GPT-3.5-turbo (legacy, mid-tier)", value=False)

    models_to_run: list[tuple[str, str]] = []
    if use_4o:
        models_to_run.append(("gpt-4o", "GPT-4o"))
    if use_mini:
        models_to_run.append(("gpt-4o-mini", "GPT-4o-mini"))
    if use_35turbo:
        models_to_run.append(("gpt-3.5-turbo", "GPT-3.5-turbo"))

pipeline_type = st.radio(
    "Pipeline type for this comparison:",
    ["RAG", "Agentic"],
    horizontal=True,
    help="RAG retrieves context once. Agentic can search multiple times and plan.",
)

run_btn = st.button(
    "🚀 Run Comparison",
    type="primary",
    disabled=not question or not models_to_run,
)

if not question:
    st.info("Select or enter a question above to begin.")

if not models_to_run:
    st.warning("Select at least one model.")

st.divider()

# ── Run comparison ─────────────────────────────────────────────────────────────

MODEL_COSTS = {
    # (input_per_million_tokens, output_per_million_tokens)
    "gpt-4o":          (2.50, 10.00),
    "gpt-4o-mini":     (0.15,  0.60),
    "gpt-3.5-turbo":   (0.50,  1.50),
}

def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp_price, out_price = MODEL_COSTS.get(model, (0, 0))
    return (prompt_tokens / 1_000_000 * inp_price) + (completion_tokens / 1_000_000 * out_price)


if run_btn and question and models_to_run:
    from rag.pipeline import run_rag_pipeline
    from agentic.agent import run_agentic_pipeline

    st.subheader("⏳ Running Models…")
    results: list[dict] = []

    for model_id, model_label in models_to_run:
        with st.status(f"Running **{model_label}**…", expanded=True) as s:
            t0 = time.time()
            try:
                if pipeline_type == "RAG":
                    res = run_rag_pipeline(question=question, n_results=5, model=model_id)
                    answer       = res.answer
                    confidence   = res.confidence
                    prompt_tok   = res.prompt_tokens
                    completion_tok = res.completion_tokens
                    retrieved    = res.retrieved_chunks
                    guardrail    = res.uncertainty_note or ""
                else:
                    res = run_agentic_pipeline(question=question, model=model_id)
                    answer       = res.answer
                    confidence   = res.confidence
                    prompt_tok   = res.prompt_tokens
                    completion_tok = res.completion_tokens
                    retrieved    = res.retrieved_chunks
                    guardrail    = res.uncertainty_note or ""

                elapsed  = time.time() - t0
                cost_usd = _estimate_cost(model_id, prompt_tok, completion_tok)

                # RAGAS-style faithfulness via LLM judge
                from evaluate.ragas_eval import _score_single
                record = {
                    "question": question,
                    "answer":   answer,
                    "contexts": [c.get("content", "") for c in retrieved],
                    "pipeline": pipeline_type.lower(),
                }
                scores = _score_single(record)

                results.append({
                    "model_id":         model_id,
                    "model_label":      model_label,
                    "answer":           answer,
                    "confidence":       confidence,
                    "guardrail":        guardrail,
                    "latency":          elapsed,
                    "prompt_tokens":    prompt_tok,
                    "completion_tokens": completion_tok,
                    "total_tokens":     prompt_tok + completion_tok,
                    "cost_usd":         cost_usd,
                    "faithfulness":     scores.get("faithfulness", 0.0),
                    "answer_relevancy": scores.get("answer_relevancy", 0.0),
                    "retrieved":        retrieved,
                })
                s.update(
                    label=f"✅ {model_label} — {elapsed:.1f}s, ${cost_usd:.5f}",
                    state="complete",
                )
            except Exception as exc:
                elapsed = time.time() - t0
                s.update(label=f"❌ {model_label} failed — {exc}", state="error")
                results.append({
                    "model_id": model_id, "model_label": model_label,
                    "answer": f"ERROR: {exc}", "confidence": 0.0, "guardrail": "",
                    "latency": elapsed, "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "cost_usd": 0.0,
                    "faithfulness": 0.0, "answer_relevancy": 0.0, "retrieved": [],
                })

    st.session_state["comparison_results"] = results
    st.session_state["comparison_question"] = question
    st.session_state["comparison_pipeline"] = pipeline_type
    st.rerun()

# ── Display results ────────────────────────────────────────────────────────────

results    = st.session_state.get("comparison_results", [])
c_question = st.session_state.get("comparison_question", "")
c_pipeline = st.session_state.get("comparison_pipeline", "RAG")

if results:
    st.subheader("📊 Results")
    st.markdown(
        f"**Question:** *{c_question}*  •  **Pipeline:** {c_pipeline}  "
        f"•  **Models tested:** {len(results)}"
    )

    # ── Summary table ──────────────────────────────────────────────────────────
    import pandas as pd

    df = pd.DataFrame([
        {
            "Model":           r["model_label"],
            "Faithfulness ↑":  f"{r['faithfulness']:.2f}",
            "Relevancy ↑":     f"{r['answer_relevancy']:.2f}",
            "Confidence":      r["confidence"],
            "Latency (s) ↓":   f"{r['latency']:.2f}",
            "Total Tokens":    r["total_tokens"],
            "Cost (USD) ↓":    f"${r['cost_usd']:.5f}",
        }
        for r in results
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown(
        """
> **Reading the table:**
> - **Faithfulness** (0–1): Does the answer stick to the retrieved context? 1.0 = no hallucinations.
> - **Relevancy** (0–1): Did the model actually answer the question? 1.0 = perfectly on-topic.
> - **↑** = higher is better.  **↓** = lower is better.
> - **Cost** is per-query in USD. Multiply by your daily query volume to see monthly spend.
    """
    )

    st.divider()

    # ── Answer comparison side-by-side ─────────────────────────────────────────
    st.subheader("📝 Answer Side-by-Side")
    st.markdown(
        "Read each answer and judge the quality yourself. Do you notice a difference? "
        "Often GPT-4o-mini produces a surprisingly similar answer at a fraction of the cost."
    )

    ans_cols = st.columns(len(results))
    for col, r in zip(ans_cols, results):
        with col:
            faith = r["faithfulness"]
            rel   = r["answer_relevancy"]
            color = "green" if faith >= 0.7 else ("orange" if faith >= 0.4 else "red")
            st.markdown(
                f"### {r['model_label']}\n"
                f"Faithfulness: :{color}[{faith:.2f}] · Relevancy: {rel:.2f}"
            )
            st.markdown(r["answer"])

            if r.get("guardrail"):
                st.caption(f"⚠️ Model note: {r['guardrail']}")

    st.divider()

    # ── Cost at scale calculator ───────────────────────────────────────────────
    st.subheader("💰 Cost at Scale Calculator")
    st.markdown(
        """
One query costs fractions of a cent. But AI products serve thousands of queries per day.
Use this calculator to see what your monthly bill would look like for each model.

> **Why this matters in an interview:** Being able to talk about cost at scale shows
> engineering maturity. "GPT-4o is smarter" is a junior answer.
> "GPT-4o-mini achieves 92% of the quality at 6% of the cost, saving $1,200/month at scale"
> is a senior answer.
        """
    )

    queries_per_day = st.slider(
        "Queries per day:",
        min_value=100,
        max_value=100_000,
        value=1_000,
        step=100,
        format="%d queries/day",
    )

    scale_data = []
    for r in results:
        tokens_per_query = r["total_tokens"] if r["total_tokens"] > 0 else 1000
        # Rough split: 80% prompt, 20% completion
        prompt_est = int(tokens_per_query * 0.80)
        compl_est  = int(tokens_per_query * 0.20)
        inp_price, out_price = MODEL_COSTS.get(r["model_id"], (0, 0))
        cost_per_query = (prompt_est / 1_000_000 * inp_price) + (compl_est / 1_000_000 * out_price)
        monthly_cost = cost_per_query * queries_per_day * 30

        scale_data.append({
            "Model":                r["model_label"],
            "Cost per Query (USD)": f"${cost_per_query:.5f}",
            "Monthly Cost":         f"${monthly_cost:,.2f}",
            "Annual Cost":          f"${monthly_cost * 12:,.2f}",
        })

    scale_df = pd.DataFrame(scale_data)
    st.dataframe(scale_df, use_container_width=True, hide_index=True)

    # Savings callout
    if len(results) >= 2:
        costs_amt = []
        for r in results:
            tokens_per_query = r["total_tokens"] if r["total_tokens"] > 0 else 1000
            prompt_est  = int(tokens_per_query * 0.80)
            compl_est   = int(tokens_per_query * 0.20)
            inp_p, out_p = MODEL_COSTS.get(r["model_id"], (0, 0))
            cpq = (prompt_est / 1_000_000 * inp_p) + (compl_est / 1_000_000 * out_p)
            costs_amt.append((r["model_label"], cpq * queries_per_day * 30))

        costs_amt.sort(key=lambda x: x[1], reverse=True)
        most_expensive = costs_amt[0]
        cheapest       = costs_amt[-1]
        savings        = most_expensive[1] - cheapest[1]
        if savings > 0:
            st.info(
                f"💡 Switching from **{most_expensive[0]}** to **{cheapest[0]}** would save "
                f"**${savings:,.2f}/month** (${savings*12:,.2f}/year) at {queries_per_day:,} queries/day."
            )

    st.divider()

    # ── Fine-tuning education section ──────────────────────────────────────────
    st.subheader("🎓 What Would Fine-Tuning Achieve?")

    tab_explain, tab_when, tab_data = st.tabs([
        "📖 Plain-English Explanation",
        "✅ When to Fine-Tune",
        "📊 At-Scale Projections",
    ])

    with tab_explain:
        st.markdown(
            """
### Fine-tuning in plain English

Imagine you hire a fresh university graduate who is brilliant at everything (**GPT-4o**).
After 6 months on the job, they've learned your company's specific terminology, your
coding standards, your preferred answer format.

**Fine-tuning** is that 6-month training period — but for an AI model, applied to a cheaper
model (like **GPT-4o-mini**).

**The process:**
1. Collect ~100-1,000 example question→ideal-answer pairs from YOUR data
2. Upload them to OpenAI's fine-tuning API (~$5–$50 one-time cost)
3. The API trains a *specialised version* of GPT-4o-mini on your examples
4. Your fine-tuned model learns your style, vocab, and domain

**The goal:** Get GPT-4o quality answers at GPT-4o-mini prices.

**The risk:** Your model only knows what you taught it. If a user asks something outside
your training data, it may perform worse than the base model on that edge case.
            """
        )

    with tab_when:
        st.markdown(
            """
### When does fine-tuning make sense?

**✅ Good candidates:**
- You have a **narrow, well-defined domain** (portfolio Q&A, legal clause detection, medical coding)
- You run **high volume** (10,000+ queries/month — otherwise savings don't offset training effort)
- You need a **specific format** (always return JSON, always use bullet points)
- You want **cost reduction** without sacrificing quality on known question types

**❌ Bad candidates:**
- Broad, open-ended assistant (GPT-4o's general reasoning is hard to replicate cheaply)
- You have **fewer than ~50 examples** (not enough signal)
- Questions change rapidly — your fine-tuned model becomes stale

**This portfolio's situation:**
The question set is narrow (portfolio Q&A) and high-quality examples are easy to generate.
Fine-tuning GPT-4o-mini on ~200 portfolio Q&A pairs would be a realistic next step,
potentially matching GPT-4o quality at 17× lower runtime cost.
            """
        )

    with tab_data:
        st.markdown(
            """
### Projected economics of fine-tuning this portfolio Q&A system
*(These are illustrative estimates based on OpenAI's published pricing as of 2025)*
            """
        )

        scale2 = st.slider(
            "Monthly query volume (for projection):",
            100, 50_000, 5_000, 100,
            key="ft_scale",
            format="%d queries/month",
        )

        avg_tokens = 1_000  # typical for portfolio Q&A

        # Base GPT-4o cost
        cost_4o_monthly   = (avg_tokens * 0.8 / 1e6 * 2.50 + avg_tokens * 0.2 / 1e6 * 10.0) * scale2
        # Base GPT-4o-mini cost
        cost_mini_monthly = (avg_tokens * 0.8 / 1e6 * 0.15 + avg_tokens * 0.2 / 1e6 * 0.60) * scale2
        # Fine-tuned mini: same runtime price as mini
        cost_ft_monthly   = cost_mini_monthly
        # One-time fine-tuning cost (rough: 200 examples × 1K tokens × $0.008/1K for training)
        onetimie_ft_cost  = 200 * 1000 / 1e6 * 8.00  # ≈ $1.60
        # Months to break even vs GPT-4o
        savings_per_month = cost_4o_monthly - cost_ft_monthly
        breakeven_months  = onetimie_ft_cost / savings_per_month if savings_per_month > 0 else 0

        proj_data = {
            "Approach":          ["GPT-4o (current best)", "GPT-4o-mini (current cheap)", "Fine-tuned GPT-4o-mini"],
            "Monthly Cost":      [f"${cost_4o_monthly:.2f}", f"${cost_mini_monthly:.2f}", f"${cost_ft_monthly:.2f}"],
            "One-Time Cost":     ["$0", "$0", f"${onetimie_ft_cost:.2f}"],
            "Expected Quality":  ["⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐ (estimated)"],
        }
        st.dataframe(pd.DataFrame(proj_data), use_container_width=True, hide_index=True)

        st.success(
            f"💡 At **{scale2:,} queries/month**, fine-tuning breaks even vs GPT-4o in "
            f"**{breakeven_months:.1f} months** — then saves **${savings_per_month:.2f}/month** indefinitely."
        )

        st.caption(
            "Quality estimate assumes fine-tuning on 200+ high-quality portfolio Q&A examples. "
            "Actual quality improvement depends on example diversity and curation."
        )

    st.divider()

    # ── Retrieved context diff ─────────────────────────────────────────────────
    with st.expander("📄 Retrieved Context Details — did all models see the same information?"):
        st.markdown(
            """
When using RAG, ALL models receive the same retrieved chunks as context.
This means the quality difference you see above is *purely* due to the model's
reasoning ability — not different information.

**Key insight:** If the faithfulness scores are both near 1.0, both models are staying
grounded in facts equally well. The relevancy difference tells you which model is
**better at synthesising** the provided information into a clear answer.
            """
        )
        for r in results:
            if r.get("retrieved"):
                st.markdown(f"**{r['model_label']}** — used {len(r['retrieved'])} retrieved chunks:")
                for j, chunk in enumerate(r["retrieved"][:3], 1):
                    repo   = chunk.get("repo_name", "unknown")
                    dist   = chunk.get("distance", "?")
                    preview = (chunk.get("content", "")[:200] + "…") if chunk.get("content") else "—"
                    st.caption(f"  Chunk {j}: [{repo}]  similarity={dist}  — {preview}")

elif not run_btn:
    st.info("Configure your comparison above and click **Run Comparison** to see results.")
