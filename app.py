"""
Streamlit app: RAG vs Agentic Retrieval — Side-by-Side Comparison.

Run with:
    streamlit run app.py

Auth:
    - Enabled when REQUIRE_AUTH=true (used on Streamlit Community Cloud)
    - Skipped locally for fast iteration
    - Allowlist stored in st.secrets["allowed_emails"]

Auto-ingest:
    - On first run / cold start, if ChromaDB is empty it re-ingests from GitHub
    - Subsequent runs use the cached DB
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG vs Agentic Retrieval", layout="wide")

# ── Authentication ────────────────────────────────────────────────────────────
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"

if REQUIRE_AUTH:
    # Guard: st.experimental_user requires [auth] to be configured in secrets.
    # If not yet set up, show a clear message instead of crashing.
    try:
        is_logged_in = st.user.is_logged_in
    except AttributeError:
        st.error("**Streamlit version too old.** `st.login()` requires Streamlit ≥ 1.41.")
        st.stop()

    if not is_logged_in:
        st.title("RAG vs Agentic Retrieval")
        st.markdown(
            "This demo compares **classic RAG** vs **agentic retrieval** on a live GitHub "
            "portfolio knowledge base. Sign in with Google to access it."
        )
        st.login("google")
        st.stop()

    # Allowlist check — add emails to secrets under allowed_emails
    allowed = st.secrets.get("allowed_emails", [])
    user_email = st.user.email or ""
    if allowed and user_email not in allowed:
        st.error(
            f"Access denied: **{user_email}** is not on the allowlist. "
            "Contact the site owner to request access."
        )
        if st.button("Sign out"):
            st.logout()
        st.stop()

# ── Knowledge base auto-ingest ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def ensure_knowledge_base() -> int:
    """
    Check ChromaDB on startup. If empty (cold start / first deploy),
    run the full ingestion pipeline and return the chunk count.
    """
    from shared.vector_store import get_or_create_collection
    col = get_or_create_collection()
    if col.count() > 0:
        return col.count()

    # Cold start — ingest from GitHub
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data.fetch_readmes import main as ingest
    ingest()
    return get_or_create_collection().count()

with st.spinner("🔍 Initialising knowledge base…"):
    chunk_count = ensure_knowledge_base()

# ── Deferred imports (after env loaded) ──────────────────────────────────────
from rag.pipeline import run_rag_pipeline
from agentic.agent import run_agentic_pipeline

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("RAG vs Agentic Retrieval")
st.caption("Compare classic RAG with agentic retrieval on the same knowledge base")

# Sidebar
with st.sidebar:
    # Auth info
    if REQUIRE_AUTH and st.user.is_logged_in:
        st.caption(f"👤 {st.user.email}")
        if st.button("Sign out", use_container_width=True):
            st.logout()
        st.divider()

    st.caption(f"Knowledge base: **{chunk_count:,} chunks**")
    st.header("Settings")
    n_results = st.slider("RAG: Chunks to retrieve", 1, 10, 5)
    max_iterations = st.slider("Agent: Max iterations", 1, 15, 8)

    st.divider()
    st.header("Sample Questions")

    sample_questions = {
        "Simple": [
            "What is the portfolio-site project about?",
            "What language is the pokemon-api written in?",
        ],
        "Multi-hop": [
            "Which projects use Angular, and how do they differ?",
            "Compare the authentication approaches in MEANAuthApp vs MEANAuthAppAngular.",
        ],
        "Ambiguous": [
            "What should I look at to see this developer's best work?",
            "Is this developer experienced enough for a full-stack role?",
        ],
    }

    for tier, questions in sample_questions.items():
        st.subheader(tier)
        for q in questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state["question"] = q

# Main input
question = st.text_input(
    "Ask a question about the GitHub portfolio:",
    value=st.session_state.get("question", ""),
    placeholder="e.g., What backend technologies has this developer used?",
)

if st.button("Compare", type="primary", use_container_width=True) and question:
    col_rag, col_agent = st.columns(2)

    # --- RAG Column ---
    with col_rag:
        st.subheader("Classic RAG")
        with st.spinner("Retrieving & generating..."):
            try:
                rag_result = run_rag_pipeline(question, n_results=n_results)

                st.success(f"Done in {rag_result.latency_seconds}s")

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("LLM Calls", rag_result.llm_calls)
                m2.metric("Tokens", rag_result.total_tokens)
                m3.metric("Chunks", len(rag_result.retrieved_chunks))

                # Answer
                st.markdown("**Answer:**")
                st.markdown(rag_result.answer)

                # Trace
                with st.expander("Retrieval Trace"):
                    for step in rag_result.steps:
                        st.text(step)

                # Retrieved chunks
                with st.expander("Retrieved Chunks"):
                    for chunk in rag_result.retrieved_chunks:
                        repo = chunk["metadata"].get("repo_name", "unknown")
                        fpath = chunk["metadata"].get("file_path", "")
                        dist = round(chunk["distance"], 3)
                        label = f"{repo}/{fpath}" if fpath else repo
                        st.markdown(f"**{label}** (distance: {dist})")
                        st.code(chunk["text"][:300], language="markdown")
                        st.divider()

            except Exception as e:
                st.error(f"RAG Error: {e}")

    # --- Agentic Column ---
    with col_agent:
        st.subheader("Agentic Retrieval")
        with st.spinner("Agent is thinking..."):
            try:
                agent_result = run_agentic_pipeline(
                    question,
                    max_iterations=max_iterations,
                    verbose=False,
                )

                st.success(f"Done in {agent_result.latency_seconds}s")

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("LLM Calls", agent_result.llm_calls)
                m2.metric("Tool Calls", len(agent_result.tool_calls))
                m3.metric("Latency", f"{agent_result.latency_seconds}s")

                # Answer
                st.markdown("**Answer:**")
                st.markdown(agent_result.answer)

                # Reasoning trace
                with st.expander("Agent Reasoning Trace"):
                    for step in agent_result.steps:
                        if step.startswith("THOUGHT"):
                            st.info(step)
                        elif step.startswith("ACTION"):
                            st.warning(step)
                        elif step.startswith("OBSERVATION"):
                            st.success(step)
                        else:
                            st.text(step)

                # Tool calls detail
                with st.expander("Tool Call Details"):
                    for tc in agent_result.tool_calls:
                        st.markdown(f"**{tc['tool']}**(`{tc['input']}`)")
                        st.code(tc["output_preview"], language="text")
                        st.divider()

            except Exception as e:
                st.error(f"Agent Error: {e}")

# Footer
st.divider()
st.caption(
    "This app compares two retrieval strategies on the same ChromaDB knowledge base. "
    "RAG performs a single retrieve-then-generate pass, while the agentic approach "
    "uses a ReAct agent that decides when and what to retrieve iteratively."
)
