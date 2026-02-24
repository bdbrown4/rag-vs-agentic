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

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Dynamic allowlist helpers ─────────────────────────────────────────────────
_ALLOWED_EMAILS_PATH = Path(__file__).parent / "data" / "allowed_emails.json"


def _load_dynamic_emails() -> list[str]:
    try:
        if _ALLOWED_EMAILS_PATH.exists():
            return json.loads(_ALLOWED_EMAILS_PATH.read_text())
    except Exception:
        pass
    return []


def _save_dynamic_emails(emails: list[str]) -> None:
    _ALLOWED_EMAILS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ALLOWED_EMAILS_PATH.write_text(json.dumps(sorted(set(emails))))


def _get_all_allowed_emails() -> list[str]:
    static = list(st.secrets.get("allowed_emails", []))
    dynamic = _load_dynamic_emails()
    return list(set(static + dynamic))


def _is_admin(email: str) -> bool:
    admin = st.secrets.get("admin_email", "")
    if not admin:
        allowed = list(st.secrets.get("allowed_emails", []))
        admin = allowed[0] if allowed else ""
    return bool(email and email == admin)

st.set_page_config(page_title="RAG vs Agentic Retrieval", layout="wide")

# ── Authentication ────────────────────────────────────────────────────────────
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"

if REQUIRE_AUTH:
    # Guard: st.experimental_user requires [auth] to be configured in secrets.
    # If not yet set up, show a clear message instead of crashing.
    # Support both st.user (≥1.41) and st.experimental_user (older)
    _user = getattr(st, "user", None) or getattr(st, "experimental_user", None)
    if _user is None or not hasattr(_user, "is_logged_in"):
        st.error("This Streamlit version does not support authentication. Contact the site owner.")
        st.stop()

    if not _user.is_logged_in:
        st.title("RAG vs Agentic Retrieval")
        st.markdown(
            "This demo compares **classic RAG** vs **agentic retrieval** on a live GitHub "
            "portfolio knowledge base. Sign in with Google to access it."
        )
        auth_configured = "auth" in st.secrets
        if not auth_configured:
            st.error(
                "Google authentication is not configured for this deployment. "
                "Add an `[auth]` section with your Google OAuth credentials to the app secrets."
            )
            st.stop()
        st.login("google")
        st.stop()

    # Allowlist check — merges secrets list + dynamically added emails
    allowed = _get_all_allowed_emails()
    user_email = _user.email or ""
    if allowed and user_email not in allowed:
        st.error(
            f"Access denied: **{user_email}** is not on the allowlist. "
            "Contact the site owner to request access."
        )
        if st.button("Sign out"):
            st.logout()
        st.stop()

# ── Knowledge base auto-ingest ────────────────────────────────────────────────
if "kb_chunk_count" not in st.session_state:
    from shared.vector_store import get_or_create_collection
    _col = get_or_create_collection()
    if _col.count() > 0:
        # Warm — DB already populated
        st.session_state.kb_chunk_count = _col.count()
    else:
        # Cold start — ingest live with visible progress
        with st.status("🔍 Initializing knowledge base…", expanded=True) as _status:
            from data.fetch_readmes import main as _ingest
            _ingest(log_fn=st.write)
            st.session_state.kb_chunk_count = get_or_create_collection().count()
            _status.update(
                label=f"✅ Knowledge base ready — {st.session_state.kb_chunk_count:,} chunks",
                state="complete",
                expanded=False,
            )

chunk_count: int = st.session_state.kb_chunk_count

# ── Deferred imports (after env loaded) ──────────────────────────────────────
from rag.pipeline import run_rag_pipeline
from agentic.agent import run_agentic_pipeline
from shared.metrics import confidence_label, log_query, load_query_log, clear_query_log

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("RAG vs Agentic Retrieval")
st.caption("Compare classic RAG with agentic retrieval on the same knowledge base")

# Sidebar
with st.sidebar:
    # Auth info
    if REQUIRE_AUTH:
        _u = getattr(st, "user", None) or getattr(st, "experimental_user", None)
        if _u and getattr(_u, "is_logged_in", False):
            st.caption(f"👤 {_u.email}")
            if st.button("Sign out", use_container_width=True):
                st.logout()
            st.divider()

            # ── Admin panel ────────────────────────────────────────────
            if _is_admin(_u.email):
                with st.expander("⚙️ Admin", expanded=False):
                    st.markdown("**Allowed emails**")
                    dynamic_emails = _load_dynamic_emails()
                    static_emails = list(st.secrets.get("allowed_emails", []))

                    # Show current list
                    all_emails = sorted(set(static_emails + dynamic_emails))
                    for em in all_emails:
                        col_em, col_rm = st.columns([4, 1])
                        tag = " _(secrets)_" if em in static_emails else ""
                        col_em.markdown(f"`{em}`{tag}")
                        if em not in static_emails:
                            if col_rm.button("✕", key=f"rm_{em}"):
                                dynamic_emails = [e for e in dynamic_emails if e != em]
                                _save_dynamic_emails(dynamic_emails)
                                st.rerun()

                    # Add new email
                    new_email = st.text_input("Add email", placeholder="user@example.com", key="new_email_input", label_visibility="collapsed")
                    if st.button("Add", use_container_width=True) and new_email.strip():
                        updated = list(set(dynamic_emails + [new_email.strip().lower()]))
                        _save_dynamic_emails(updated)
                        st.success(f"Added {new_email.strip()}")
                        st.rerun()

                    st.caption("_Emails added here persist until the app restarts. For permanent access, add to Streamlit secrets._")

                    st.divider()
                    st.markdown("**Knowledge base**")
                    if st.button("🔄 Refresh (re-ingest GitHub)", use_container_width=True):
                        from shared.vector_store import get_client, COLLECTION_NAME
                        try:
                            get_client().delete_collection(COLLECTION_NAME)
                        except Exception:
                            pass
                        del st.session_state["kb_chunk_count"]
                        st.rerun()

                    st.divider()
                    st.markdown("**📋 Query Log**")
                    _qlog = load_query_log(20)
                    st.caption(f"{len(_qlog)} recent queries logged")
                    if st.button("🗑️ Clear Log", use_container_width=True, key="clear_qlog"):
                        clear_query_log()
                        st.success("Log cleared")
                        st.rerun()
                    for _entry in _qlog[:5]:
                        st.json(_entry, expanded=False)

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
    _rag_result = None
    _agent_result = None
    col_rag, col_agent = st.columns(2)

    # --- RAG Column ---
    with col_rag:
        st.subheader("Classic RAG")
        with st.spinner("Retrieving & generating..."):
            try:
                rag_result = run_rag_pipeline(question, n_results=n_results)
                _rag_result = rag_result

                st.success(f"Done in {rag_result.latency_seconds}s")

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("LLM Calls", rag_result.llm_calls)
                m2.metric("Tokens", rag_result.total_tokens)
                m3.metric("Chunks", len(rag_result.retrieved_chunks))
                m4.metric("Cost", f"${rag_result.cost_usd:.4f}")
                _emoji, _label = confidence_label(rag_result.confidence)
                st.caption(f"Confidence: {_emoji} {_label} ({rag_result.confidence:.2f})")

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
                _agent_result = agent_result

                st.success(f"Done in {agent_result.latency_seconds}s")

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("LLM Calls", agent_result.llm_calls)
                m2.metric("Tool Calls", len(agent_result.tool_calls))
                m3.metric("Latency", f"{agent_result.latency_seconds}s")
                m4.metric("Cost", f"${agent_result.cost_usd:.4f}")

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

    # ── Audit log ──────────────────────────────────────────────────────────
    if _rag_result and _agent_result:
        log_query({
            "question": question,
            "rag_answer": _rag_result.answer[:200],
            "rag_tokens": _rag_result.total_tokens,
            "rag_cost_usd": round(_rag_result.cost_usd, 6),
            "rag_confidence": round(_rag_result.confidence, 3),
            "agent_tokens": _agent_result.total_tokens,
            "agent_cost_usd": round(_agent_result.cost_usd, 6),
            "agent_llm_calls": _agent_result.llm_calls,
        })

# Footer
st.divider()
st.caption(
    "This app compares two retrieval strategies on the same ChromaDB knowledge base. "
    "RAG performs a single retrieve-then-generate pass, while the agentic approach "
    "uses a ReAct agent that decides when and what to retrieve iteratively."
)
