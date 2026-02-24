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
import random
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from shared.tracing import setup_tracing, record_trace, load_traces, clear_traces, trace_summary
TRACE_STATUS = setup_tracing()

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
from rag.pipeline import run_rag_pipeline, stream_rag_pipeline
from agentic.agent import run_agentic_pipeline, stream_agentic_pipeline
from shared.metrics import confidence_label, log_query, load_query_log, clear_query_log
from shared.guardrails import confidence_from_schema

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
            if st.button("Sign out", width='stretch'):
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
                    if st.button("Add", width='stretch') and new_email.strip():
                        updated = list(set(dynamic_emails + [new_email.strip().lower()]))
                        _save_dynamic_emails(updated)
                        st.success(f"Added {new_email.strip()}")
                        st.rerun()

                    st.caption("_Emails added here persist until the app restarts. For permanent access, add to Streamlit secrets._")

                    st.divider()
                    st.markdown("**Knowledge base**")
                    if st.button("🔄 Refresh (re-ingest GitHub)", width='stretch'):
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
                    if st.button("🗑️ Clear Log", width='stretch', key="clear_qlog"):
                        clear_query_log()
                        st.success("Log cleared")
                        st.rerun()
                    for _entry in _qlog[:5]:
                        st.json(_entry, expanded=False)

                    st.divider()
                    st.markdown("**🔍 Trace Log**")
                    st.caption(f"Status: {TRACE_STATUS}")
                    _tsummary = trace_summary()
                    _by_pipeline = _tsummary.get("by_pipeline", {})
                    if _by_pipeline:
                        for _pipe, _stats in _by_pipeline.items():
                            st.caption(
                                f"{_pipe}: {_stats['count']} runs · "
                                f"avg ${_stats['avg_cost']:.4f} · avg {_stats['avg_latency']:.1f}s"
                            )
                    elif _tsummary.get("count", 0) == 0:
                        st.caption("No traces recorded yet.")
                    if st.button("🗑️ Clear Traces", width='stretch', key="clear_traces"):
                        clear_traces()
                        st.success("Traces cleared")
                        st.rerun()
                    for _tr in load_traces(3):
                        st.json(_tr, expanded=False)

                st.divider()

    st.caption(f"Knowledge base: **{chunk_count:,} chunks**")
    st.header("Settings")
    n_results = st.slider("RAG: Chunks to retrieve", 1, 10, 5)
    max_iterations = st.slider("Agent: Max iterations", 1, 15, 8)

    pipeline_mode = st.selectbox("Mode", ["Compare Both", "A/B Test"])
    if pipeline_mode == "A/B Test":
        if "ab_group" not in st.session_state:
            st.session_state["ab_group"] = random.choice(["rag", "agentic"])
        st.caption(f"A/B group: **{st.session_state['ab_group']}** (fixed per session)")

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
            if st.button(q, key=q, width='stretch'):
                st.session_state["question"] = q

# Main input
question = st.text_input(
    "Ask a question about the GitHub portfolio:",
    value=st.session_state.get("question", ""),
    placeholder="e.g., What backend technologies has this developer used?",
)

if st.button("Compare", type="primary", width='stretch') and question:
    st.session_state.pop("compare_rag_error", None)
    st.session_state.pop("compare_agent_error", None)
    st.session_state["compare_question"] = question
    st.session_state["compare_pipeline_mode"] = pipeline_mode

    ab_group = st.session_state.get("ab_group", None) if pipeline_mode == "A/B Test" else None
    run_rag   = (pipeline_mode == "Compare Both") or (ab_group == "rag")
    run_agent = (pipeline_mode == "Compare Both") or (ab_group == "agentic")

    col_rag, col_agent = st.columns(2)

    # ── Streaming RAG ─────────────────────────────────────────────────────────
    if run_rag:
        with col_rag:
            st.subheader("🗂️ Classic RAG")
            st.caption(
                "**How RAG works:** The question is converted to a vector, used to find the "
                "most similar document chunks in ChromaDB, then passed to the LLM in a single call. "
                "Tokens stream below as GPT generates them — no waiting for the full answer."
            )
            _rag_status = st.empty()
            _rag_chunks_info = st.empty()
            st.markdown("**Answer** *(streaming)*")
            _rag_answer_box = st.empty()
            _rag_metrics_box = st.empty()
            _rag_note_box = st.empty()

            _rag_result = None
            _rag_live_answer = ""
            try:
                for _ev in stream_rag_pipeline(question, n_results=n_results):
                    if _ev["type"] == "status":
                        _rag_status.info(_ev["text"])
                    elif _ev["type"] == "chunks":
                        _ch = _ev["chunks"]
                        _rag_chunks_info.caption(
                            f"📄 Retrieved {len(_ch)} chunks · "
                            f"repos: {', '.join(set(c['metadata'].get('repo_name','?') for c in _ch))}"
                        )
                    elif _ev["type"] == "token":
                        _rag_live_answer += _ev["text"]
                        _rag_answer_box.markdown(_rag_live_answer + "▌")
                    elif _ev["type"] == "result":
                        _rag_result = _ev["result"]
                        _rag_answer_box.markdown(_rag_result.answer)
                        _rag_status.empty()

                if _rag_result:
                    _emoji, _label = confidence_label(_rag_result.confidence)
                    _rag_metrics_box.markdown(
                        f"✅ Done in **{_rag_result.latency_seconds}s** · "
                        f"{_rag_result.total_tokens} tokens · "
                        f"${_rag_result.cost_usd:.4f} · "
                        f"Confidence: {_emoji} {_label} ({_rag_result.confidence:.2f})"
                    )
                    if _rag_result.uncertainty_note:
                        _rag_note_box.warning(f"⚠️ {_rag_result.uncertainty_note}")
                    st.session_state["compare_rag"] = _rag_result
            except Exception as _e:
                import traceback
                _rag_status.error(f"RAG error: {_e}")
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc())
                st.session_state["compare_rag"] = None
                st.session_state["compare_rag_error"] = str(_e)
    else:
        st.session_state.pop("compare_rag", None)

    # ── Streaming Agentic ─────────────────────────────────────────────────────
    if run_agent:
        with col_agent:
            st.subheader("🤖 Agentic Retrieval")
            st.caption(
                "**How the agent works:** Before touching any data, a planner generates a "
                "step-by-step retrieval strategy. The executor then calls tools one at a time, "
                "reading intermediate results to decide what to look up next — like a detective "
                "following clues. Each step appears below as it happens."
            )
            _ag_status = st.empty()
            _plan_box  = st.empty()
            _steps_container = st.container()
            st.markdown("**Answer**")
            _ag_answer_box = st.empty()
            _ag_metrics_box = st.empty()

            _ag_result = None
            _step_html_lines: list[str] = []
            try:
                for _ev in stream_agentic_pipeline(question, max_iterations=max_iterations):
                    if _ev["type"] == "status":
                        _ag_status.info(_ev["text"])
                    elif _ev["type"] == "plan":
                        _plan_box.info(f"📋 **Retrieval Plan**\n\n{_ev['content']}")
                    elif _ev["type"] == "tool_call":
                        _step_html_lines.append(
                            f"🔧 **Tool call:** `{_ev['tool']}` with args `{_ev['args']}`"
                        )
                        with _steps_container:
                            st.markdown("\n\n".join(_step_html_lines[-6:]))
                    elif _ev["type"] == "observation":
                        preview = _ev["content"][:150]
                        _step_html_lines.append(
                            f"📊 **Result from `{_ev['tool']}`:** {preview}…"
                        )
                        with _steps_container:
                            st.markdown("\n\n".join(_step_html_lines[-6:]))
                    elif _ev["type"] == "answer":
                        _ag_answer_box.markdown(_ev["content"])
                    elif _ev["type"] == "result":
                        _ag_result = _ev["result"]
                        _ag_status.empty()

                if _ag_result:
                    _ag_metrics_box.markdown(
                        f"✅ Done in **{_ag_result.latency_seconds}s** · "
                        f"{_ag_result.llm_calls} LLM calls · "
                        f"{len(_ag_result.tool_calls)} tool calls · "
                        f"${_ag_result.cost_usd:.4f}"
                    )
                    st.session_state["compare_agent"] = _ag_result
            except Exception as _e:
                import traceback
                _ag_status.error(f"Agent error: {_e}")
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc())
                st.session_state["compare_agent"] = None
                st.session_state["compare_agent_error"] = str(_e)
    else:
        st.session_state.pop("compare_agent", None)

    # ── Audit log + tracing (after both complete) ─────────────────────────────
    _r = st.session_state.get("compare_rag")
    _a = st.session_state.get("compare_agent")
    if _r:
        log_query({
            "question": question, "pipeline": "rag", "ab_group": ab_group,
            "rag_answer": _r.answer[:200], "rag_tokens": _r.total_tokens,
            "rag_cost_usd": round(_r.cost_usd, 6), "rag_confidence": round(_r.confidence, 3),
        })
        record_trace(
            pipeline="rag", question=question, answer=_r.answer,
            prompt_tokens=_r.prompt_tokens, completion_tokens=_r.completion_tokens,
            cost_usd=_r.cost_usd, latency_seconds=_r.latency_seconds,
            tool_calls=[], confidence=_r.confidence, ab_group=ab_group,
        )
    if _a:
        log_query({
            "question": question, "pipeline": "agentic", "ab_group": ab_group,
            "agent_tokens": _a.total_tokens, "agent_cost_usd": round(_a.cost_usd, 6),
            "agent_llm_calls": _a.llm_calls,
        })
        _ag_conf = confidence_from_schema(_a.guardrails.confidence) if _a.guardrails else 0.5
        record_trace(
            pipeline="agentic", question=question, answer=_a.answer,
            prompt_tokens=_a.prompt_tokens, completion_tokens=_a.completion_tokens,
            cost_usd=_a.cost_usd, latency_seconds=_a.latency_seconds,
            tool_calls=_a.tool_calls, confidence=_ag_conf, ab_group=ab_group,
        )

    # Rerun to show the clean post-stream static render
    st.rerun()

# ── Results (rendered from session_state — survives reruns after streaming) ───
if "compare_question" in st.session_state:
    rag_result   = st.session_state.get("compare_rag")
    agent_result = st.session_state.get("compare_agent")
    rag_err      = st.session_state.get("compare_rag_error")
    agent_err    = st.session_state.get("compare_agent_error")
    saved_q      = st.session_state["compare_question"]

    st.markdown(f"### Results for: *{saved_q}*")
    st.caption(
        "Results below are saved from the last run. "
        "Expand each section to explore the pipeline's reasoning, retrieved sources, and metrics. "
        "Use the **📊 Eval Dashboard** page (left sidebar) to run RAGAS quality scoring on many questions at once."
    )

    col_rag, col_agent = st.columns(2)

    # --- RAG Column ---
    with col_rag:
        st.subheader("🗂️ Classic RAG")
        st.markdown(
            """
            **What you're seeing:** The pipeline retrieved the most similar text chunks from the
            knowledge base using vector search, then sent them all at once to GPT as context.
            One retrieval pass → one LLM call → one answer. Fast and cheap, but limited to what
            it happened to retrieve on the first try.
            """
        )
        if rag_err:
            st.error(f"RAG Error: {rag_err}")
        elif rag_result:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("LLM Calls", rag_result.llm_calls,
                      help="RAG always makes exactly 1 LLM call — retrieve then generate")
            m2.metric("Tokens", rag_result.total_tokens,
                      help="Total tokens consumed (input context + output answer)")
            m3.metric("Chunks", len(rag_result.retrieved_chunks),
                      help="Number of document segments retrieved from the vector database")
            m4.metric("Cost", f"${rag_result.cost_usd:.4f}",
                      help="OpenAI API cost for this single query")
            _emoji, _label = confidence_label(rag_result.confidence)
            st.caption(
                f"Retrieval confidence: {_emoji} **{_label}** ({rag_result.confidence:.2f})  \n"
                "_Confidence = 1 − average cosine distance of retrieved chunks. "
                "Low confidence means the knowledge base didn't have a strong match._"
            )
            if getattr(rag_result, "uncertainty_note", None):
                st.warning(f"⚠️ {rag_result.uncertainty_note}")
            st.markdown("**Answer:**")
            st.markdown(rag_result.answer)

            with st.expander("🔍 Retrieval Trace — what happened step-by-step"):
                st.caption(
                    "Every RAG pipeline has exactly these steps: **Retrieve → Build Context → Generate**. "
                    "The trace shows you what was searched for, what was found, and how the answer was produced."
                )
                for step in rag_result.steps:
                    st.text(step)

            with st.expander("📄 Retrieved Chunks — the raw source material"):
                st.caption(
                    "These are the exact text segments the LLM saw as context. "
                    "The distance score shows how similar each chunk is to your question "
                    "(0 = identical, 1 = unrelated). Lower distance = more relevant."
                )
                for chunk in rag_result.retrieved_chunks:
                    repo  = chunk["metadata"].get("repo_name", "unknown")
                    fpath = chunk["metadata"].get("file_path", "")
                    dist  = round(chunk["distance"], 3)
                    lbl   = f"{repo}/{fpath}" if fpath else repo
                    st.markdown(f"**{lbl}** · distance: `{dist}`")
                    st.code(chunk["text"][:300], language="markdown")
                    st.divider()

    # --- Agentic Column ---
    with col_agent:
        st.subheader("🤖 Agentic Retrieval")
        st.markdown(
            """
            **What you're seeing:** Instead of one fixed retrieval pass, the agent first
            **plans** what to look up, then **executes** tool calls one at a time — using each
            result to decide what to search for next. Like a researcher who follows up on leads
            rather than only reading the first page of search results.
            """
        )
        if agent_err:
            st.error(f"Agent Error: {agent_err}")
        elif agent_result:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("LLM Calls", agent_result.llm_calls,
                      help="Each planning/execution/synthesis step is a separate LLM call")
            m2.metric("Tool Calls", len(agent_result.tool_calls),
                      help="Number of times the agent called a retrieval tool (search, fetch, list, etc.)")
            m3.metric("Latency", f"{agent_result.latency_seconds}s",
                      help="Total wall-clock time including all LLM + tool calls")
            m4.metric("Cost", f"${agent_result.cost_usd:.4f}",
                      help="Total OpenAI spend — higher than RAG due to multiple LLM calls")

            if getattr(agent_result, "plan", None):
                with st.expander("📋 Agent Plan — strategy before any tools were called", expanded=True):
                    st.caption(
                        "The **Planner node** runs first on a cheap model (gpt-4o-mini) and generates "
                        "this numbered retrieval plan *before* any tool is called. This makes the agent's "
                        "strategy transparent and prevents aimless tool-calling."
                    )
                    st.markdown(agent_result.plan)

            if agent_result.guardrails and agent_result.guardrails.uncertainty_note:
                st.warning(f"⚠️ {agent_result.guardrails.uncertainty_note}")

            st.markdown("**Answer:**")
            st.markdown(agent_result.answer)

            with st.expander("🧠 Agent Reasoning Trace — thought process step-by-step"):
                st.caption(
                    "Each step in the trace is a node in the LangGraph execution graph: "
                    "PLAN → EXECUTE → TOOLS → repeat → SYNTHESIZE. "
                    "THOUGHT = the agent deciding what to do. ACTION = calling a tool. "
                    "OBSERVATION = reading the tool's result."
                )
                for step in agent_result.steps:
                    if step.startswith("THOUGHT"):
                        st.info(step)
                    elif step.startswith("ACTION"):
                        st.warning(step)
                    elif step.startswith("OBSERVATION"):
                        st.success(step)
                    else:
                        st.text(step)

            with st.expander("🔧 Tool Call Details — what the agent retrieved"):
                st.caption(
                    "Each tool call shows the exact data the agent received. "
                    "This is the 'context' the agent built up iteratively — "
                    "compared to RAG's single-shot retrieval."
                )
                for i, tc in enumerate(agent_result.tool_calls, 1):
                    st.markdown(f"**Step {i}: `{tc['tool']}`**")
                    st.code(tc["output_preview"], language="text")
                    st.divider()

# Footer
st.divider()
st.caption(
    "This app compares two retrieval strategies on the same ChromaDB knowledge base. "
    "RAG uses Pydantic-validated structured output with confidence gating. The agentic "
    "pipeline uses a LangGraph PLANNER→EXECUTOR→SYNTHESIZER graph with explicit plan generation. "
    "Navigate to **📊 Eval Dashboard** to run RAGAS quality scoring, "
    "**🧪 Model Comparison** to see fine-tuning trade-offs, "
    "or **🕸️ Knowledge Graph** to explore the tech relationship graph."
)
