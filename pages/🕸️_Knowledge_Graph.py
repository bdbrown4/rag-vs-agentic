"""
Knowledge Graph Explorer
========================
A knowledge graph is a fundamentally different way to store and query information
compared to the vector DB used by RAG and Agentic pipelines.

Vector DB  → "find text similar to my question"
Knowledge Graph → "traverse structured relationships: REPO --uses--> TECHNOLOGY"

This page lets you build that graph once (calls GPT-4o-mini per repo) and then run
instant, structured queries against it — no LLM needed at query time.
"""

import streamlit as st

st.set_page_config(
    page_title="Knowledge Graph — RAG vs Agentic",
    page_icon="🕸️",
    layout="wide",
)

from shared.knowledge_graph import (
    load_graph,
    build_graph_from_kb,
    repos_using,
    technologies_in,
    technology_frequency,
    repo_similarity,
    category_groups,
    graph_to_edge_dataframe,
    technology_heatmap_data,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🕸️ Knowledge Graph")
    st.caption("Explore portfolio structure as a graph of repos and technologies")

    st.markdown("---")
    st.markdown("### About this tool")
    st.markdown(
        """
A **knowledge graph** stores *relationships* between things, not just text.

Here the relationships are:

```
Portfolio
  └─ has ─► Repository
              └─ uses ─► Technology
```

This lets you ask **structural questions**:
- *What repos use TypeScript?*
- *Which repos are most similar?*
- *What is the most-used technology?*

These queries run **instantly** because the structure is pre-computed — no LLM
call needed at query time. The up-front cost is one GPT call per repo to extract
its tech stack from the README.
        """
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🕸️ Knowledge Graph Explorer")
st.markdown(
    """
**What you're looking at:** A structured map of every repo in this portfolio, linked to the
technologies it uses.  This is built *on top of* the same ChromaDB knowledge base that powers
the RAG and Agentic pipelines — but rather than searching for similar text, we extract
*structured facts* (tech stack, category) and store them as a graph.

> **Why does this matter to a non-technical interviewer?**
> Imagine a library's card catalogue versus a map of which books reference each other.
> Vector search is the catalogue (find books *about* React). A knowledge graph is the map
> (show me every project *built with* React, and what they have in common).
"""
)

st.divider()

# ── Build / Load graph ────────────────────────────────────────────────────────

graph = load_graph()
has_graph = bool(graph.get("repos"))

col_build, col_status = st.columns([1, 2])

with col_build:
    rebuild_label = "🔄 Rebuild Graph" if has_graph else "🔨 Build Knowledge Graph"
    do_build = st.button(rebuild_label, type="primary", use_container_width=True)
    st.caption(
        "Queries ChromaDB for each repo's README, then calls GPT-4o-mini (~$0.01 total) "
        "to extract the tech stack. Run once; results are cached to disk."
    )

with col_status:
    if has_graph:
        n_repos  = len(graph["repos"])
        n_techs  = len(graph["technologies"])
        st.success(
            f"✅ Graph loaded — **{n_repos} repos**, **{n_techs} unique technologies**. "
            "Rebuild any time after ingesting new repos."
        )
    else:
        st.info("⬅️ The graph hasn't been built yet. Click the button to generate it.")

if do_build:
    log_lines: list[str] = []
    log_box = st.empty()

    def _log(msg: str):
        log_lines.append(msg)
        log_box.code("\n".join(log_lines[-30:]))   # show last 30 lines

    with st.spinner("Building knowledge graph…"):
        graph = build_graph_from_kb(log_fn=_log)

    st.success(
        f"✅ Done! Graph has **{len(graph['repos'])} repos** and "
        f"**{len(graph['technologies'])} technologies**."
    )
    has_graph = True
    st.rerun()

if not has_graph:
    st.stop()

# ── Overview stats ────────────────────────────────────────────────────────────

st.subheader("📊 Portfolio at a Glance")
st.markdown(
    "These numbers come from the graph structure — computed in microseconds with no LLM call."
)

n_repos = len(graph["repos"])
n_techs = len(graph["technologies"])
top_techs = technology_frequency(graph)[:3]
cats = category_groups(graph)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Repositories", n_repos, help="Number of GitHub repos in the knowledge base")
m2.metric("Unique Technologies", n_techs, help="Distinct languages, frameworks, and tools across all repos")
m3.metric(
    "Top Technology",
    top_techs[0][0] if top_techs else "—",
    help=f"Used in {top_techs[0][1]} repos" if top_techs else "",
)
m4.metric("Project Categories", len(cats), help="frontend · backend · fullstack · mobile · tool · data · other")

st.divider()

# ── Heatmap ───────────────────────────────────────────────────────────────────

st.subheader("🔥 Technology Usage Heatmap")
st.markdown(
    """
Each cell shows whether a repo (row) uses a technology (column).
**Dark = yes, light = no.**
This instantly reveals patterns: which repos share a tech stack, which technologies are universal,
and which are niche.

> A recruiter can glance at this and understand the breadth of the portfolio in 5 seconds —
> something that would take hours to extract by reading each README manually.
"""
)

try:
    import altair as alt
    import pandas as pd

    df_heat = technology_heatmap_data(graph)

    if not df_heat.empty:
        # Colour by category
        cat_order = sorted(df_heat["category"].unique())
        # Build heatmap
        heatmap = (
            alt.Chart(df_heat)
            .mark_rect(stroke="white", strokeWidth=0.5)
            .encode(
                x=alt.X("technology:N", title="Technology", sort="-y"),
                y=alt.Y("repo:N", title="Repository"),
                color=alt.Color(
                    "uses:Q",
                    scale=alt.Scale(range=["#f0f0f0", "#2563eb"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("repo:N", title="Repo"),
                    alt.Tooltip("technology:N", title="Technology"),
                    alt.Tooltip("uses:O", title="Uses?"),
                    alt.Tooltip("category:N", title="Category"),
                ],
            )
            .properties(height=max(300, n_repos * 28))
        )
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("No data to display in heatmap.")

except ImportError:
    st.warning("Altair not installed — install with `pip install altair` to see the heatmap.")

st.divider()

# ── Technology frequency bar chart ────────────────────────────────────────────

st.subheader("📈 Technology Frequency")
st.markdown(
    "How many repos use each technology? This is a simple COUNT query on the graph — no text search needed."
)

try:
    freq_data = technology_frequency(graph)
    freq_df = pd.DataFrame(freq_data[:20], columns=["Technology", "Repos"])

    bar = (
        alt.Chart(freq_df)
        .mark_bar(color="#2563eb")
        .encode(
            x=alt.X("Repos:Q", title="Number of Repos"),
            y=alt.Y("Technology:N", sort="-x", title=""),
            tooltip=["Technology", "Repos"],
        )
        .properties(height=420)
    )
    st.altair_chart(bar, use_container_width=True)

except Exception:
    # Fallback table if altair fails
    import pandas as pd
    freq_data = technology_frequency(graph)
    st.dataframe(
        pd.DataFrame(freq_data[:20], columns=["Technology", "Repos"]),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ── Interactive filter ─────────────────────────────────────────────────────────

st.subheader("🔍 Filter Repos by Technology")
st.markdown(
    """
Select a technology to see every repo that uses it — and for each matching repo,
see its full stack.

> This is a **graph traversal** query. In code it's just:
> `graph["technologies"]["React"]["repos"]`
> — a single dictionary lookup, instant, no AI involved.
"""
)

all_techs_sorted = [t for t, _ in technology_frequency(graph)]
selected_tech = st.selectbox("Show repos that use:", all_techs_sorted, index=0)

if selected_tech:
    matching_repos = repos_using(graph, selected_tech)
    if matching_repos:
        st.success(f"**{len(matching_repos)} repo(s)** use **{selected_tech}**:")
        for rname in sorted(matching_repos):
            rdata   = graph["repos"].get(rname, {})
            desc    = rdata.get("description", "")
            cat     = rdata.get("category", "other")
            stack   = rdata.get("technologies", [])
            with st.expander(f"📦 {rname}  ·  *{cat}*", expanded=len(matching_repos) <= 4):
                st.markdown(f"**Description:** {desc}")
                st.markdown(f"**Full tech stack:** {', '.join(stack)}")
                # Similar repos
                similar = repo_similarity(graph, rname, top_n=3)
                if similar:
                    similar_str = ", ".join(f"{s} ({n} shared)" for s, n in similar)
                    st.markdown(f"**Most similar repos:** {similar_str}")
    else:
        st.info(f"No repos found that use {selected_tech}.")

st.divider()

# ── Repo deep dive ────────────────────────────────────────────────────────────

st.subheader("🔬 Repo Deep Dive")
st.markdown(
    "Select any repo to see its full tech stack and discover similar repos based on shared technologies."
)

repo_names_sorted = sorted(graph["repos"].keys())
selected_repo = st.selectbox("Select a repository:", repo_names_sorted, index=0)

if selected_repo:
    rdata  = graph["repos"].get(selected_repo, {})
    desc   = rdata.get("description", "No description.")
    cat    = rdata.get("category", "other")
    stack  = rdata.get("technologies", [])

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(f"**Description:** {desc}")
        st.markdown(f"**Category:** `{cat}`")
        if stack:
            st.markdown("**Tech stack:**")
            for t in stack:
                count = len(graph["technologies"].get(t, {}).get("repos", []))
                st.markdown(f"  - {t} *(used in {count} repo{'s' if count != 1 else ''})*")

    with rc2:
        similar = repo_similarity(graph, selected_repo, top_n=5)
        if similar:
            st.markdown("**Most similar repos (by shared technologies):**")
            for s_name, score in similar:
                s_desc = graph["repos"].get(s_name, {}).get("description", "")
                st.markdown(f"  - **{s_name}** — {score} shared tech{'s' if score != 1 else ''}")
                if s_desc:
                    st.caption(f"    {s_desc}")
        else:
            st.info("No similar repos found (unique tech stack).")

st.divider()

# ── Category breakdown ────────────────────────────────────────────────────────

st.subheader("📂 Projects by Category")
st.markdown(
    """
The knowledge graph automatically groups projects by type — frontend, backend, mobile, etc.
This kind of **taxonomy** is something a knowledge graph excels at and a vector search DB
doesn't provide natively.
"""
)

cats = category_groups(graph)
cat_cols = st.columns(min(len(cats), 4))
for i, (cat_name, cat_repos) in enumerate(sorted(cats.items())):
    with cat_cols[i % len(cat_cols)]:
        st.markdown(f"**{cat_name.title()}** ({len(cat_repos)})")
        for r in sorted(cat_repos):
            st.caption(f"· {r}")

st.divider()

# ── Educational comparison ─────────────────────────────────────────────────────

st.subheader("🎓 Vector Search vs Knowledge Graph — What's the Difference?")

tab1, tab2 = st.tabs(["📌 Side-by-side Comparison", "🔧 When to Use Each"])

with tab1:
    st.markdown(
        """
| Dimension | Vector Search (RAG) | Knowledge Graph |
|-----------|--------------------|-----------------| 
| **What it stores** | Document embeddings (math vectors) | Entities + typed relationships |
| **Query type** | Semantic similarity ("find text *like* this") | Structural traversal ("find repos *that use* X") |
| **Speed** | Milliseconds (ANN search) | Microseconds (dict lookup) |
| **LLM at query time?** | Yes — to formulate answer | No — structure gives the answer |
| **Handles "how many?"** | Poorly (has to count in prose) | Perfectly (it's just `len(repos)`) |
| **Handles "which?"** | Approximately (may miss some) | Exactly (complete relationship list) |
| **Requires structure** | No — works on raw text | Yes — must extract entities first |
| **Best for** | Open-ended Q&A | Inventory, filtering, recommendations |
        """
    )

with tab2:
    st.markdown(
        """
### Use Vector Search (RAG/Agentic) when:
- The question is open-ended: *"Explain how Luke handles authentication"*
- You don't know what you're looking for in advance
- The answer is buried in unstructured prose

### Use a Knowledge Graph when:
- The question is structural: *"How many repos use Python?"*
- You want to filter or group: *"All mobile projects"*
- You need recommendations: *"Find projects similar to X"*

### Use **both together** (as this portfolio does) when:
- You need a precise filter first (graph), then a detailed explanation (RAG)
- Example: graph finds the right repos → RAG answers a specific question about one of them
        """
    )
