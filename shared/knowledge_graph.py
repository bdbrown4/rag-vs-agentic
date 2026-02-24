"""
Knowledge Graph — structured relationship map of the portfolio's repos and technologies.

Why a knowledge graph on top of vector search?
  Vector search answers: "Which docs mention React?"
  Knowledge graphs answer: "Which *repos* USE React, and what else do they have in common?"

The graph has two types of nodes:
  - REPO nodes  — one per GitHub repository
  - TECH nodes  — programming languages, frameworks, tools, concepts

And one type of edge:
  - REPO --[uses]--> TECH

This lets us run structured queries like:
  "Find all repos that use both Python and FastAPI"
  "What technologies does this developer use most?"
  "Which repos are most similar based on shared tech?"

Building strategy: query ChromaDB for each repo's README text, then call GPT-4o-mini
to extract the tech stack. This is cheap (~$0.001 per repo) and produces structured data
that pure vector search cannot provide.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_GRAPH_PATH = Path(__file__).parent.parent / "data" / "knowledge_graph.json"

# ── Data model ────────────────────────────────────────────────────────────────
# Stored as JSON for portability (no networkx dependency at runtime).
# Shape:
#   {
#     "repos": {
#       "repo-name": {
#         "description": "...",
#         "technologies": ["React", "TypeScript", ...],
#         "category": "frontend" | "backend" | "mobile" | "fullstack" | "tool" | "other"
#       }
#     },
#     "technologies": {
#       "React": {
#         "category": "frontend-framework",
#         "repos": ["portfolio-site", "crypto-component", ...]
#       }
#     }
#   }


def _empty_graph() -> dict:
    return {"repos": {}, "technologies": {}}


def load_graph() -> dict:
    """Load knowledge graph from disk. Returns empty graph if not built yet."""
    if not _GRAPH_PATH.exists():
        return _empty_graph()
    try:
        return json.loads(_GRAPH_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _empty_graph()


def save_graph(graph: dict) -> None:
    _GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GRAPH_PATH.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Query helpers ─────────────────────────────────────────────────────────────

def repos_using(graph: dict, technology: str) -> list[str]:
    """Return all repo names that use the given technology (case-insensitive)."""
    tech_lower = technology.lower()
    return [
        repo_name
        for repo_name, repo_data in graph["repos"].items()
        if any(t.lower() == tech_lower for t in repo_data.get("technologies", []))
    ]


def technologies_in(graph: dict, repo_name: str) -> list[str]:
    """Return the tech stack for a given repo."""
    return graph["repos"].get(repo_name, {}).get("technologies", [])


def shared_technologies(graph: dict, repo_a: str, repo_b: str) -> list[str]:
    """Technologies that two repos have in common."""
    a_techs = set(t.lower() for t in technologies_in(graph, repo_a))
    b_techs = set(t.lower() for t in technologies_in(graph, repo_b))
    shared = a_techs & b_techs
    return [t for t in technologies_in(graph, repo_a) if t.lower() in shared]


def technology_frequency(graph: dict) -> list[tuple[str, int]]:
    """Return (technology, repo_count) sorted by frequency descending."""
    freq: dict[str, int] = {}
    for repo_data in graph["repos"].values():
        for tech in repo_data.get("technologies", []):
            freq[tech] = freq.get(tech, 0) + 1
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)


def repo_similarity(graph: dict, repo_name: str, top_n: int = 5) -> list[tuple[str, int]]:
    """Find repos most similar to the given one, by shared technology count."""
    target_techs = set(t.lower() for t in technologies_in(graph, repo_name))
    if not target_techs:
        return []
    scores: list[tuple[str, int]] = []
    for other_name, other_data in graph["repos"].items():
        if other_name == repo_name:
            continue
        other_techs = set(t.lower() for t in other_data.get("technologies", []))
        overlap = len(target_techs & other_techs)
        if overlap > 0:
            scores.append((other_name, overlap))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]


def category_groups(graph: dict) -> dict[str, list[str]]:
    """Group repos by their category."""
    groups: dict[str, list[str]] = {}
    for repo_name, repo_data in graph["repos"].items():
        cat = repo_data.get("category", "other")
        groups.setdefault(cat, []).append(repo_name)
    return groups


# ── Graph builder (calls GPT-4o-mini) ─────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a technical analyst. Given a GitHub repository's README text, extract its tech stack.

Return a JSON object with these fields:
  "description": one sentence describing the project (max 15 words)
  "technologies": list of strings — programming languages, frameworks, libraries, tools, APIs, databases
  "category": one of: "frontend", "backend", "fullstack", "mobile", "tool", "data", "other"

Rules:
  - Only list technologies actually mentioned or strongly implied in the README
  - Normalize names: "React.js" → "React", "node" → "Node.js", "ts" → "TypeScript"
  - Include the primary language even if not explicitly stated (infer from file extensions/code)
  - Keep technologies list focused: 3-12 items, no duplicates

README text:
---
{readme_text}
---

Respond with ONLY the JSON object, no markdown fences."""


def build_graph_from_kb(log_fn=None) -> dict:
    """
    Build the knowledge graph by querying ChromaDB and calling GPT-4o-mini.

    This runs once and saves to disk. Re-call to refresh after KB changes.

    Args:
        log_fn: Optional callable(str) for progress logging (e.g. st.write)
    """
    from shared.vector_store import get_or_create_collection
    from langchain_openai import ChatOpenAI
    import re

    def _log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    collection = get_or_create_collection()

    # Get all unique repo names from the collection
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []
    repo_names = list({m.get("repo_name", "") for m in metadatas if m.get("repo_name")})

    _log(f"Found {len(repo_names)} repos in knowledge base. Extracting tech stacks…")
    graph = _empty_graph()

    for i, repo_name in enumerate(sorted(repo_names)):
        _log(f"  [{i+1}/{len(repo_names)}] Analyzing: {repo_name}")

        # Pull the top chunks for this repo from ChromaDB
        repo_results = collection.query(
            query_texts=[f"technology stack framework language {repo_name}"],
            n_results=5,
            where={"repo_name": repo_name},
        )
        docs = repo_results.get("documents", [[]])[0]
        readme_text = "\n\n".join(docs)[:3000]  # cap at 3K chars to keep cost low

        if not readme_text.strip():
            _log(f"    ⚠️  No content found for {repo_name}, skipping")
            continue

        try:
            resp = llm.invoke(_EXTRACTION_PROMPT.format(readme_text=readme_text))
            raw = resp.content.strip()
            # Strip markdown fences if model disobeyed
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)

            description  = str(data.get("description", ""))
            technologies = [str(t) for t in data.get("technologies", [])]
            category     = str(data.get("category", "other"))

            # Add repo node
            graph["repos"][repo_name] = {
                "description":  description,
                "technologies": technologies,
                "category":     category,
            }

            # Add/update technology nodes
            for tech in technologies:
                if tech not in graph["technologies"]:
                    graph["technologies"][tech] = {"repos": []}
                if repo_name not in graph["technologies"][tech]["repos"]:
                    graph["technologies"][tech]["repos"].append(repo_name)

            _log(f"    ✅  {repo_name}: {category} — {', '.join(technologies[:5])}")

        except Exception as exc:
            _log(f"    ❌  {repo_name}: extraction failed — {exc}")

    save_graph(graph)
    _log(f"\n✅ Knowledge graph built: {len(graph['repos'])} repos, {len(graph['technologies'])} technologies")
    return graph


# ── Altair visualisation helpers ──────────────────────────────────────────────

def graph_to_edge_dataframe(graph: dict):
    """
    Convert the graph to a pandas DataFrame of edges for Altair visualisation.
    Each row = one (repo, technology) pair.
    """
    import pandas as pd
    rows = []
    for repo_name, repo_data in graph["repos"].items():
        category = repo_data.get("category", "other")
        for tech in repo_data.get("technologies", []):
            rows.append({
                "repo":     repo_name,
                "tech":     tech,
                "category": category,
            })
    return pd.DataFrame(rows)


def technology_heatmap_data(graph: dict):
    """
    Returns a DataFrame suitable for an Altair heatmap:
    rows = repos, columns = top-20 technologies, values = 1/0
    """
    import pandas as pd
    top_techs = [t for t, _ in technology_frequency(graph)[:20]]
    rows = []
    for repo_name, repo_data in graph["repos"].items():
        repo_techs = set(repo_data.get("technologies", []))
        row = {"repo": repo_name, "category": repo_data.get("category", "other")}
        for tech in top_techs:
            row[tech] = 1 if tech in repo_techs else 0
        rows.append(row)

    df_wide = pd.DataFrame(rows)
    # Melt to long form for Altair
    df_long = df_wide.melt(id_vars=["repo", "category"], var_name="technology", value_name="uses")
    return df_long
