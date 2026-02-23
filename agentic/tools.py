"""
Tools for the agentic retrieval agent.

These tools give the agent the ability to:
1. Search the vector store (like RAG, but the agent controls when/how)
2. Get the full content of a specific document
3. List all available documents
4. Fetch live repo info from GitHub (the heterogeneous source advantage)
5. (Optional) Call tools on the hosted GitHub Portfolio MCP server when
   MCP_SERVER_URL is configured — enables richer GitHub queries like
   get_profile, get_tech_stack_summary, search_repos, get_languages.
"""

import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

from shared.vector_store import query_similar, list_all_documents, get_document_chunks

load_dotenv()

GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "bdbrown4")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "").rstrip("/")


def _github_headers() -> dict:
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _call_mcp(tool_name: str, args: dict | None = None) -> str:
    """
    Call a tool on the hosted MCP server's REST proxy endpoint.
    Expects the MCP server to expose POST /call/:tool_name.
    """
    if not MCP_SERVER_URL:
        return f"MCP server not configured (set MCP_SERVER_URL env var)."
    url = f"{MCP_SERVER_URL}/call/{tool_name}"
    try:
        resp = requests.post(url, json={"args": args or {}}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Extract text content from MCP response format
        content = data.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(texts) or str(data)
    except requests.exceptions.ConnectionError:
        return f"Could not connect to MCP server at {MCP_SERVER_URL}."
    except Exception as e:
        return f"MCP call failed: {e}"


# ── Vector store tools ────────────────────────────────────────────────────────

@tool
def search_docs(query: str) -> str:
    """
    Search the source code and documentation knowledge base for chunks relevant to a query.
    Returns the top 5 most similar text chunks with their source repos and file paths.
    Use this to find information about specific topics, technologies, implementations, or projects.
    """
    results = query_similar(query, n_results=5)
    if not results:
        return "No relevant documents found."

    output = []
    for r in results:
        repo = r["metadata"].get("repo_name", "unknown")
        fpath = r["metadata"].get("file_path", "")
        score = round(1 - r["distance"], 3)  # cosine similarity
        label = f"{repo}/{fpath}" if fpath else repo
        output.append(f"[{label}] (relevance: {score})\n{r['text'][:300]}...")

    return "\n\n---\n\n".join(output)


@tool
def get_full_document(doc_id: str) -> str:
    """
    Retrieve the full content of a specific document by its ID.
    The doc_id is typically 'repo_name/file_path' (e.g., 'portfolio-site/README.md',
    'pokemon-api/src/index.ts'). Use list_available_docs to discover valid IDs.
    Use this when you need the complete context of a file, not just a chunk.
    """
    chunks = get_document_chunks(doc_id)
    if not chunks:
        return f"No document found for '{doc_id}'. Use list_available_docs to see what's available."

    full_text = "\n".join(c["text"] for c in chunks)
    if len(full_text) > 3000:
        return full_text[:3000] + f"\n\n[TRUNCATED — showing first 3000 of {len(full_text)} chars]"
    return full_text


@tool
def list_available_docs() -> str:
    """
    List all source files and documents available in the knowledge base.
    Use this to discover what repos and files exist before searching or retrieving.
    Returns document IDs in 'repo_name/file_path' format.
    """
    docs = list_all_documents()
    if not docs:
        return "No documents in the knowledge base yet."

    # Group by repo for readability
    repos: dict[str, list[str]] = {}
    for d in docs:
        parts = d.split("/", 1)
        repo = parts[0]
        fpath = parts[1] if len(parts) > 1 else d
        repos.setdefault(repo, []).append(fpath)

    lines = [f"Available files across {len(repos)} repos ({len(docs)} total files):"]
    for repo, files in sorted(repos.items()):
        lines.append(f"\n  [{repo}] ({len(files)} files)")
        for f in sorted(files)[:15]:  # Cap per-repo listing
            lines.append(f"    - {f}")
        if len(files) > 15:
            lines.append(f"    ... and {len(files) - 15} more")
    return "\n".join(lines)


@tool
def fetch_live_repo_info(repo_name: str) -> str:
    """
    Fetch LIVE information about a repo directly from the GitHub API.
    This gets real-time data including latest commits, open issues count, etc.
    Use this when you need current information that might not be in the README knowledge base.
    The repo_name should be just the repo name (e.g., 'portfolio-site'), not the full path.
    """
    resp = requests.get(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}",
        headers=_github_headers(),
    )
    if resp.status_code == 404:
        return f"Repository '{repo_name}' not found on GitHub."
    resp.raise_for_status()
    data = resp.json()

    # Also fetch languages
    lang_resp = requests.get(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/languages",
        headers=_github_headers(),
    )
    languages = lang_resp.json() if lang_resp.ok else {}

    return f"""Repository: {data['name']}
Description: {data.get('description') or 'None'}
Language: {data.get('language') or 'Unknown'}
Stars: {data.get('stargazers_count', 0)}
Forks: {data.get('forks_count', 0)}
Open Issues: {data.get('open_issues_count', 0)}
Last Updated: {data.get('updated_at', 'Unknown')}
Topics: {', '.join(data.get('topics', []))}
Languages: {', '.join(f'{k}: {v} bytes' for k, v in languages.items())}
URL: {data.get('html_url', '')}"""


# ── MCP server tools (loaded dynamically if MCP_SERVER_URL is set) ────────────

@tool
def mcp_get_profile() -> str:
    """
    Get the developer's full GitHub profile via the Portfolio MCP server.
    Returns bio, location, company, follower count, website, and more.
    Requires MCP_SERVER_URL to be configured.
    """
    return _call_mcp("get_profile")


@tool
def mcp_get_tech_stack() -> str:
    """
    Get an aggregated summary of all programming languages and topics used across every
    repository in the portfolio, via the Portfolio MCP server.
    Great for high-level questions like 'what does this developer know?'.
    Requires MCP_SERVER_URL to be configured.
    """
    return _call_mcp("get_tech_stack_summary")


@tool
def mcp_search_repos(query: str) -> str:
    """
    Search the developer's GitHub repositories by keyword using the Portfolio MCP server.
    Matches against repo name, description, and topics.
    Use this for finding repos related to a specific technology or domain.
    Requires MCP_SERVER_URL to be configured.
    """
    return _call_mcp("search_repos", {"query": query})


@tool
def mcp_get_repo_details(repo_name: str) -> str:
    """
    Get detailed information about a specific repository including its full README,
    via the Portfolio MCP server. Use this for deep-dives into a single project.
    Requires MCP_SERVER_URL to be configured.
    """
    return _call_mcp("get_repo_details", {"repo_name": repo_name})


# ── Tool registry ─────────────────────────────────────────────────────────────

# Base tools always available
_BASE_TOOLS = [search_docs, get_full_document, list_available_docs, fetch_live_repo_info]

# MCP tools — only included when an MCP server is wired up
_MCP_TOOLS = (
    [mcp_get_profile, mcp_get_tech_stack, mcp_search_repos, mcp_get_repo_details]
    if MCP_SERVER_URL
    else []
)

ALL_TOOLS = _BASE_TOOLS + _MCP_TOOLS

