"""
Tools for the agentic retrieval agent.

These tools give the agent the ability to:
1. Search the vector store (like RAG, but the agent controls when/how)
2. Get the full content of a specific document
3. List all available documents
4. Fetch live repo info from GitHub (the heterogeneous source advantage)
"""

import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool

from shared.vector_store import query_similar, list_all_documents, get_document_chunks

load_dotenv()

GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "bdbrown4")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def _github_headers() -> dict:
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


@tool
def search_docs(query: str) -> str:
    """
    Search the README knowledge base for chunks relevant to a query.
    Returns the top 5 most similar text chunks with their source repos.
    Use this to find information about specific topics, technologies, or projects.
    """
    results = query_similar(query, n_results=5)
    if not results:
        return "No relevant documents found."

    output = []
    for r in results:
        repo = r["metadata"].get("repo_name", "unknown")
        score = round(1 - r["distance"], 3)  # cosine similarity
        output.append(f"[{repo}] (relevance: {score})\n{r['text'][:300]}...")

    return "\n\n---\n\n".join(output)


@tool
def get_full_document(repo_name: str) -> str:
    """
    Retrieve the full README content for a specific repository by name.
    Use this when you need the complete context of a project, not just a chunk.
    The repo_name should match exactly (e.g., 'portfolio-site', 'pokemon-api').
    """
    chunks = get_document_chunks(repo_name)
    if not chunks:
        return f"No document found for repo '{repo_name}'. Use list_available_docs to see what's available."

    full_text = "\n".join(c["text"] for c in chunks)
    return full_text[:3000]  # Cap at 3000 chars to avoid context overflow


@tool
def list_available_docs() -> str:
    """
    List all repository documents available in the knowledge base.
    Use this to discover what repos exist before searching or retrieving.
    """
    docs = list_all_documents()
    if not docs:
        return "No documents in the knowledge base yet."
    return f"Available repos ({len(docs)}):\n" + "\n".join(f"  - {d}" for d in docs)


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


# Collect all tools for easy import
ALL_TOOLS = [search_docs, get_full_document, list_available_docs, fetch_live_repo_info]
