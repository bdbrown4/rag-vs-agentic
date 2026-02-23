"""
Fetch READMEs from GitHub for a given user.

This script:
1. Lists all public non-forked repos for the configured GitHub user
2. Fetches the README for each repo
3. Saves them as structured documents ready for embedding
4. Ingests them into ChromaDB

Usage:
    python -m data.fetch_readmes
"""

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.embeddings import chunk_documents
from shared.vector_store import add_documents

load_dotenv()

GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "bdbrown4")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
BASE_URL = "https://api.github.com"


def _headers() -> dict:
    """Build request headers, optionally with auth."""
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def fetch_repos() -> list[dict]:
    """Fetch all public non-forked repos for the GitHub user."""
    repos = []
    page = 1

    while True:
        resp = requests.get(
            f"{BASE_URL}/users/{GITHUB_USERNAME}/repos",
            headers=_headers(),
            params={"per_page": 100, "page": page, "type": "owner", "sort": "updated"},
        )
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        for repo in batch:
            if not repo.get("fork", False):
                repos.append({
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "description": repo.get("description") or "",
                    "language": repo.get("language") or "Unknown",
                    "stars": repo.get("stargazers_count", 0),
                    "topics": repo.get("topics", []),
                    "url": repo.get("html_url", ""),
                    "updated_at": repo.get("updated_at", ""),
                })

        page += 1

    return repos


def fetch_readme(repo_full_name: str) -> str | None:
    """Fetch the decoded README content for a repo."""
    resp = requests.get(
        f"{BASE_URL}/repos/{repo_full_name}/readme",
        headers={**_headers(), "Accept": "application/vnd.github.v3.raw"},
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text


def build_documents(repos: list[dict]) -> list[dict]:
    """Build document objects from repos + their READMEs."""
    documents = []

    for repo in repos:
        print(f"  Fetching README for {repo['name']}...", end=" ")
        readme = fetch_readme(repo["full_name"])

        if not readme:
            print("(no README)")
            continue

        # Build a rich document combining repo metadata + README
        doc_text = f"""# {repo['name']}

**Description:** {repo['description']}
**Language:** {repo['language']}
**Stars:** {repo['stars']}
**Topics:** {', '.join(repo['topics']) if repo['topics'] else 'None'}
**URL:** {repo['url']}

---

{readme}
"""
        documents.append({
            "id": repo["name"],
            "text": doc_text,
            "metadata": {
                "repo_name": repo["name"],
                "language": repo["language"],
                "stars": repo["stars"],
                "description": repo["description"],
                "url": repo["url"],
            },
        })
        print("OK")

    return documents


def main():
    """Fetch READMEs, chunk, embed, and store in ChromaDB."""
    print(f"\n=== Fetching repos for {GITHUB_USERNAME} ===\n")
    repos = fetch_repos()
    print(f"Found {len(repos)} original (non-forked) repos.\n")

    print("=== Fetching READMEs ===\n")
    documents = build_documents(repos)
    print(f"\nFetched {len(documents)} READMEs.\n")

    # Save raw documents for reference
    data_dir = Path(__file__).parent
    with open(data_dir / "documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    print(f"Saved raw documents to {data_dir / 'documents.json'}")

    # Chunk and embed
    print("\n=== Chunking documents ===\n")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.\n")

    print("=== Embedding and storing in ChromaDB ===\n")
    count = add_documents(chunks)
    print(f"Stored {count} chunks in ChromaDB.\n")
    print("Done! Vector store is ready.")


if __name__ == "__main__":
    main()
