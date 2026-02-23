"""
Fetch all source files from GitHub repos for a given user.

This script:
1. Lists all public non-forked repos for the configured GitHub user
2. Fetches the full file tree for each repo
3. Downloads text-based source files (skipping binaries, lockfiles, build artifacts)
4. Saves them as structured documents ready for embedding
5. Ingests them into ChromaDB

Usage:
    python -m data.fetch_readmes
"""

import json
import os
import sys
import time
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

# ── File filtering ───────────────────────────────────────────────────
# Extensions we consider "text / source" (case-insensitive)
TEXT_EXTENSIONS = {
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".kts",
    ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
    ".swift", ".dart", ".lua", ".sh", ".bash", ".ps1",
    ".sql", ".r", ".scala", ".zig", ".ex", ".exs", ".clj",
    # Web / markup
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".vue", ".svelte", ".astro",
    # Config / data (only meaningful ones — not every .json is useful)
    ".yaml", ".yml", ".toml", ".env.example",
    ".prisma", ".proto", ".graphql", ".gql",
    # Docs
    ".md", ".mdx", ".txt", ".rst", ".adoc",
    # Build / infra
    ".gradle", ".tf", ".hcl", ".dockerfile",
}

# Exact filenames we always want (no extension match needed)
TEXT_FILENAMES = {
    "Dockerfile", "Makefile", "CMakeLists.txt", "Procfile",
    "Gemfile", "Rakefile", "Justfile", "Vagrantfile",
    ".gitignore", ".dockerignore",
    "go.mod",
    "package.json", "pyproject.toml", "Cargo.toml",
    "next.config.js", "next.config.mjs", "next.config.ts",
    "vite.config.js", "vite.config.ts",
    "tailwind.config.js", "tailwind.config.ts",
    "webpack.config.js", "webpack.config.ts",
    ".env.example",
}

# Directories to skip entirely
SKIP_DIRS = {
    # Dependencies / build output
    "node_modules", ".next", ".nuxt", "dist", "build", "out",
    "__pycache__", ".git", ".idea", ".vscode",
    "vendor", "target", ".dart_tool", ".pub-cache",
    "Pods", ".build", "DerivedData",
    "bin", "obj", ".terraform",
    # Gradle internals (the wrapper JARs, caches — not useful for RAG)
    ".gradle", "gradle",
    # CI / tooling config dirs
    ".github", ".circleci", ".husky", ".changeset",
    # Test artifacts
    "coverage", "__fixtures__", "__mocks__", "e2e",
    # Generated / minified assets
    "generated", "gen", "proto_generated",
}

# Files to always skip (exact filename match)
SKIP_FILES = {
    # Lockfiles
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "composer.lock", "Gemfile.lock", "Cargo.lock",
    "poetry.lock", "Pipfile.lock", "go.sum",
    # Gradle wrappers (binary shell scripts, not informative)
    "gradlew", "gradlew.bat",
    # Boilerplate build / linting configs (auto-generated, rarely customised)
    "angular.json",
    "karma.conf.js",
    "babel.config.js", "babel.config.cjs", "babel.config.mjs",
    "postcss.config.js", "postcss.config.cjs", "postcss.config.mjs",
    "jest.config.js", "jest.config.ts", "jest.config.cjs",
    ".eslintignore", ".npmignore", ".prettierignore",
    # tsconfig variants that are just framework boilerplate
    "tsconfig.app.json", "tsconfig.spec.json", "tsconfig.node.json",
}

MAX_FILE_SIZE_BYTES = 50_000  # Skip files larger than 50 KB
MAX_FILES_PER_REPO = 250      # Raised — smart filtering keeps noise low


def _headers() -> dict:
    """Build request headers, optionally with auth."""
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _should_include_file(path: str, size: int) -> bool:
    """Decide whether a file path is a text source file worth embedding."""
    filename = path.rsplit("/", 1)[-1] if "/" in path else path

    # Skip known unwanted files
    if filename in SKIP_FILES:
        return False

    # Skip files in unwanted directories
    parts = path.split("/")
    for part in parts[:-1]:  # all directory segments
        if part in SKIP_DIRS:
            return False

    # Skip oversized files
    if size > MAX_FILE_SIZE_BYTES:
        return False

    # Include if filename matches exactly
    if filename in TEXT_FILENAMES:
        return True

    # Include if extension matches
    ext = os.path.splitext(filename)[1].lower()
    if ext in TEXT_EXTENSIONS:
        return True

    # Special case: dotfiles that are often config
    if filename.startswith(".") and ext in {"", ".json", ".yaml", ".yml"}:
        return True

    return False


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
                    "default_branch": repo.get("default_branch", "main"),
                    "updated_at": repo.get("updated_at", ""),
                })

        page += 1

    return repos


def fetch_file_tree(repo: dict) -> list[dict]:
    """Fetch the full recursive file tree for a repo using the Git Trees API."""
    branch = repo["default_branch"]
    resp = requests.get(
        f"{BASE_URL}/repos/{repo['full_name']}/git/trees/{branch}",
        headers=_headers(),
        params={"recursive": "1"},
    )
    if resp.status_code == 409:
        # Empty repo
        return []
    resp.raise_for_status()
    data = resp.json()
    return [
        {"path": item["path"], "size": item.get("size", 0)}
        for item in data.get("tree", [])
        if item["type"] == "blob"
    ]


def fetch_file_content(repo_full_name: str, file_path: str) -> str | None:
    """Fetch raw file content from GitHub."""
    resp = requests.get(
        f"https://raw.githubusercontent.com/{repo_full_name}/HEAD/{file_path}",
        headers=_headers(),
    )
    if resp.status_code != 200:
        return None
    try:
        return resp.text
    except UnicodeDecodeError:
        return None  # Binary file slipped through


def build_documents(repos: list[dict], log_fn=print) -> list[dict]:
    """Build document objects from repos + all their source files."""
    documents = []
    total_files = 0

    for repo in repos:
        files = fetch_file_tree(repo)
        text_files = [f for f in files if _should_include_file(f["path"], f["size"])]

        # Prioritize: READMEs and top-level files first, then by depth
        def _sort_key(f: dict) -> tuple:
            p = f["path"]
            is_readme = p.lower().endswith("readme.md") or p.lower() == "readme"
            depth = p.count("/")
            return (0 if is_readme else 1, depth, p)

        text_files.sort(key=_sort_key)

        if len(text_files) > MAX_FILES_PER_REPO:
            log_fn(f"**{repo['name']}** — {len(text_files)} eligible of {len(files)} total, capped to {MAX_FILES_PER_REPO}")
            text_files = text_files[:MAX_FILES_PER_REPO]
        else:
            log_fn(f"**{repo['name']}** — ingesting {len(text_files)} of {len(files)} total ({len(files) - len(text_files)} filtered out)")

        repo_file_count = 0
        for file_info in text_files:
            fpath = file_info["path"]
            content = fetch_file_content(repo["full_name"], fpath)
            if not content or not content.strip():
                continue

            # Detect language from extension
            ext = os.path.splitext(fpath)[1].lower()

            doc_text = f"""# {repo['name']} — {fpath}

**Repository:** {repo['name']}
**File:** {fpath}
**Description:** {repo['description']}
**Primary Language:** {repo['language']}
**Stars:** {repo['stars']}
**URL:** {repo['url']}

---

```
{content}
```
"""
            doc_id = f"{repo['name']}/{fpath}"
            documents.append({
                "id": doc_id,
                "text": doc_text,
                "metadata": {
                    "repo_name": repo["name"],
                    "file_path": fpath,
                    "language": repo["language"],
                    "file_extension": ext,
                    "stars": repo["stars"],
                    "description": repo["description"],
                    "url": repo["url"],
                },
            })
            repo_file_count += 1

        total_files += repo_file_count
        log_fn(f"  ↳ fetched {repo_file_count} files")

        # Small delay to stay under rate limits
        time.sleep(0.1)

    return documents


def main(log_fn=print):
    """Fetch all source files, chunk, embed, and store in ChromaDB."""
    log_fn(f"--- Fetching repos for **{GITHUB_USERNAME}** ---")
    repos = fetch_repos()
    log_fn(f"Found **{len(repos)}** original (non-forked) repos.")

    log_fn("--- Fetching source files ---")
    documents = build_documents(repos, log_fn=log_fn)
    log_fn(f"Fetched **{len(documents)}** files across all repos.")

    # Save raw documents for reference
    data_dir = Path(__file__).parent
    with open(data_dir / "documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    # Chunk and embed
    log_fn("--- Chunking documents ---")
    chunks = chunk_documents(documents)
    log_fn(f"Created **{len(chunks)}** chunks from **{len(documents)}** files.")

    log_fn("--- Embedding and storing in ChromaDB (slowest step) ---")
    count = add_documents(chunks)
    log_fn(f"✅ Stored **{count}** chunks. Vector store is ready.")


if __name__ == "__main__":
    main()
