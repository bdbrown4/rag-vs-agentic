# Future Improvements

## 1. Agentic Token Tracking
`AgenticResult.total_tokens` is always `0` because LangChain's `AgentExecutor` doesn't surface per-call token counts. Wrap agent calls with `get_openai_callback()` context manager to capture actual usage.

## 2. Empty Collection Guard
Running the app before ingesting data produces confusing empty results. Add a startup check in `app.py` and `evaluate/compare.py` (or in `query_similar`) that warns/errors gracefully if no documents are ingested.

## 3. Add `data/__init__.py`
The `data/` directory has no `__init__.py`, inconsistent with `agentic/`, `rag/`, and `shared/` packages.

## 4. Environment Validation on Startup
Missing `OPENAI_API_KEY` causes a generic error. Add explicit validation at startup with a clear error message.

## 5. Make ChromaDB `PERSIST_DIR` Absolute
`PERSIST_DIR = ".chroma"` is relative, so it depends on the working directory. Resolve it relative to the project root: `Path(__file__).parent.parent / ".chroma"`.

## 6. Truncation Awareness in `get_full_document`
`get_full_document` in `agentic/tools.py` caps output at 3000 chars without telling the agent. Append a note like `"[TRUNCATED — showing first 3000 of {len(full_text)} chars]"` so the agent can act on it.

## 7. Verify Question Repo Names
Some questions in `evaluate/questions.json` reference repos (`pokemon-api`, `MEANAuthApp`, etc.) that may not exist for the configured GitHub user. Audit and update to match real repos.

## 8. Add Unit Tests
No tests exist. At minimum, test chunking logic, vector store operations (with a temporary collection), and the context-building function.
