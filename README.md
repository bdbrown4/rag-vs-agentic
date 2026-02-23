# RAG vs Agentic Retrieval

A side-by-side comparison showing the differences between **classic RAG** (Retrieval-Augmented Generation) and **agentic retrieval** using the same knowledge base — your GitHub portfolio READMEs.

## Why This Project?

| | Classic RAG | Agentic Retrieval |
|---|---|---|
| **Retrieval** | Single pass: embed query → top-k chunks | Iterative: agent decides when/what to retrieve |
| **Query Refinement** | None — one shot | Agent can rephrase after poor results |
| **Tool Use** | Vector search only | Multiple tools (search, full doc, live API) |
| **Multi-hop** | Struggles with cross-doc questions | Chains retrievals across documents |
| **Reasoning** | Implicit | Explicit thought → action → observation trace |
| **Cost** | 1 LLM call | 3-8+ LLM calls |
| **Latency** | Fast (~1-3s) | Slower (~5-15s) |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Shared Layer                     │
│  ChromaDB (vector store) + OpenAI Embeddings     │
│  Knowledge Base: GitHub Portfolio READMEs         │
└──────────┬──────────────────────┬────────────────┘
           │                      │
    ┌──────▼──────┐       ┌──────▼──────────┐
    │  Classic RAG │       │  Agentic (ReAct) │
    │             │       │                  │
    │ Query       │       │ Thought          │
    │  → Retrieve │       │  → Action        │
    │  → Generate │       │  → Observation   │
    │  → Answer   │       │  → ...repeat...  │
    │             │       │  → Final Answer  │
    │ (1 LLM call)│       │ (N LLM calls)   │
    └──────┬──────┘       └──────┬───────────┘
           │                      │
    ┌──────▼──────────────────────▼────────────┐
    │         Streamlit Comparison UI            │
    │   Side-by-side answers + metrics + traces │
    └───────────────────────────────────────────┘
```

## Project Structure

```
rag-vs-agentic/
├── data/
│   └── fetch_readmes.py    # Fetch READMEs from GitHub → chunk → embed → store
├── shared/
│   ├── embeddings.py       # Chunking + OpenAI embedding logic
│   └── vector_store.py     # ChromaDB operations
├── rag/
│   └── pipeline.py         # Classic retrieve-then-generate pipeline
├── agentic/
│   ├── tools.py            # Agent tools: search, full doc, live API, etc.
│   └── agent.py            # ReAct agent with LangChain
├── evaluate/
│   ├── questions.json      # Test questions across 3 difficulty tiers
│   └── compare.py          # Run both pipelines, compare results
├── app.py                  # Streamlit UI — side-by-side comparison
├── requirements.txt
└── .env.example
```

## Quick Start

### 1. Setup

```bash
cd rag-vs-agentic
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 3. Ingest Data

Fetch your GitHub READMEs, chunk them, and store in ChromaDB:

```bash
python -m data.fetch_readmes
```

### 4. Run the Comparison UI

```bash
streamlit run app.py
```

### 5. Run the CLI Comparison

```bash
# All questions
python -m evaluate.compare

# Filter by tier
python -m evaluate.compare --tier simple
python -m evaluate.compare --tier multi-hop
python -m evaluate.compare --tier ambiguous

# Save results
python -m evaluate.compare --output results.json --verbose
```

## Question Tiers

The evaluation uses 3 tiers of questions designed to expose the tradeoffs:

| Tier | Example | RAG Expected | Agentic Expected |
|------|---------|-------------|-----------------|
| **Simple** | "What is the portfolio-site project about?" | Works well | Works, but more overhead |
| **Multi-hop** | "Which projects use Angular, and how do they differ?" | Incomplete (single retrieval) | Complete (chains lookups) |
| **Ambiguous** | "Is this developer experienced enough for a full-stack role?" | Noisy/shallow | Explores, then synthesizes |

## Tech Stack

- **Python 3.11+**
- **OpenAI** — gpt-4o (generation) + text-embedding-3-small (embeddings)
- **ChromaDB** — vector storage
- **LangChain** — ReAct agent framework
- **Streamlit** — comparison UI
- **GitHub API** — data source (+ live queries for agentic pipeline)

## Connection to MCP Server

The agentic pipeline includes a `fetch_live_repo_info` tool that queries the GitHub API directly — similar to the tools exposed by the companion [github-portfolio-mcp-server](../github-portfolio-mcp-server). This demonstrates a key agentic advantage: **heterogeneous source retrieval** — pulling from both a vector store and a live API within the same reasoning chain.
