# RAG vs Agentic Retrieval

A **production-grade AI engineering demo** comparing classic RAG and a LangGraph agentic pipeline on the same knowledge base. Features live token streaming, automated RAGAS evaluation, model cost comparison, and a knowledge graph explorer — all deployed on Streamlit Cloud with Google OAuth.

**Live demo:** [rag-vs-agentic.streamlit.app](https://rag-vs-agentic-fmm3inchpstkf5eobfgqqb.streamlit.app)

---

## 🎯 Why This Project?

| Dimension | Classic RAG | LangGraph Agentic |
|-----------|-------------|-------------------|
| **Retrieval** | Single pass: embed query → top-k chunks | Iterative: agent plans, then retrieves multiple times |
| **Planning** | None — direct generation | Explicit PLAN node before execution |
| **Tool Use** | Vector search only | Multiple tools (semantic search, full doc retrieval, live APIs) |
| **Multi-hop** | Struggles with cross-repo questions | Chains retrievals across documents with context |
| **Reasoning** | Implicit in LLM response | Explicit PLAN → EXECUTE → TOOL CALLS → SYNTHESIZE trace |
| **Cost** | 1 LLM call | 3–8+ LLM calls |
| **Latency** | Fast (~2s) | Slower (~10s) |
| **Streaming** | Token-level on main response | Plan, tool calls, and answer streamed separately |
| **Guardrails** | Pydantic schema validation + confidence gating | Pydantic validation + uncertainty tracking |

### The Tradeoff

**Classic RAG** wins on **speed and cost** — perfect for latency-sensitive applications.  
**Agentic** wins on **reasoning and accuracy** — ideal when you need transparent decision-making.

This project lets you **see the difference live** and decide which to use.

---

## 🚀 Features

### 🤖 RAG vs Agentic Comparison (Main Page)
- **Live token streaming**: Watch both RAG and agentic pipelines respond in real-time
- **Side-by-side answers**: Compare output quality directly
- **Confidence metrics**: Model confidence scores + uncertainty notes
- **Reasoning traces**: See the agent's step-by-step plan and tool calls
- **Cost tracking**: Per-query token counts and estimated costs

### 📊 Eval Dashboard (`pages/📊_Eval_Dashboard.py`)
- **RAGAS evaluation**: Automatic scoring of faithfulness, answer relevancy, and context precision
- **Streaming results**: Watch scores compute in real-time as you run evals
- **Tier-based questions**: Simple, multi-hop, and ambiguous questions expose different strengths
- **Visualizations**: Scatter plots showing the (faithfulness × relevancy) trade-off space
- **CSV export**: Download results for further analysis

### 🧪 Model Comparison (`pages/🧪_Model_Comparison.py`)
- **Multi-model support**: Run the same question through GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **RAGAS scoring**: Automatic quality assessment for each model
- **Cost at scale**: Calculate monthly spend for different query volumes
- **Fine-tuning ROI**: See the break-even point for fine-tuning a smaller model
- **Side-by-side comparison**: Read raw answers and judge quality yourself

### 🕸️ Knowledge Graph (`pages/🕸️_Knowledge_Graph.py`)
- **Graph construction**: One-click build from ChromaDB chunks using GPT-4o-mini extraction
- **Technology heatmap**: Visualize which repos use which technologies
- **Interactive filters**: Find all repos using a specific technology or language
- **Similarity ranking**: Discover repos with similar tech stacks
- **Educational tabs**: Learn when to use knowledge graphs vs vector search

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Shared Knowledge Base                      │
│  ChromaDB (vector store) + OpenAI text-embedding-3-small     │
│  GitHub portfolio READMEs chunked and indexed                │
│  Metadata: repo_name, file_path, chunk_index                 │
└──────────────┬────────────────────────────┬──────────────────┘
               │                            │
        ┌──────▼──────┐          ┌─────────▼──────────────┐
        │ RAG Pipeline │          │  LangGraph Agentic     │
        │              │          │                        │
        │ Input        │          │ Input question         │
        │  ├─ Embed    │          │  ├─ PLANNER node      │
        │  ├─ Retrieve │          │  │  (mini-model)      │
        │  └─ Generate │          │  ├─ EXECUTOR loop     │
        │              │          │  │  (tool binding)    │
        │ Streaming:   │          │  ├─ TOOLS node        │
        │  token-by-   │          │  │  (semantic search)  │
        │  token via   │          │  ├─ SYNTHESIZER       │
        │  ChatOpenAI. │          │  │  (final answer)     │
        │  stream()    │          │  │                     │
        │              │          │  └─ Streaming: plan,  │
        │ Output:      │          │     tool calls, obs    │
        │  RAGResult   │          │                        │
        │              │          │  Output: AgenticResult │
        └──────┬───────┘          └────────┬────────────────┘
               │                           │
        ┌──────▼───────────────────────────▼─────────────────┐
        │     Streamlit Multi-Page UI with Navigation        │
        │                                                     │
        │  🤖 RAG vs Agentic (main) — live comparison      │
        │  📊 Eval Dashboard — RAGAS + tier questions       │
        │  🧪 Model Comparison — cost vs quality trade-off  │
        │  🕸️  Knowledge Graph — tech relationship explorer  │
        │                                                     │
        │  Auth: Google OAuth (Streamlit Cloud)             │
        │  Tracing: LangSmith (optional, with local fallback)│
        └─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
rag-vs-agentic/
├── data/
│   ├── fetch_readmes.py    # Ingest GitHub portfolio → chunk → embed → ChromaDB
│   ├── traces.jsonl        # LLM call trace log (created at runtime)
│   └── knowledge_graph.json # Knowledge graph (created on demand)
│
├── shared/
│   ├── vector_store.py     # ChromaDB collection operations
│   ├── embeddings.py       # Chunking + text-embedding-3-small
│   ├── metrics.py          # Token counting, confidence labels, query logging
│   ├── guardrails.py       # Pydantic schemas (RAGAnswer, AgenticAnswer)
│   ├── tracing.py          # LangSmith trace writing + summaries
│   └── knowledge_graph.py  # Graph build/query helpers + Altair data
│
├── rag/
│   └── pipeline.py         # RAG: retrieve → generate + streaming variant
│
├── agentic/
│   ├── agent.py            # LangGraph StateGraph (PLAN→EXECUTE→TOOLS→SYNTHESIZE)
│   ├── tools.py            # Vector search + live GitHub API tools
│   └── schema.py           # Agent state definition (TypedDict)
│
├── evaluate/
│   ├── ragas_eval.py       # RAGAS faithfulness/relevancy/precision scoring
│   ├── questions.json      # 30 questions across 3 tiers (simple/multi-hop/ambiguous)
│   └── __init__.py
│
├── pages/
│   ├── 📊_Eval_Dashboard.py      # RAGAS eval UI with streaming + charts
│   ├── 🧪_Model_Comparison.py    # Multi-model comparison + fine-tuning ROI
│   └── 🕸️_Knowledge_Graph.py      # Graph builder + explorer + heatmap
│
├── app.py                  # Main Streamlit app with st.navigation()
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variable template
└── README.md              # This file
```

---

## 🧠 Key Components

### **RAG Pipeline** (`rag/pipeline.py`)
- **Retrieval**: Cosine similarity search via ChromaDB
- **Generation**: GPT-4o with Pydantic structured output
- **Streaming**: `stream_rag_pipeline()` yields per-token events
- **Guardrails**: `RAGAnswer` Pydantic schema + confidence gating (threshold 0.25)
- **Metrics**: Confidence score (avg cosine sim of top-k) + token counting

### **Agentic Pipeline** (`agentic/agent.py`)
- **Architecture**: LangGraph StateGraph with 4 nodes
  - **PLANNER**: GPT-4o-mini generates step-by-step retrieval strategy (cheap + audit-friendly)
  - **EXECUTOR**: GPT-4o with bound tools decides next action (retrieve/done)
  - **TOOLS**: ToolNode executes semantic search or live API calls
  - **SYNTHESIZER**: Generates final answer with Pydantic validation
- **Streaming**: `stream_agentic_pipeline()` yields node-by-node events (PLAN, TOOL_CALL, OBSERVATION, ANSWER)
- **Guardrails**: `AgenticAnswer` schema + uncertainty tracking
- **Iterative**: Runs up to 8 iterations (configurable max_iterations)

### **Evaluation** (`evaluate/ragas_eval.py`)
- **RAGAS metrics**:
  - **Faithfulness**: Does the answer stick to retrieved context? (0–1, higher = better)
  - **Answer Relevancy**: Does the answer address the question? (0–1, higher = better)
  - **Context Precision**: Are retrieved chunks useful? (0–1, higher = better)
- **Questions**: 30 curated questions split into tiers (simple 10, multi-hop 10, ambiguous 10)
- **Interleave**: "All" tier randomly samples across all three for balanced eval

### **Knowledge Graph** (`shared/knowledge_graph.py`)
- **Structure**: Nodes (repos, technologies) + Edges (repo --uses--> tech)
- **Build process**: GPT-4o-mini extracts tech stack from each repo's README
- **Storage**: JSON node-link format (no runtime dependency on networkx)
- **Queries**: Grouping, filtering, similarity, frequency analysis
- **Visualization**: Altair heatmaps, bar charts, tech matrices

### **Guardrails** (`shared/guardrails.py`)
- **RAGAnswer**: `answer`, `sources`, `confidence` (str label), `uncertainty_note`
- **AgenticAnswer**: `answer`, `reasoning_summary`, `tools_used`, `confidence`, `uncertainty_note`
- **Confidence mapping**: "high" → 0.8, "medium" → 0.5, "low" → 0.2, "insufficient-context" → 0.0
- **Gating**: If confidence < 0.25 after retrieval, return a gated response instead of a low-confidence answer

### **Tracing** (`shared/tracing.py`)
- **Storage**: `data/traces.jsonl` (append-only log)
- **LangSmith**: Automatic LangChain callback when `LANGSMITH_API_KEY` is set (optional)
- **Admin panel**: View trace summaries (count, avg cost, avg latency per pipeline)
- **Clear traces**: Button to archival (renames file with timestamp)

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/bdbrown4/rag-vs-agentic
cd rag-vs-agentic
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` with:
```bash
OPENAI_API_KEY=sk-...              # Required
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GITHUB_TOKEN=ghp_...               # Optional (for live GitHub API calls in agentic)
GITHUB_USERNAME=you                # Your GitHub username (for portfolio input)
LANGSMITH_API_KEY=...              # Optional (for LangSmith tracing)
REQUIRE_AUTH=false                 # Set to true on Streamlit Cloud
```

### 3. Ingest Knowledge Base

On first run, the app will auto-ingest if ChromaDB is empty. Or manually:

```bash
python -c "from data.fetch_readmes import main; main()"
```

This fetches your GitHub READMEs, chunks them at 800 tokens, and stores in ChromaDB.

### 4. Run Locally

```bash
streamlit run app.py
```

Opens http://localhost:8501 with the RAG vs Agentic comparison.

**Navigate the pages:**
- 🤖 **RAG vs Agentic** — main comparison with live streaming
- 📊 **Eval Dashboard** — run RAGAS evaluation
- 🧪 **Model Comparison** — compare gpt-4o vs gpt-4o-mini vs gpt-3.5-turbo
- 🕸️ **Knowledge Graph** — explore technology relationships

### 5. Run Evaluation (CLI)

```bash
# Run RAGAS on all 30 questions
python -m evaluate.ragas_eval --all --output eval_results.json

# Filter by tier
python -m evaluate.ragas_eval --tier simple
python -m evaluate.ragas_eval --tier multi-hop
python -m evaluate.ragas_eval --tier ambiguous
```

Results include faithfulness, answer_relevancy, and context_precision for each question on each pipeline.

---

## 📊 Evaluation Questions

The app uses **30 curated questions** across 3 difficulty tiers:

### Simple (10 questions)
*Single-hop retrieval, clear answers*
- "What is the purpose of the portfolio-site project?"
- "Which projects use TypeScript?"
- "List the frameworks Luke has used"

### Multi-hop (10 questions)
*Require combining information from multiple repos*
- "Which backends projects use Python, and what frameworks do they employ?"
- "Compare the Android and Kotlin mobile projects — what do they share?"
- "What is the relationship between the MCP server and the main AI projects?"

### Ambiguous (10 questions)
*Require reasoning, synthesis, or judgment*
- "Is this developer senior-level? Why or why not?"
- "What gaps exist in the technical skill set?"
- "If you were to recommend improvements, what would they be?"

**Expected behavior:**
- **Simple tier**: Both RAG and agentic perform well. RAG slightly faster.
- **Multi-hop tier**: Agentic advantage emerges (chains retrievals).
- **Ambiguous tier**: Agentic shines (explicit reasoning + synthesis).

---

## 🎓 Educational Value

This project demonstrates:

1. **RAG fundamentals**: Embedding, retrieval, generation pipeline with guardrails
2. **Agentic reasoning**: Plan generation, tool binding, iterative refinement (LangGraph)
3. **Streaming UX**: Both pipelines stream their responses for better perceived performance
4. **Evaluation**: RAGAS framework for automatic quality assessment
5. **Cost analysis**: Model comparison exploring fine-tuning ROI
6. **Knowledge graphs**: Structured vs unstructured retrieval trade-offs
7. **Production concerns**: Auth, tracing, metrics, error handling, Streamlit Cloud deployment

---

## 📈 Tech Stack

| Layer | Tools |
|-------|-------|
| **LLM & Embeddings** | OpenAI (gpt-4o, gpt-4o-mini, text-embedding-3-small) |
| **RAG Framework** | LangChain (pipelines, callbacks, output parsing) |
| **Agentic** | LangGraph (StateGraph, nodes, edges) |
| **Vector Store** | ChromaDB (local, persistent) |
| **Guardrails** | Pydantic v2 (structured output validation) |
| **Evaluation** | RAGAS (faithfulness, relevancy, precision) |
| **UI** | Streamlit 1.43+ (multi-page, navigation, auth) |
| **Visualization** | Altair (interactive charts), matplotlib (network graphs) |
| **Knowledge Graphs** | networkx (graph algorithms), JSON persistence |
| **Tracing & Monitoring** | LangSmith (optional), local `traces.jsonl` |
| **Data** | pandas (evaluation results), datasets (HuggingFace) |
| **Infrastructure** | Streamlit Cloud (deployment), Google OAuth (auth) |
| **Auth** | Streamlit native auth + Google OAuth |
| **Development** | Python 3.11+, pip, .env configuration |

---

## 🔗 Companion Projects

- **[github-portfolio-mcp-server](https://github.com/bdbrown4/github-portfolio-mcp-server)** — MCP server exposing these same tools as a Claude Desktop plugin or API
- **[portfolio-site](https://github.com/bdbrown4/portfolio-site)** — Next.js portfolio showcasing the RAG project with embedded demo links

---

## 🐛 Troubleshooting

### **ChromaDB "n_results > matches" error**
Repos with fewer chunks than `n_results` cause ChromaDB to reject the query. **Fix**: App uses `collection.get()` instead of `collection.query()` — automatically returns all available chunks.

### **Pydantic warnings about `parsed` field**
LangChain's `with_structured_output()` stores the parsed model in `AIMessage.parsed`. Pydantic warns the field schema expects `None`. This is benign and suppressed at module load time. No functional impact.

### **LangSmith not showing traces**
Set `LANGSMITH_API_KEY` in `.env`. Without it, the app falls back to local `traces.jsonl` logging — traces still work, just not visible in LangSmith Cloud.

### **Knowledge graph build times out**
The graph builder calls GPT-4o-mini for each repo (~$0.01 total, ~30 repos = $0.30–0.50). If you have 50+ repos or slow API, consider filtering to a subset via `where` clauses in ChromaDB queries.

---

## 📝 License

MIT

---

## 🙋 Questions?

Refer to the embedded educational explanations in each Streamlit page — they're designed for non-technical interviewers to understand what's happening.
