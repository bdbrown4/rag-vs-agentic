# Deployment Guide

Two services to deploy:

1. **MCP Server** → [Railway](https://railway.app) (Node.js, always-on)
2. **Streamlit App** → [Streamlit Community Cloud](https://share.streamlit.io) (Python, free tier)

---

## Step 1 — Push repos to GitHub

Both projects need to be in **public** GitHub repos.

```bash
# From c:\dev\rag-vs-agentic
git init
git remote add origin https://github.com/bdbrown4/rag-vs-agentic.git
git add .
git commit -m "initial commit"
git push -u origin main

# From c:\dev\github-portfolio-mcp-server
git add .
git commit -m "add railway.json + HTTP/SSE mode"
git push
```

---

## Step 2 — Deploy the MCP Server to Railway

1. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo**
2. Select `github-portfolio-mcp-server`
3. Railway reads `railway.json` automatically — it runs `npm install && npm run build` then `node dist/index.js`
4. Add environment variables in Railway dashboard (Variables tab):

   | Variable | Value |
   |---|---|
   | `GITHUB_TOKEN` | Your GitHub PAT (`read:public_repo` scope) |
   | `GITHUB_USERNAME` | `bdbrown4` |

   > **Note:** Railway injects `PORT` automatically — do not set it manually.

5. Under **Settings → Networking → Generate Domain** — copy the public URL, e.g.:
   ```
   https://github-portfolio-mcp-server-production.up.railway.app
   ```
6. Verify:
   ```bash
   curl https://github-portfolio-mcp-server-production.up.railway.app/health
   # {"status":"ok","username":"bdbrown4","tools":6}
   ```

---

## Step 3 — Deploy the Streamlit App

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Connect GitHub → select `bdbrown4/rag-vs-agentic`, main file: `app.py`
3. Click **Advanced settings → Secrets** and paste:

```toml
OPENAI_API_KEY = "sk-..."
GITHUB_TOKEN   = "ghp_..."
REQUIRE_AUTH   = "true"
APP_PASSWORD   = "choose-a-strong-password"
MCP_SERVER_URL = "https://github-portfolio-mcp-server-production.up.railway.app"
```

4. Deploy. On the first cold start the app auto-ingests your GitHub repos into ChromaDB (~2–3 min).

---

## Step 4 — Verify end-to-end

1. Open your Streamlit app URL
2. Enter the password you set as `APP_PASSWORD`
3. Ask a question like: *"What languages does Ben use most?"*
4. Both RAG and Agentic tabs should respond — the Agentic tab uses MCP tools since `MCP_SERVER_URL` is set

---

## Local dev reference

```bash
# MCP server (HTTP mode on port 3000)
cd github-portfolio-mcp-server
npm run start:http

# Streamlit app (with MCP wired in, no auth)
cd rag-vs-agentic
$env:MCP_SERVER_URL="http://localhost:3000"
$env:REQUIRE_AUTH="false"
C:\dev\rag-vs-agentic\.venv\Scripts\python.exe -m streamlit run app.py
```
