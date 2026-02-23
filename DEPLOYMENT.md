# Deployment Guide

Two services to deploy:

1. **MCP Server** → [Railway](https://railway.app) (Node.js, always-on)
2. **Streamlit App** → [Streamlit Community Cloud](https://share.streamlit.io) (Python, free tier)

---

## Step 1 — Push repos to GitHub

Both projects need to be in **public** (or private with deploy access) GitHub repos.

```bash
# From c:\dev\rag-vs-agentic
git init
git remote add origin https://github.com/bdbrown4/rag-vs-agentic.git
git add .
git commit -m "initial commit"
git push -u origin main

# From c:\dev\github-portfolio-mcp-server
git add .
git commit -m "add HTTP/SSE + REST proxy mode"
git push
```

---

## Step 2 — Set up Google OAuth credentials

The Streamlit app uses Google OAuth via Streamlit's native `st.login()`.

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project (or use an existing one)
3. **APIs & Services → OAuth consent screen**
   - App name: `RAG vs Agentic`
   - User type: External
   - Add your email as a test user
4. **APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID**
   - Application type: **Web application**
   - Authorized redirect URIs:
     ```
     https://<your-app>.streamlit.app/oauth2callback
     ```
     *(Add `http://localhost:8501/oauth2callback` for local dev too)*
5. Copy **Client ID** and **Client Secret** — you'll need these in Step 4.

---

## Step 3 — Deploy the MCP Server to Railway

1. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo**
2. Select `github-portfolio-mcp-server`
3. Railway auto-detects Node.js. Set **Start Command** to:
   ```
   node dist/index.js
   ```
4. Add environment variables in Railway dashboard:
   | Variable | Value |
   |---|---|
   | `PORT` | `3000` (Railway sets this automatically — just make sure it's present) |
   | `GITHUB_TOKEN` | Your GitHub PAT (read:public_repo scope) |
   | `GITHUB_USERNAME` | `bdbrown4` |
5. Under **Settings → Networking → Generate Domain** — copy the public URL, e.g.:
   ```
   https://github-portfolio-mcp-server.up.railway.app
   ```
6. Verify it's running:
   ```bash
   curl https://github-portfolio-mcp-server.up.railway.app/health
   # {"status":"ok","username":"bdbrown4","tools":6}
   ```

---

## Step 4 — Deploy the Streamlit App

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Connect your GitHub account → select `bdbrown4/rag-vs-agentic`
3. Set **Main file path**: `app.py`
4. Click **Advanced settings** → paste secrets (see template below)
5. Deploy. On first cold start the app will automatically ingest your GitHub repos into ChromaDB (takes ~2–3 min for 600+ files).

### Secrets (paste into Streamlit Cloud "Secrets" UI)

```toml
OPENAI_API_KEY = "sk-..."
GITHUB_TOKEN   = "ghp_..."
REQUIRE_AUTH   = "true"
MCP_SERVER_URL = "https://github-portfolio-mcp-server.up.railway.app"

# Comma-separated list of emails allowed to log in
allowed_emails = ["you@gmail.com", "friend@gmail.com"]

[auth]
redirect_uri = "https://<your-app>.streamlit.app/oauth2callback"
cookie_secret = "<generate a random 32-char string>"

[auth.google]
client_id     = "<Google OAuth Client ID>"
client_secret = "<Google OAuth Client Secret>"
```

> **Generate a cookie secret:** `python -c "import secrets; print(secrets.token_hex(32))"`

---

## Step 5 — Verify end-to-end

1. Open your Streamlit app URL
2. Click **Sign in with Google** — only emails in `allowed_emails` will be allowed through
3. Ask a question like: *"What languages does Ben use most?"*
4. Both RAG and Agentic tabs should respond — the Agentic tab will use MCP tools if `MCP_SERVER_URL` is set

---

## Local dev reference

```bash
# MCP server (HTTP mode on port 3000)
cd github-portfolio-mcp-server
npm run start:http

# Streamlit app (with MCP wired in)
cd rag-vs-agentic
$env:MCP_SERVER_URL="http://localhost:3000"
$env:REQUIRE_AUTH="false"
C:\dev\rag-vs-agentic\.venv\Scripts\python.exe -m streamlit run app.py
```
