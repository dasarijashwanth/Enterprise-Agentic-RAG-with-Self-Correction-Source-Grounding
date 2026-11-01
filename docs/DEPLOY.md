# VeriRAG deployment playbook

## Architecture recap

- `backend/`: FastAPI + LangGraph agent, LEANN (HNSW) dense index on
  disk, pure-Python BM25, optional Cohere Rerank v3, optional Langfuse.
- `frontend/`: Next.js 15 + TypeScript, static chat UI. The browser
  calls the backend directly, so the frontend build needs
  `NEXT_PUBLIC_API_URL` set to the browser-reachable backend URL.

## Environment variables (all optional; keys come from the environment ONLY)

| Variable | Purpose | Default |
| --- | --- | --- |
| `VERIRAG_API_KEY` | Enables `X-API-Key` auth on ask/search/feedback endpoints | unset = auth disabled |
| `COHERE_API_KEY` | Enables Cohere Rerank v3 over fused candidates | unset = fused order, `reranked: false` |
| `OPENAI_API_KEY` | Reserved for LLM-backed agent components (not yet wired) | unset = deterministic offline components |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Enables Langfuse tracing; both must be set | unset = no-op tracer |
| `VERIRAG_RATE_LIMIT_PER_MIN` | Sliding-window rate limit per client on ask/search/feedback | 60 |
| `NEXT_PUBLIC_API_URL` | Backend URL baked into the frontend build | http://localhost:8000 |
| `NEXT_PUBLIC_VERIRAG_API_KEY` | Optional API key the frontend sends | unset |

Never commit a `.env` file. The repo's `.env.example` lists the names
with empty values.

## Option A: local Docker Compose (recommended first run)

```bash
docker compose up --build
```

- Backend: http://localhost:8000 (health: `GET /health`)
- Frontend: http://localhost:3000

The backend image is large (PyMuPDF, LEANN, torch via LEANN): expect a
multi-GB image and a slow first build. LEANN index data persists in the
`verirag-data` volume.

## Option B: local without Docker

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
cd frontend
npm install
npm run build && npm run start
```

## Option C: frontend on Vercel, backend on a host

The frontend is a standard Next.js 15 app and deploys to Vercel cleanly:

1. Push this repo to GitHub.
2. In Vercel: Add New > Project > import the repo, set the root
   directory to `frontend`.
3. Environment variable: `NEXT_PUBLIC_API_URL` = your backend's public
   URL (e.g. `https://verirag-backend.onrender.com`).
4. Deploy. Vercel gives you a public URL.

The backend needs a host with a persistent disk (the LEANN index lives
on disk) and enough RAM for the embedding model. Free-tier candidates
(their free tiers change; check before committing):

- **Render**: Web Service from the repo, Docker runtime, add a disk
  for `/code/data`. Set the env vars above in the Render dashboard.
  Point `NEXT_PUBLIC_API_URL` at the Render URL and redeploy the
  frontend.
- **Railway** or **Fly.io**: same shape (Dockerfile in `backend/`,
  persistent volume for `/code/data`).

No paid service is required for a demo, and no account should be
created in anyone's name without them. Whoever deploys keeps the keys
in the host's env-var dashboard, never in files.

## Smoke test after any deploy

```bash
curl https://<backend>/health
curl -X POST https://<backend>/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"what is the termination notice period"}'
```

An empty index answers honestly (`grounded: false`, "no documents
have been indexed yet") with HTTP 200. Upload a PDF in the frontend's
Context panel, then ask again for a cited answer.
