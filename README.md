# VeriRAG: Enterprise Self-Corrective Agentic RAG with Visual Source Grounding

Agentic RAG for high-stakes enterprise domains (financial disclosures,
legal contracts, compliance audits): hybrid retrieval, a LangGraph
self-correction loop, visual source grounding with PDF bounding-box
citations, and Langfuse observability. Built over 7 daily slices,
Oct 26 to Nov 1, 2026.

## Status: Day 7 (Sun Nov 1, 2026) - complete

All 7 slices built, tested, and verified locally. Backend tests 42/42
offline. Ragas-style eval 14 questions: faithfulness 1.0, citation
recall 1.0, context recall 1.0, answer rate 1.0, decline honesty 1.0,
citation precision 0.87 (see the honesty note in Eval). Next.js 15
frontend builds with zero TypeScript errors.

## Features

- **PDF ingestion with visual grounding**: PyMuPDF extracts words with
  bounding boxes; chunks are 200-word windows with 40-word overlap that
  never cross a page boundary. Every chunk carries `[page, x0, y0, x1,
  y1]` in the index metadata.
- **Hybrid retrieval**: LEANN dense search (HNSW) plus pure-Python BM25
  over the same chunks, fused with reciprocal rank fusion (k=60), then
  Cohere Rerank v3 over the fused top candidates. With no
  `COHERE_API_KEY`, or if the API call fails, retrieval degrades
  gracefully to the fused order (`reranked: false`). A missing index
  returns `[]` with HTTP 200, never an error.
- **Self-correcting LangGraph agent**: retrieve -> grade each chunk's
  relevance -> rewrite the query and retrieve again on a bounded loop
  (max 2 rewrites) when nothing grades relevant -> generate an answer
  whose claims each carry a `[Page, X1, Y1, X2, Y2]` citation -> check
  every claim against its cited chunk, with one bounded regeneration
  that drops unsupported claims, then an honest gap report instead of a
  hallucinated answer.
- **Honest by construction**: an empty index or no relevant context
  returns "I could not find relevant context... I will not guess"
  (`grounded: false`), never a fabricated answer.
- **Streaming chat**: `POST /api/v1/ask/stream` emits one `trace` event
  per agent node, then one `answer` event, then `done` (SSE).
- **Feedback loop**: `POST /api/v1/feedback` stores `{query, answer,
  rating: up|down, comment?}` in `data/feedback.jsonl` (gitignored),
  feeding the eval regression set.
- **Observability**: every ask/search request logged as structured JSON
  to `data/requests.jsonl`; Langfuse tracing activates ONLY when both
  Langfuse keys are set, otherwise it is a no-op.
- **Security**: `X-API-Key` auth against `VERIRAG_API_KEY` on ask,
  ask/stream, search, and feedback (disabled when unset, so local dev
  keeps working); in-memory sliding-window rate limiter
  (`VERIRAG_RATE_LIMIT_PER_MIN`, default 60), HTTP 429 with
  `Retry-After`.
- **Next.js 15 frontend**: split-screen chat (SSE streaming answers,
  live node indicator, trace drawer, grounded/rewrite badges, thumbs
  up/down feedback) plus a document context panel (PDF upload,
  indexed-document list, page/bbox citations). Talks to
  `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

## Architecture

```
PDF upload -> PyMuPDF (words + bboxes) -> page-safe chunker (200w/40 overlap)
    -> LEANN HNSW dense index (+ passages file with chunk metadata)
User question -> LangGraph agent:
    retrieve (LEANN dense + BM25 -> RRF -> Cohere Rerank v3)
    -> grade_documents (relevance score per chunk)
    -> rewrite_query (bounded: max 2 rewrites) --+
    -> generate (one cited claim per relevant chunk) |
    -> check_hallucination (drop unsupported claims, bounded regen)
    -> grounded answer with [Page, X1, Y1, X2, Y2] citations, or an honest decline
```

### The Qdrant-to-LEANN swap

The original spec called for Qdrant Cloud as the dense index. During
Day 1 the dense index was swapped to **LEANN** (MIT-licensed,
HNSW backend): it runs fully locally with no cloud account, no API
key, and no network dependency, which matches the offline-first build
of this project. BM25, RRF, and Cohere Rerank v3 are unchanged, and
the interfaces are narrow enough that Qdrant could be dropped back in
behind the same retriever contract. No `QDRANT_*` settings remain in
config or `.env.example`.

### Deterministic offline components

API keys were not available during the build, so every agent component
(grader, rewriter, generator, hallucination checker) ships with a
deterministic offline default: a token-overlap grader, a content-term
rewriter, an extractive generator (one claim per chunk, from the
chunk's first sentence), and a token-overlap checker. The interfaces
are narrow (`score` / `rewrite` / `generate` / `check`), so LLM-backed
components (e.g. OpenAI GPT-4o-mini / GPT-4o per the spec) can be
injected later through `build_graph(...)` without touching the graph.

## Project structure

```
backend/
  app/
    main.py                 FastAPI app
    api/v1/                 ingest, search, ask, ask_stream, feedback
    core/                   config, observability (JSONL + Langfuse), security
    graph/                  state, components, nodes, workflow (LangGraph)
    services/               parser, chunker, index (LEANN), retriever (BM25+RRF+Cohere)
  eval/                     ragas-style eval harness (offline, deterministic)
  scripts/                  verify_*.py (live checks), make_sample_pdf.py, demo_scenario_{1,2,3}.py
  tests/                    42 offline tests
  Dockerfile
frontend/                   Next.js 15 + TypeScript chat UI (+ Dockerfile)
docs/                       DEPLOY.md, FDE_DEMO_SCENARIOS.md
docker-compose.yml          backend + frontend
```

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
TMPDIR=$HOME/.tmp-pip pip install -r requirements.txt   # torch wheels need the TMPDIR workaround
HF_HUB_DISABLE_XET=1 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
cd frontend
npm install
npm run build && npm run start    # http://localhost:3000
```

Or with Docker: `docker compose up --build` (frontend on :3000,
backend on :8000). Copy `.env.example` to `backend/.env` and fill in
keys locally; the `.env` file is gitignored and keys are never read
from the repo. API docs: http://localhost:8000/docs.

## Usage

```bash
# ingest a PDF
curl -F "file=@doc.pdf;type=application/pdf" http://localhost:8000/api/v1/ingest
# hybrid search
curl -X POST http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' -d '{"query":"termination notice period"}'
# ask the agent
curl -X POST http://localhost:8000/api/v1/ask \
  -H 'Content-Type: application/json' -d '{"question":"what is the termination notice period"}'
```

## Eval (measured, not claimed)

`cd backend && python -m eval.ragas_eval` runs a 14-question
Ragas-style harness (10 answerable, 4 adversarial) against the
deterministic offline pipeline. Results on Nov 1, 2026:

| Metric | Measured |
| --- | --- |
| faithfulness (claims supported by cited chunk) | 1.0 |
| citation precision (cited chunks genuinely relevant) | 0.87 |
| citation recall | 1.0 |
| context recall (expected chunks survive retrieve+grade) | 1.0 |
| answer rate (answerable questions answered grounded) | 1.0 |
| decline honesty (adversarial questions honestly declined) | 1.0 |
| rewrite loop bound (max 2) | respected on all 14 |
| avg latency (offline pipeline, local VM) | 3.55 ms |

Honesty note: this measures pipeline correctness (the grounding
guarantee, citation validity, the rewrite bound), not LLM reasoning
quality. Faithfulness is 1.0 by construction with the extractive
offline generator; the metric's value is regression detection. The
precision dips on two questions are the grader keeping topically
related chunks (e.g. other validation chunks for a cross-validation
question): genuine pipeline behavior. The spec's template benchmark
figures are not repeated anywhere in this repo.

## FDE demo scenarios

Three runnable, one-command demos for a forward-deployed engineer
call; see `docs/FDE_DEMO_SCENARIOS.md`:

1. **Contract Q&A with bbox citations** (`scripts/demo_scenario_1`):
   parse a real 2-page vendor agreement, answer "what is the
   termination notice period" with a page-1 `[Page, X1, Y1, X2, Y2]`
   citation.
2. **Rewrite-loop recovery** (`scripts/demo_scenario_2`): round 1
   misses, the bounded rewrite recovers a grounded answer; the trace
   shows the full loop.
3. **Feedback loop** (`scripts/demo_scenario_3`): a thumbs-down is
   recorded through the real feedback endpoint and read back from
   `data/feedback.jsonl`, feeding the eval regression set.

## Deployment

See `docs/DEPLOY.md`: Docker Compose (recommended first run), plain
local run, and the Vercel-frontend plus hosted-backend route with the
exact environment variables and smoke tests. Keys come from the
environment only.

## Roadmap (all 7 days shipped)

- Day 1: scaffold (FastAPI, config, health).
- Day 2: PDF parser with bbox extraction, page-safe chunker, LEANN
  HNSW index, ingest endpoint. Tests 6/6.
- Day 3: hybrid retrieval (BM25 + RRF + Cohere Rerank v3), search
  endpoint. Tests 18/18.
- Day 4: LangGraph self-correction agent, ask endpoint. Tests 30/30.
- Day 5: SSE streaming, feedback, request logging, Langfuse no-op
  tracing, auth + rate limiting. Tests 42/42.
- Day 6: Next.js 15 frontend (chat, context panel, trace drawer,
  feedback buttons). `next build` clean, zero TS errors.
- Day 7: Ragas-style eval (this README's table), final README,
  Dockerfile + compose, deployment playbook, 3 FDE demo scenarios.
