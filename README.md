# VeriRAG: Enterprise Self-Corrective Agentic RAG with Visual Source Grounding

Production-grade agentic RAG for high-stakes enterprise domains (financial
disclosures, legal contracts, compliance audits), built exactly per the
project spec: hybrid retrieval, LangGraph self-correction, visual source
grounding with PDF bounding-box citations, and Langfuse observability.

## Status: Day 3 (Wed Oct 28, 2026) — Day 2 pushed, Day 3 built locally

Day 1 scaffold plus Day 2 ingestion (pushed as `7a0eab5`) plus Day 3
hybrid retrieval, built and verified locally:

- `POST /api/v1/ingest` accepts a PDF upload, parses it with PyMuPDF
  (text plus word bounding boxes), chunks it into overlapping
  word windows that never cross a page boundary, and builds a local
  LEANN dense index (HNSW backend). Every chunk keeps
  `[page, x0, y0, x1, y1]` grounding in the index metadata.
- LEANN replaces Qdrant Cloud as the dense index (approved deviation;
  Day 7 README will document the swap). No `QDRANT_*` settings remain
  in config or `.env.example`.
- `POST /api/v1/search` runs hybrid retrieval: LEANN dense search plus
  BM25 over the same chunks, fused with reciprocal rank fusion, then
  Cohere Rerank v3 over the fused top candidates. With no
  `COHERE_API_KEY` set, or if the API call fails, retrieval degrades
  gracefully to the fused order (`reranked: false`), so the endpoint
  works fully offline. Every hit returns its chunk's page/bbox
  grounding for Day 4's agent citations.

## Run locally

```bash
cd backend
~/.venv-verirag/bin/python -m pip install -r requirements.txt
TMPDIR=$HOME/.tmp-pip HF_HUB_DISABLE_XET=1 \
  ~/.venv-verirag/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs. Health check: http://localhost:8000/health.
Ingest a PDF: `curl -F "file=@doc.pdf;type=application/pdf" http://localhost:8000/api/v1/ingest`.

Notes:

- `HF_HUB_DISABLE_XET=1` is required on this VM: the egress proxy
  drops HuggingFace xet token fetches made by the `requests` session,
  so the first embedding-model download fails without it. The model
  (`BAAI/bge-small-en-v1.5`) is cached after the first build.
- `TMPDIR=$HOME/.tmp-pip` is required for the first `pip install
  leann`: its torch/CUDA wheels overflow the 512 MB /tmp tmpfs.
- `data/` (built indexes) is gitignored; re-ingest rebuilds it.

## Verify Day 2

```bash
cd backend
~/.venv-verirag/bin/python -m pytest tests/ -q   # 18 tests, offline
TMPDIR=$HOME/.tmp-pip HF_HUB_DISABLE_XET=1 \
  ~/.venv-verirag/bin/python scripts/verify_ingest.py  # live endpoint + LEANN search
TMPDIR=$HOME/.tmp-pip HF_HUB_DISABLE_XET=1 \
  ~/.venv-verirag/bin/python scripts/verify_search.py  # live hybrid search
```

The live scripts prove page/bbox metadata round-trips: a
"bounding boxes" query returns the page 2 chunk first, a
"cross validation" query returns the page 1 chunk first, each with
its bbox, and the search script proves the fused ranking carries
dense rank, BM25 rank, and RRF score on every hit.

## Roadmap (from the spec, one slice per day)

- Day 2: Document ingestion, PDF parser with bounding-box extraction, ingest endpoint, LEANN indexing (HNSW).
- Day 3: Hybrid retrieval, dense plus BM25 with reciprocal rank fusion, Cohere Rerank v3.
- Day 4: LangGraph agent, state, retrieval, relevance grading, query rewrite, generation with hallucination check, self-correction loop.
- Day 5: SSE streaming chat endpoint, feedback and logging, Langfuse tracing, auth and rate limits.
- Day 6: Next.js frontend, split-screen layout, chat box, PDF viewer with bounding-box highlight overlay, live trace drawer.
- Day 7: Ragas eval suite with golden dataset, final README, Dockerfile and compose, deployment playbook, FDE demo scenarios.
