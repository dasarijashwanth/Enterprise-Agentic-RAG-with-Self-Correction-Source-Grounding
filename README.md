# VeriRAG: Enterprise Self-Corrective Agentic RAG with Visual Source Grounding

Production-grade agentic RAG for high-stakes enterprise domains (financial
disclosures, legal contracts, compliance audits), built exactly per the
project spec: hybrid retrieval, LangGraph self-correction, visual source
grounding with PDF bounding-box citations, and Langfuse observability.

## Status: Day 1 (Mon Oct 26, 2026)

Backend skeleton only. The FastAPI app serves `GET /health` with no API
keys required. No retrieval, graph, chat, frontend, or eval code exists
yet; those land one slice per day below.

## Run Day 1 locally

```bash
cd backend
python -m venv ~/.venv-verirag
~/.venv-verirag/bin/python -m pip install -r requirements.txt
~/.venv-verirag/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs. Health check: http://localhost:8000/health.

## Roadmap (from the spec, one slice per day)

- Day 2: Document ingestion, PDF parser with bounding-box extraction, ingest endpoint, Qdrant indexing.
- Day 3: Hybrid retrieval, dense plus BM25 with reciprocal rank fusion, Cohere Rerank v3.
- Day 4: LangGraph agent, state, retrieval, relevance grading, query rewrite, generation with hallucination check, self-correction loop.
- Day 5: SSE streaming chat endpoint, feedback and logging, Langfuse tracing, auth and rate limits.
- Day 6: Next.js frontend, split-screen layout, chat box, PDF viewer with bounding-box highlight overlay, live trace drawer.
- Day 7: Ragas eval suite with golden dataset, final README, Dockerfile and compose, deployment playbook, FDE demo scenarios.
