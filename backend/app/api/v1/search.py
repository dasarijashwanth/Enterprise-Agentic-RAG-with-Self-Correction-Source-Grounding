"""Search endpoint: hybrid retrieval over the Day 2 LEANN index (Day 3).

POST /api/v1/search runs LEANN dense + BM25 + RRF + Cohere rerank and
returns ranked chunks with page/bbox grounding. Day 4's LangGraph agent
calls this endpoint. A missing index returns an empty result list with
HTTP 200, never an error, so an un-ingested service stays clean.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...core.config import settings
from ...services.retriever import CohereReranker, HybridRetriever

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    chunk_id: str
    source: str
    page: int
    bbox: list[float]
    text: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    reranked: bool


def build_retriever() -> HybridRetriever:
    """Factory (monkeypatchable in tests) for the configured retriever."""
    return HybridRetriever(
        index_path=settings.LEANN_INDEX_PATH,
        reranker=CohereReranker(api_key=settings.COHERE_API_KEY),
    )


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    out = build_retriever().search(req.query, top_k=req.top_k)
    return SearchResponse(
        query=out["query"],
        results=[SearchHit(**hit) for hit in out["results"]],
        reranked=out["reranked"],
    )
