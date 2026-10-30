"""Ask endpoint: run the Day 4 LangGraph agent (Day 4).

POST /api/v1/ask runs retrieve -> grade -> rewrite loop -> generate ->
hallucination check and returns the grounded answer with [Page, X1, Y1,
X2, Y2] citations. The agent runs fully offline with the deterministic
components; LLM-backed components can be injected later without
changing the graph. An empty index returns an honest answer with
HTTP 200, never an error.

Day 5: wrapped with optional Langfuse tracing, request logging, and
the shared auth + rate-limit dependency. The response shape is
unchanged from Day 4.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...core.config import settings
from ...core.observability import request_logger, tracer
from ...core.security import rate_limit
from ...graph.workflow import build_graph, run_question
from ...services.retriever import CohereReranker, HybridRetriever

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class AskCitation(BaseModel):
    chunk_id: str
    page: int
    bbox: list[float]
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[AskCitation]
    grounded: bool
    rewrites: int
    trace: list[str]


def build_graph_app():
    """Factory (monkeypatchable in tests) for the configured agent."""
    retriever = HybridRetriever(
        index_path=settings.LEANN_INDEX_PATH,
        reranker=CohereReranker(api_key=settings.COHERE_API_KEY),
    )
    return build_graph(retriever)


def _answer_payload(final: dict, question: str) -> AskResponse:
    """Assemble the Day 4 response shape from a final graph state."""
    claims = final.get("claims", [])
    by_id = {c["chunk_id"]: c for c in final.get("relevant", [])}
    citations = [
        AskCitation(
            chunk_id=c["chunk_id"],
            page=c["page"],
            bbox=c["bbox"],
            text=by_id.get(c["chunk_id"], {}).get("text", ""),
        )
        for c in claims
    ]
    return AskResponse(
        question=question,
        answer=final.get("answer", ""),
        citations=citations,
        grounded=bool(final.get("grounded", False)),
        rewrites=int(final.get("rewrite_count", 0)),
        trace=list(final.get("trace", [])),
    )


@router.post(
    "/ask", response_model=AskResponse, dependencies=[Depends(rate_limit)]
)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    with tracer.span("ask", {"question": req.question}):
        with request_logger.timed("/api/v1/ask") as extras:
            final = run_question(build_graph_app(), req.question, top_k=req.top_k)
            payload = _answer_payload(final, req.question)
            extras["grounded"] = payload.grounded
            extras["rewrites"] = payload.rewrites
    return payload
