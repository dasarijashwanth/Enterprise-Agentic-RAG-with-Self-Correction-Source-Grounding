"""SSE streaming variant of POST /api/v1/ask (Day 5).

POST /api/v1/ask/stream returns text/event-stream. It runs the same
Day 4 agent as /api/v1/ask and streams its progress as events:

  event: trace    data: {"node": "<node name>"}   one per agent node
  event: answer   data: {AskResponse fields}      the final payload
  event: done     data: {}                        stream terminator

The existing non-streaming /api/v1/ask is unchanged. Same auth,
rate-limit, request-logging, and optional Langfuse tracing as /ask.
"""

from __future__ import annotations

import json
from typing import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...core.observability import request_logger, tracer
from ...core.security import rate_limit
from ...graph.workflow import run_question
from . import ask as ask_module

router = APIRouter()


class AskStreamRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _event_stream(question: str, top_k: int) -> Iterator[str]:
    """Run the agent once, then stream its node trace and final answer."""
    with tracer.span("ask_stream", {"question": question}):
        with request_logger.timed("/api/v1/ask/stream") as extras:
            final = run_question(
                ask_module.build_graph_app(), question, top_k=top_k
            )
            payload = ask_module._answer_payload(final, question)
            extras["grounded"] = payload.grounded
            extras["rewrites"] = payload.rewrites
            for node in payload.trace:
                yield _sse("trace", json.dumps({"node": node}))
            yield _sse("answer", payload.model_dump_json())
            yield _sse("done", "{}")


@router.post("/ask/stream", dependencies=[Depends(rate_limit)])
def ask_stream(req: AskStreamRequest) -> StreamingResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    return StreamingResponse(
        _event_stream(req.question, req.top_k),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
