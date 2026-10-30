"""Feedback endpoint (Day 5).

POST /api/v1/feedback stores a thumbs up/down rating (with an
optional comment) for one answer to the local JSONL log at
DATA_DIR/feedback.jsonl (gitignored). Feeds the Day 7 eval loop.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...core.observability import feedback_logger
from ...core.security import rate_limit

router = APIRouter()


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: Literal["up", "down"]
    comment: str | None = None


class FeedbackResponse(BaseModel):
    status: str


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    dependencies=[Depends(rate_limit)],
)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    feedback_logger.log(
        {
            "query": req.query,
            "answer": req.answer,
            "rating": req.rating,
            "comment": req.comment,
        }
    )
    return FeedbackResponse(status="recorded")
