"""Day 4: LangGraph agent state for the VeriRAG self-correction loop.

Spec pipeline: retrieve -> grade -> rewrite loop -> generate ->
hallucination check. Every claim the agent makes carries its source
chunk's [Page, X1, Y1, X2, Y2] grounding, so the Day 6 frontend can
highlight the exact PDF passage a citation points at.
"""

from __future__ import annotations

from typing import TypedDict


class Claim(TypedDict):
    """One grounded claim: answer text plus the chunk it came from."""

    text: str
    chunk_id: str
    page: int
    bbox: list[float]


class GraphState(TypedDict, total=False):
    """State threaded through the agent graph."""

    question: str  # original user question
    query: str  # current retrieval query (rewritten or original)
    documents: list[dict]  # retrieved chunks (SearchResult.to_dict())
    relevant: list[dict]  # chunks that passed the relevance grade
    claims: list[Claim]  # grounded claims the answer is built from
    answer: str  # final answer text with [Page, X1, Y1, X2, Y2] citations
    grounded: bool  # True when every claim checked out against its chunk
    gap_note: str  # honest note when context was missing or a claim dropped
    index_empty: bool  # True when no index exists at all
    rewrite_count: int  # bounded rewrite loop counter
    regen_count: int  # bounded regeneration counter
    trace: list[str]  # node names visited, for the Day 6 trace drawer
