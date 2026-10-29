"""Day 4: injectable agent components with deterministic offline defaults.

The spec calls for OpenAI GPT-4o-mini (grading/rewriting) and GPT-4o
(generation). API keys are not confirmed for this build, so every
component ships with an offline default that is fully deterministic
and testable with no keys and no network. The interfaces are narrow
enough that a real LLM can be dropped in later without touching the
graph: implement ``score`` / ``rewrite`` / ``generate`` / ``check``
and pass the instance to :func:`workflow.build_graph`.

Grading, rewriting, generation, and the hallucination check all run on
plain text here; nothing prints or stores API keys.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..services.retriever import tokenize
from .state import Claim

# Question words and filler carry no topic signal; dropping them keeps
# the offline grader and rewriter focused on content terms.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else when where which who whom what
    how why does do did is are was were be been being have has had
    with for from into over under about between through during
    i me my we our you your he she it they them his her its their
    this that these those as of on in to at by not no yes can could
    should would will just very more most such own same than too
    tell give show find get got make made say said like also any
    each per perhap maybe please
    """.split()
)


def content_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens minus stopwords and single chars."""
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1]


def citation(page: int, bbox: list[float]) -> str:
    """Format a grounding citation as [Page, X1, Y1, X2, Y2]."""
    coords = ", ".join(f"{v:g}" for v in bbox)
    return f"[{page:g}, {coords}]"


def first_sentence(text: str) -> str:
    """First sentence of a chunk, for extractive claim building."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text.strip()


class Grader(Protocol):
    """Scores one retrieved chunk's relevance to the question (0..1)."""

    def score(self, question: str, chunk: dict) -> float: ...


class OfflineGrader:
    """Deterministic grader: fraction of question content tokens that
    appear in the chunk. A chunk about the question's topic scores
    high; an off-topic chunk scores ~0. No network, no keys."""

    def score(self, question: str, chunk: dict) -> float:
        q = set(content_tokens(question))
        if not q:
            return 0.0
        c = set(content_tokens(chunk.get("text", "")))
        return len(q & c) / len(q)


class Rewriter(Protocol):
    """Produces a new retrieval query from the current one."""

    def rewrite(self, question: str) -> str: ...


class OfflineRewriter:
    """Deterministic rewriter: strips question words and stopwords down
    to the content terms (e.g. 'how do bounding boxes work' becomes
    'bounding boxes work'), which usually matches the index vocabulary
    better than the raw question."""

    def rewrite(self, question: str) -> str:
        terms = content_tokens(question)
        return " ".join(terms) if terms else question


class Generator(Protocol):
    """Builds grounded claims from the relevant chunks."""

    def generate(self, question: str, chunks: list[dict]) -> list[Claim]: ...


class OfflineGenerator:
    """Deterministic extractive generator: one claim per chunk, taken
    from the chunk's first sentence, so every claim is grounded in its
    cited chunk by construction."""

    def generate(self, question: str, chunks: list[dict]) -> list[Claim]:
        claims: list[Claim] = []
        for chunk in chunks:
            sentence = first_sentence(chunk.get("text", ""))
            if not sentence:
                continue
            claims.append(
                {
                    "text": sentence,
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "page": int(chunk.get("page", 0)),
                    "bbox": [float(v) for v in chunk.get("bbox", [])],
                }
            )
        return claims


class HallucinationChecker(Protocol):
    """Returns the claims that are NOT supported by their cited chunk."""

    def check(self, claims: list[Claim], chunks: list[dict]) -> list[Claim]: ...


class OfflineHallucinationChecker:
    """Deterministic checker: a claim is supported when at least
    SUPPORT_RATIO of its content tokens appear in its cited chunk's
    text. Extractive claims pass by construction; a claim invented
    from thin air (e.g. an adversarial injection) fails."""

    SUPPORT_RATIO = 0.3

    def check(self, claims: list[Claim], chunks: list[dict]) -> list[Claim]:
        by_id = {str(c.get("chunk_id", "")): c for c in chunks}
        unsupported: list[Claim] = []
        for claim in claims:
            chunk = by_id.get(claim["chunk_id"])
            claim_terms = set(content_tokens(claim["text"]))
            if not chunk or not claim_terms:
                unsupported.append(claim)
                continue
            chunk_terms = set(content_tokens(chunk.get("text", "")))
            if len(claim_terms & chunk_terms) / len(claim_terms) < self.SUPPORT_RATIO:
                unsupported.append(claim)
        return unsupported
