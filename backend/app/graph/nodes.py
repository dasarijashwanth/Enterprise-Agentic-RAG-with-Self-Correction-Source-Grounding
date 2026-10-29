"""Day 4: LangGraph node functions for the self-correction loop.

Node names (matching the spec's pipeline diagram):
  retrieve -> grade_documents -> rewrite_query -> generate ->
  check_hallucination

Dependencies (retriever, grader, rewriter, generator, checker) are
passed in by ``workflow.build_graph`` via ``functools.partial``, so
tests inject fakes and production injects the configured components.
Every node appends its name to ``state["trace"]`` for the Day 6
trace drawer.
"""

from __future__ import annotations

from .components import (
    Grader,
    HallucinationChecker,
    OfflineGrader,
    OfflineHallucinationChecker,
    OfflineRewriter,
    Rewriter,
    citation,
    Generator,
    OfflineGenerator,
)
from .state import Claim, GraphState

# Bounds keep both self-correction loops finite: the rewrite loop runs
# at most MAX_REWRITES extra retrievals, and a failed hallucination
# check triggers at most MAX_REGENERATIONS answer rebuilds before the
# agent reports the gap honestly instead of guessing.
MAX_REWRITES = 2
MAX_REGENERATIONS = 1
RELEVANCE_THRESHOLD = 0.2

NO_INDEX_ANSWER = (
    "I could not answer this: no documents have been indexed yet. "
    "Ingest a PDF first, then ask again."
)
NO_CONTEXT_ANSWER = (
    "I could not find relevant context in the indexed documents to "
    "answer this question, so I will not guess."
)


def _visited(state: GraphState, node: str) -> list[str]:
    return [*state.get("trace", []), node]


def render_answer(claims: list[Claim]) -> str:
    """Render claims as sentences, each followed by its [Page, ...] cite."""
    lines = []
    for claim in claims:
        text = claim["text"].rstrip()
        if text and text[-1] not in ".!?":
            text += "."
        lines.append(f"{text} {citation(claim['page'], claim['bbox'])}")
    return "\n\n".join(lines)


def retrieve_node(state: GraphState, *, retriever, top_k: int) -> dict:
    """Run hybrid retrieval for the current query (Day 3 retriever)."""
    query = state.get("query") or state["question"]
    out = retriever.search(query, top_k=top_k)
    docs = out["results"] if isinstance(out, dict) else []
    return {
        "query": query,
        "documents": docs,
        "index_empty": not docs and not retriever.docs(),
        "trace": _visited(state, "retrieve"),
    }


def grade_documents_node(
    state: GraphState, *, grader: Grader | None = None
) -> dict:
    """Score each retrieved chunk; keep the ones relevant to the
    question. An empty index skips grading entirely."""
    grader = grader or OfflineGrader()
    if state.get("index_empty"):
        return {"relevant": [], "trace": _visited(state, "grade_documents")}
    question = state["question"]
    relevant = [
        doc
        for doc in state.get("documents", [])
        if grader.score(question, doc) >= RELEVANCE_THRESHOLD
    ]
    return {"relevant": relevant, "trace": _visited(state, "grade_documents")}


def rewrite_query_node(
    state: GraphState, *, rewriter: Rewriter | None = None
) -> dict:
    """Bounded query rewrite: new query from the current one, counter
    incremented so the workflow can stop the loop."""
    rewriter = rewriter or OfflineRewriter()
    new_query = rewriter.rewrite(state.get("query") or state["question"])
    if not new_query.strip() or new_query == state.get("query"):
        # A rewrite that changes nothing still counts: the loop must
        # terminate even with a degenerate rewriter.
        new_query = f"{state.get('query') or state['question']} details"
    return {
        "query": new_query,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "trace": _visited(state, "rewrite_query"),
    }


def _honest_answer(state: GraphState, text: str) -> dict:
    return {
        "answer": text,
        "claims": [],
        "grounded": False,
        "gap_note": text,
        "trace": _visited(state, "generate"),
    }


def generate_node(
    state: GraphState, *, generator: Generator | None = None
) -> dict:
    """Build a cited answer from the relevant chunks. When no chunk
    passed grading (or nothing was indexed), say so honestly instead
    of generating from thin air."""
    if state.get("index_empty"):
        return _honest_answer(state, NO_INDEX_ANSWER)
    relevant = state.get("relevant", [])
    if not relevant:
        return _honest_answer(state, NO_CONTEXT_ANSWER)
    generator = generator or OfflineGenerator()
    claims = generator.generate(state["question"], relevant)
    if not claims:
        return _honest_answer(state, NO_CONTEXT_ANSWER)
    return {
        "claims": claims,
        "answer": render_answer(claims),
        "grounded": True,
        "gap_note": "",
        "trace": _visited(state, "generate"),
    }


def check_hallucination_node(
    state: GraphState,
    *,
    checker: HallucinationChecker | None = None,
    generator: Generator | None = None,
) -> dict:
    """Verify every claim against its cited chunk. One bounded
    regeneration: drop unsupported claims and rebuild the answer from
    what remains. If nothing verifiable remains, report the gap
    honestly instead of shipping an ungrounded answer."""
    checker = checker or OfflineHallucinationChecker()
    chunks = state.get("relevant", []) or state.get("documents", [])
    claims = list(state.get("claims", []))
    trace = _visited(state, "check_hallucination")

    if not claims:
        # Honest no-answer path (empty index / no relevant context):
        # nothing to verify, keep the incoming grounded flag as-is.
        return {
            "grounded": bool(state.get("grounded", False)),
            "gap_note": state.get("gap_note", ""),
            "trace": trace,
        }

    bad = checker.check(claims, chunks)

    if not bad:
        return {"grounded": True, "gap_note": "", "trace": trace}

    regen_count = state.get("regen_count", 0)
    if regen_count >= MAX_REGENERATIONS:
        return _gap_report(state, claims, bad, trace)

    kept = [c for c in claims if c not in bad]
    if not kept:
        return _gap_report(state, claims, bad, trace)

    # One bounded regeneration from the surviving grounded claims.
    still_bad = checker.check(kept, chunks)
    if still_bad:
        return _gap_report(state, claims, still_bad, trace)
    return {
        "claims": kept,
        "answer": render_answer(kept),
        "grounded": True,
        "gap_note": "",
        "regen_count": regen_count + 1,
        "trace": trace,
    }


def _gap_report(
    state: GraphState, claims: list[Claim], bad: list[Claim], trace: list[str]
) -> dict:
    """Honest gap report: name what could not be verified, keep any
    verified claims, never ship the unsupported ones."""
    kept = [c for c in claims if c not in bad]
    dropped = "; ".join(f'"{c["text"]}"' for c in bad)
    note = (
        "I could not verify the following against the retrieved "
        f"documents, so I left them out: {dropped}."
    )
    answer = render_answer(kept) if kept else ""
    if kept:
        answer = f"{answer}\n\n{note}"
    else:
        answer = note
    return {
        "claims": kept,
        "answer": answer,
        "grounded": False,
        "gap_note": note,
        "regen_count": state.get("regen_count", 0) + 1,
        "trace": trace,
    }
