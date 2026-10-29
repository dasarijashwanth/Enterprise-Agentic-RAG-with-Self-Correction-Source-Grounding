"""Day 4: LangGraph state machine for the self-corrective RAG agent.

Graph (spec pipeline):

    retrieve -> grade_documents -+-> generate -> check_hallucination -> END
                                 |
                                 +-> rewrite_query -> retrieve   (loop,
                                     bounded by MAX_REWRITES)

The rewrite loop fires when no retrieved chunk grades relevant; after
MAX_REWRITES failed rewrites the agent answers honestly instead of
guessing. The hallucination check's single bounded regeneration lives
inside ``check_hallucination_node``. All loops terminate by
construction, with or without LLM components.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from .components import (
    Grader,
    HallucinationChecker,
    Generator,
    Rewriter,
)
from .nodes import (
    MAX_REWRITES,
    check_hallucination_node,
    generate_node,
    grade_documents_node,
    retrieve_node,
    rewrite_query_node,
)
from .state import GraphState


def _route_after_grade(state: GraphState) -> str:
    """Decide the next node after grading.

    Relevant chunks (or an empty index / exhausted rewrite budget) go
    to generation; otherwise rewrite the query and retrieve again.
    """
    if state.get("index_empty"):
        return "generate"
    if state.get("relevant"):
        return "generate"
    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "generate"  # honest "no relevant context" answer
    return "rewrite_query"


def build_graph(
    retriever,
    *,
    top_k: int = 5,
    grader: Grader | None = None,
    rewriter: Rewriter | None = None,
    generator: Generator | None = None,
    checker: HallucinationChecker | None = None,
):
    """Compile the agent graph. Components default to the deterministic
    offline implementations; pass LLM-backed ones when keys exist."""
    builder = StateGraph(GraphState)
    builder.add_node(
        "retrieve", partial(retrieve_node, retriever=retriever, top_k=top_k)
    )
    builder.add_node("grade_documents", partial(grade_documents_node, grader=grader))
    builder.add_node("rewrite_query", partial(rewrite_query_node, rewriter=rewriter))
    builder.add_node("generate", partial(generate_node, generator=generator))
    builder.add_node(
        "check_hallucination",
        partial(
            check_hallucination_node, checker=checker, generator=generator
        ),
    )
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        _route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate", "check_hallucination")
    builder.add_edge("check_hallucination", END)
    return builder.compile()


def run_question(
    graph, question: str, *, top_k: int = 5
) -> GraphState:
    """Run the agent on one question; returns the final graph state."""
    return graph.invoke(
        {
            "question": question,
            "query": question,
            "documents": [],
            "relevant": [],
            "claims": [],
            "answer": "",
            "grounded": False,
            "gap_note": "",
            "index_empty": False,
            "rewrite_count": 0,
            "regen_count": 0,
            "trace": [],
        }
    )
