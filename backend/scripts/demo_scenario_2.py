"""FDE demo scenario 2: rewrite-loop recovery.

Shows the self-correction loop doing its job: the first retrieval
round returns nothing relevant for a gardening question (the index
round hides the gardening doc), the rewriter steers the query toward
the gardening vocabulary, and the second retrieval round recovers a
grounded, cited answer. The loop stays bounded at MAX_REWRITES.

Run: cd backend && python -m scripts.demo_scenario_2
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import build_graph, run_question
from app.services.retriever import ChunkDoc
from eval.ragas_eval import make_corpus


class QueryControlledRetriever:
    """First query sees only non-gardening docs; the rewritten query
    (containing 'tomatoes') sees the gardening doc. Fully deterministic."""

    def __init__(self, docs: list[ChunkDoc]):
        self._docs = docs

    def docs(self):
        return [d.result_dict() for d in self._docs]

    def search(self, query, top_k=5):
        if "soil" in query:
            want = {"gardening.pdf:p1:c0", "rag-guide.pdf:p1:c0"}
        else:
            want = {"rag-guide.pdf:p1:c0", "ml-eval.pdf:p1:c0"}
        hits = [d.result_dict() for d in self._docs if d.chunk_id in want]
        return {"query": query, "results": hits[:top_k], "reranked": False}


class GardenRewriter:
    """Steers the question toward the gardening doc's vocabulary."""

    def rewrite(self, question: str) -> str:
        return "tomatoes soil gardening tips"


def main() -> int:
    graph = build_graph(
        QueryControlledRetriever(make_corpus()), rewriter=GardenRewriter()
    )
    question = "how to grow tomatoes in a home garden"
    final = run_question(graph, question)

    print(f"question: {question}")
    print(f"trace: {' -> '.join(final['trace'])}")
    print(f"rewrites: {final['rewrite_count']} (limit 2)")
    print(f"grounded: {final['grounded']}")
    print(f"answer: {final['answer']}")
    assert final["rewrite_count"] == 1
    assert final["grounded"] is True
    assert "[1, 72, 72, 400, 90]" in final["answer"]
    print("\nOK: missed on round 1, recovered via one bounded rewrite, "
          "grounded with a page-1 bbox citation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
