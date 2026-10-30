"""Day 4 verification: run the LangGraph agent end to end, offline.

Builds a tiny throwaway LEANN-free index (passages.jsonl only; the
dense searcher is faked), runs one question through retrieve -> grade
-> generate -> check_hallucination, and prints the answer with its
[Page, X1, Y1, X2, Y2] citations plus the node trace. No API keys,
no network.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import build_graph, run_question
from app.services.retriever import ChunkDoc, HybridRetriever


class FakeHit:
    def __init__(self, chunk_id, score):
        self.metadata = {"chunk_id": chunk_id}
        self.score = score


class FakeDenseSearcher:
    def search(self, query, top_k):
        return [
            FakeHit("report.pdf:p1:c0", 0.92),
            FakeHit("report.pdf:p2:c0", 0.81),
        ][:top_k]


class NoOpReranker:
    def rerank(self, query, docs, top_n):
        return docs, [None] * len(docs), False


def write_passages(prefix: Path) -> None:
    rows = [
        {
            "id": "0",
            "text": "Cross validation tests a model on several different slices of data",
            "metadata": {
                "chunk_id": "report.pdf:p1:c0",
                "source": "report.pdf",
                "page": 1,
                "bbox": [72.0, 72.0, 500.0, 100.0],
            },
        },
        {
            "id": "1",
            "text": "Bounding boxes let a reader click a citation and see the exact passage",
            "metadata": {
                "chunk_id": "report.pdf:p2:c0",
                "source": "report.pdf",
                "page": 2,
                "bbox": [72.0, 72.0, 500.0, 200.0],
            },
        },
    ]
    path = Path(str(prefix) + ".passages.jsonl")
    path.write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "leann_index")
        write_passages(Path(prefix))
        retriever = HybridRetriever(
            index_path=prefix,
            dense_searcher=FakeDenseSearcher(),
            reranker=NoOpReranker(),
        )
        graph = build_graph(retriever)
        question = sys.argv[1] if len(sys.argv) > 1 else (
            "how do bounding boxes work"
        )
        final = run_question(graph, question)
        print(f"question : {final['question']}")
        print(f"grounded : {final['grounded']}")
        print(f"rewrites : {final['rewrite_count']}")
        print(f"trace    : {' -> '.join(final['trace'])}")
        print("answer   :")
        print(final["answer"] or "(no answer)")
        if final.get("gap_note"):
            print(f"gap note : {final['gap_note']}")
        ok = final["grounded"] and "[2, 72, 72, 500, 200]" in final["answer"]
        print("VERIFY:", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
