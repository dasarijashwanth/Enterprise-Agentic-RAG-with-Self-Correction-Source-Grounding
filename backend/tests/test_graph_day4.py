"""Day 4 tests: LangGraph self-correction agent.

Everything runs offline: a fake dense searcher, a no-op reranker, and
the deterministic offline grader/rewriter/generator/checker. No live
Cohere or OpenAI call is ever made.
"""

import re

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import ask as ask_module
from app.graph import MAX_REWRITES, build_graph, run_question
from app.graph.components import (
    OfflineGrader,
    OfflineHallucinationChecker,
    OfflineRewriter,
    citation,
)
from app.graph.nodes import NO_CONTEXT_ANSWER, NO_INDEX_ANSWER
from app.main import app
from app.services import retriever as retriever_module
from app.services.retriever import ChunkDoc, HybridRetriever

CITATION_RE = re.compile(r"\[\d+, [\d.]+, [\d.]+, [\d.]+, [\d.]+\]")


def make_docs() -> list[ChunkDoc]:
    return [
        ChunkDoc(
            passage_id="0",
            text="Cross validation tests a model on several different slices of data",
            chunk_id="doc.pdf:p1:c0",
            source="doc.pdf",
            page=1,
            bbox=[72.0, 72.0, 500.0, 100.0],
        ),
        ChunkDoc(
            passage_id="1",
            text="Bounding boxes let a reader click a citation and see the exact passage",
            chunk_id="doc.pdf:p2:c0",
            source="doc.pdf",
            page=2,
            bbox=[72.0, 72.0, 500.0, 200.0],
        ),
        ChunkDoc(
            passage_id="2",
            text="Cross validation folds keep model scores honest in evaluation",
            chunk_id="doc.pdf:p1:c1",
            source="doc.pdf",
            page=1,
            bbox=[72.0, 120.0, 500.0, 160.0],
        ),
        ChunkDoc(
            passage_id="3",
            text="Gardening tips for summer tomatoes and rich soil",
            chunk_id="doc.pdf:p3:c0",
            source="doc.pdf",
            page=3,
            bbox=[72.0, 72.0, 400.0, 90.0],
        ),
    ]


class FakeHit:
    def __init__(self, chunk_id, score):
        self.metadata = {"chunk_id": chunk_id}
        self.score = score


class FakeDenseSearcher:
    """Fixed dense ranking used by most tests."""

    def search(self, query, top_k):
        order = [
            ("doc.pdf:p2:c0", 0.9),
            ("doc.pdf:p1:c0", 0.8),
            ("doc.pdf:p1:c1", 0.7),
            ("doc.pdf:p3:c0", 0.6),
        ]
        return [FakeHit(cid, s) for cid, s in order[:top_k]]


class NoOpReranker:
    def rerank(self, query, docs, top_n):
        return docs, [None] * len(docs), False


class SoilRewriter:
    """Test rewriter: steers the question toward the gardening doc's
    vocabulary ('soil'), which the first retrieval missed."""

    def rewrite(self, question):
        return "tomatoes soil gardening tips"


class FabricatingGenerator:
    """Adversarial generator: one grounded claim plus one invented
    claim with a real-looking citation."""

    def generate(self, question, chunks):
        grounded = {
            "text": "Bounding boxes let a reader click a citation and see the exact passage",
            "chunk_id": "doc.pdf:p2:c0",
            "page": 2,
            "bbox": [72.0, 72.0, 500.0, 200.0],
        }
        invented = {
            "text": "The CEO acquired a competitor in 2025",
            "chunk_id": "doc.pdf:p2:c0",
            "page": 2,
            "bbox": [72.0, 72.0, 500.0, 200.0],
        }
        return [grounded, invented]


class AllFabricatingGenerator:
    def generate(self, question, chunks):
        return [
            {
                "text": "The CEO acquired a competitor in 2025",
                "chunk_id": "doc.pdf:p2:c0",
                "page": 2,
                "bbox": [72.0, 72.0, 500.0, 200.0],
            }
        ]


def make_retriever(monkeypatch, searcher=None):
    monkeypatch.setattr(
        retriever_module, "load_chunk_docs", lambda path: make_docs()
    )
    return HybridRetriever(
        index_path="test-index",
        dense_searcher=searcher or FakeDenseSearcher(),
        reranker=NoOpReranker(),
    )


def test_happy_path_answer_has_bbox_citations(monkeypatch):
    graph = build_graph(make_retriever(monkeypatch))
    final = run_question(graph, "how do bounding boxes work")
    assert final["grounded"] is True
    assert CITATION_RE.search(final["answer"])
    # The boxes chunk is page 2 with bbox [72, 72, 500, 200].
    assert "[2, 72, 72, 500, 200]" in final["answer"]
    assert "rewrite_query" not in final["trace"]
    assert final["trace"] == [
        "retrieve",
        "grade_documents",
        "generate",
        "check_hallucination",
    ]


class StubRetriever:
    """Query-controlled retriever for loop tests: returns a fixed doc
    list per query substring (first match wins), so the test fully
    controls what each retrieval round sees."""

    def __init__(self, by_query):
        self.by_query = by_query
        self._docs = [d.result_dict() for d in make_docs()]

    def docs(self):
        return self._docs

    def search(self, query, top_k=5):
        for key, docs in self.by_query.items():
            if key in query:
                hits = [d.result_dict() for d in docs[:top_k]]
                return {"query": query, "results": hits, "reranked": False}
        return {"query": query, "results": [], "reranked": False}


def test_rewrite_loop_recovers_and_terminates(monkeypatch):
    docs = {d.chunk_id: d for d in make_docs()}
    retriever = StubRetriever(
        {
            # The rewrite adds the "soil" term, which surfaces the
            # gardening doc; the default round hides it.
            "soil": [docs["doc.pdf:p3:c0"], docs["doc.pdf:p2:c0"]],
            "": [docs["doc.pdf:p2:c0"], docs["doc.pdf:p1:c0"]],
        }
    )
    graph = build_graph(retriever, rewriter=SoilRewriter())
    final = run_question(graph, "how to grow tomatoes in a home garden")
    assert final["rewrite_count"] == 1
    assert final["grounded"] is True
    assert "[3, 72, 72, 400, 90]" in final["answer"]
    assert final["trace"].count("retrieve") == 2
    assert final["trace"].count("rewrite_query") == 1


def test_rewrite_loop_is_bounded(monkeypatch):
    graph = build_graph(make_retriever(monkeypatch))
    final = run_question(graph, "quantum banana telephone")
    assert final["rewrite_count"] == MAX_REWRITES
    assert final["grounded"] is False
    assert final["answer"] == NO_CONTEXT_ANSWER
    assert "will not guess" in final["answer"]


def test_hallucination_check_drops_unsupported_claim(monkeypatch):
    graph = build_graph(
        make_retriever(monkeypatch), generator=FabricatingGenerator()
    )
    final = run_question(graph, "how do bounding boxes work")
    assert "The CEO acquired a competitor in 2025" not in final["answer"]
    assert "Bounding boxes let a reader click a citation" in final["answer"]
    assert final["grounded"] is True


def test_hallucination_check_reports_gap_when_nothing_verifiable(monkeypatch):
    graph = build_graph(
        make_retriever(monkeypatch), generator=AllFabricatingGenerator()
    )
    final = run_question(graph, "how do bounding boxes work")
    assert final["grounded"] is False
    assert final["claims"] == []
    # The gap is reported honestly, naming the dropped claim.
    assert "could not verify" in final["answer"]
    assert "The CEO acquired a competitor in 2025" in final["gap_note"]


def test_empty_index_returns_honest_answer(monkeypatch):
    retriever = HybridRetriever(
        index_path="nonexistent-prefix-xyz",
        dense_searcher=FakeDenseSearcher(),
        reranker=NoOpReranker(),
    )
    graph = build_graph(retriever)
    final = run_question(graph, "how do bounding boxes work")
    assert final["index_empty"] is True
    assert final["answer"] == NO_INDEX_ANSWER
    assert final["grounded"] is False
    assert "rewrite_query" not in final["trace"]


def test_offline_grader_separates_relevant_from_irrelevant():
    grader = OfflineGrader()
    docs = {d.chunk_id: d.result_dict() for d in make_docs()}
    assert grader.score("how do bounding boxes work", docs["doc.pdf:p2:c0"]) >= 0.2
    assert grader.score("how do bounding boxes work", docs["doc.pdf:p3:c0"]) == 0.0


def test_offline_rewriter_extracts_content_terms():
    assert OfflineRewriter().rewrite("how do bounding boxes work") == (
        "bounding boxes work"
    )


def test_offline_checker_catches_invented_claim():
    checker = OfflineHallucinationChecker()
    docs = [d.result_dict() for d in make_docs()]
    bad = [
        {
            "text": "The CEO acquired a competitor in 2025",
            "chunk_id": "doc.pdf:p2:c0",
            "page": 2,
            "bbox": [72.0, 72.0, 500.0, 200.0],
        }
    ]
    assert checker.check(bad, docs) == bad


def test_citation_format_matches_spec():
    assert citation(2, [72.0, 72.0, 500.0, 200.0]) == "[2, 72, 72, 500, 200]"


def test_ask_endpoint_runs_agent(monkeypatch):
    monkeypatch.setattr(
        ask_module,
        "build_graph_app",
        lambda: build_graph(make_retriever(monkeypatch)),
    )
    client = TestClient(app)
    resp = client.post(
        "/api/v1/ask", json={"question": "how do bounding boxes work"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert "[2, 72, 72, 500, 200]" in body["answer"]
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["page"] == 2
    assert body["citations"][0]["bbox"] == [72.0, 72.0, 500.0, 200.0]
    assert body["trace"] == [
        "retrieve",
        "grade_documents",
        "generate",
        "check_hallucination",
    ]


def test_ask_endpoint_rejects_empty_question():
    client = TestClient(app)
    resp = client.post("/api/v1/ask", json={"question": "   "})
    assert resp.status_code == 400
