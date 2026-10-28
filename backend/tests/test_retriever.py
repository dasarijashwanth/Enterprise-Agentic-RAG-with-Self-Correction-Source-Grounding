"""Day 3 tests: hybrid retrieval (LEANN dense + BM25 + RRF + rerank).

All retrieval tests run offline with fake dense searchers and stub
rerankers; no live Cohere call is ever made. The real LEANN search
path is exercised by scripts/verify_search.py against a live index.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import search as search_module
from app.main import app
from app.services import retriever as retriever_module
from app.services.retriever import (
    BM25Index,
    ChunkDoc,
    CohereReranker,
    HybridRetriever,
    load_chunk_docs,
    rrf_fuse,
)


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
    """Fixed dense ranking: boxes doc first, then the two CV docs."""

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


class ReverseReranker:
    def rerank(self, query, docs, top_n):
        rev = list(reversed(docs))
        return rev, [0.9 - 0.1 * i for i in range(len(rev))], True


def retriever_with(monkeypatch, reranker):
    monkeypatch.setattr(
        retriever_module, "load_chunk_docs", lambda path: make_docs()
    )
    return HybridRetriever(
        index_path="test-index",
        dense_searcher=FakeDenseSearcher(),
        reranker=reranker,
    )


def test_bm25_ranks_exact_term_match_first():
    docs = make_docs()
    top = BM25Index(docs).search("bounding boxes citation", top_k=2)
    assert top[0][0].chunk_id == "doc.pdf:p2:c0"
    assert top[0][1] > 0


def test_bm25_ignores_unrelated_docs():
    docs = make_docs()
    top = BM25Index(docs).search("tomatoes soil", top_k=4)
    assert [d.chunk_id for d, _ in top] == ["doc.pdf:p3:c0"]


def test_rrf_agreement_beats_single_first():
    # "y" is 1st in one list and 2nd in the other; "x" is 1st in only
    # one list. RRF must rank the agreed doc first.
    fused = rrf_fuse([["y"], ["x", "y"]])
    assert fused[0][0] == "y"
    fused2 = rrf_fuse([["a"], ["b", "a"]])
    assert fused2[0][0] == "a"


def test_fusion_returns_results_with_metadata(monkeypatch):
    out = retriever_with(monkeypatch, NoOpReranker()).search(
        "bounding boxes", top_k=3
    )
    assert out["reranked"] is False
    results = out["results"]
    assert len(results) == 3
    assert results[0]["chunk_id"] == "doc.pdf:p2:c0"
    for hit in results:
        assert hit["source"] == "doc.pdf"
        assert hit["page"] in (1, 2, 3)
        assert len(hit["bbox"]) == 4
        assert hit["text"].strip()
        assert hit["rrf_score"] > 0
    # The boxes doc is dense rank 1 and BM25 rank 1.
    assert results[0]["dense_rank"] == 1
    assert results[0]["bm25_rank"] == 1


def test_rerank_stub_reorders_results(monkeypatch):
    base = retriever_with(monkeypatch, NoOpReranker()).search(
        "bounding boxes", top_k=4
    )
    last_fused = base["results"][-1]["chunk_id"]
    out = retriever_with(monkeypatch, ReverseReranker()).search(
        "bounding boxes", top_k=4
    )
    assert out["reranked"] is True
    assert out["results"][0]["chunk_id"] == last_fused
    assert out["results"][0]["rerank_score"] == pytest.approx(0.9)


def test_no_key_reranker_keeps_fused_order():
    docs = make_docs()
    kept, scores, flag = CohereReranker(api_key="").rerank("q", docs, 4)
    assert kept == docs
    assert scores == [None] * len(docs)
    assert flag is False


def test_reranker_degrades_on_api_failure(monkeypatch):
    import httpx

    def boom(*args, **kwargs):
        raise httpx.ConnectError("offline", request=None)

    monkeypatch.setattr(retriever_module.httpx, "post", boom)
    docs = make_docs()
    kept, scores, flag = CohereReranker(api_key="fake").rerank("q", docs, 4)
    assert kept == docs
    assert flag is False


def test_empty_index_returns_clean(monkeypatch):
    monkeypatch.setattr(retriever_module, "load_chunk_docs", lambda path: [])
    out = HybridRetriever(
        index_path="no-such-index",
        dense_searcher=FakeDenseSearcher(),
        reranker=NoOpReranker(),
    ).search("anything", top_k=5)
    assert out == {"query": "anything", "results": [], "reranked": False}


def test_load_chunk_docs_missing_index_is_empty(tmp_path):
    assert load_chunk_docs(str(tmp_path / "nope")) == []


def test_search_endpoint_shape_and_metadata(monkeypatch):
    monkeypatch.setattr(
        retriever_module, "load_chunk_docs", lambda path: make_docs()
    )
    monkeypatch.setattr(
        search_module,
        "build_retriever",
        lambda: HybridRetriever(
            index_path="test-index",
            dense_searcher=FakeDenseSearcher(),
            reranker=NoOpReranker(),
        ),
    )
    client = TestClient(app)
    resp = client.post("/api/v1/search", json={"query": "bounding boxes", "top_k": 2})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["query"] == "bounding boxes"
    assert body["reranked"] is False
    assert len(body["results"]) == 2
    hit = body["results"][0]
    assert hit["chunk_id"] == "doc.pdf:p2:c0"
    assert hit["page"] == 2
    assert hit["bbox"] == [72.0, 72.0, 500.0, 200.0]


def test_search_endpoint_rejects_empty_query(monkeypatch):
    monkeypatch.setattr(
        search_module,
        "build_retriever",
        lambda: HybridRetriever(
            index_path="test-index",
            dense_searcher=FakeDenseSearcher(),
            reranker=NoOpReranker(),
        ),
    )
    client = TestClient(app)
    resp = client.post("/api/v1/search", json={"query": "   ", "top_k": 5})
    assert resp.status_code == 400


def test_search_endpoint_empty_index_is_200(monkeypatch):
    monkeypatch.setattr(retriever_module, "load_chunk_docs", lambda path: [])
    monkeypatch.setattr(
        search_module,
        "build_retriever",
        lambda: HybridRetriever(
            index_path="no-such-index",
            dense_searcher=FakeDenseSearcher(),
            reranker=NoOpReranker(),
        ),
    )
    client = TestClient(app)
    resp = client.post("/api/v1/search", json={"query": "hello", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["results"] == []
