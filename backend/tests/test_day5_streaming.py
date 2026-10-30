"""Day 5 tests: SSE streaming, feedback, request logging, optional
Langfuse tracing, API-key auth, and rate limiting.

Everything runs offline: the fake dense searcher, no-op reranker, and
deterministic offline agent components. No live Cohere, OpenAI, or
Langfuse call is ever made. DATA_DIR is redirected to a tmp dir so no
test touches the real backend/data/ logs.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import ask as ask_module
from app.api.v1 import search as search_module
from app.core import config as config_module
from app.core import observability as obs_module
from app.core import security as security_module
from app.core.observability import LangfuseTracer
from app.graph import build_graph
from app.main import app
from app.services import retriever as retriever_module
from app.services.retriever import ChunkDoc, HybridRetriever


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
    ]


class FakeHit:
    def __init__(self, chunk_id, score):
        self.metadata = {"chunk_id": chunk_id}
        self.score = score


class FakeDenseSearcher:
    def search(self, query, top_k):
        return [
            FakeHit("doc.pdf:p2:c0", 0.9),
            FakeHit("doc.pdf:p1:c0", 0.8),
        ][:top_k]


class NoOpReranker:
    def rerank(self, query, docs, top_n):
        return docs, [None] * len(docs), False


class StubRetriever:
    """Drop-in for HybridRetriever.search with fixed docs."""

    def __init__(self, docs):
        self._docs = [d.result_dict() for d in docs]

    def docs(self):
        return self._docs

    def search(self, query, top_k=5):
        return {"query": query, "results": self._docs[:top_k], "reranked": False}


def make_graph(monkeypatch):
    monkeypatch.setattr(
        retriever_module, "load_chunk_docs", lambda path: make_docs()
    )
    retriever = HybridRetriever(
        index_path="test-index",
        dense_searcher=FakeDenseSearcher(),
        reranker=NoOpReranker(),
    )
    return build_graph(retriever)


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Redirect JSONL logs to tmp, reset the rate limiter."""
    monkeypatch.setattr(config_module.settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        config_module.settings, "VERIRAG_RATE_LIMIT_PER_MIN", 60
    )
    security_module.rate_limiter.reset()
    security_module.rate_limiter.per_minute = 60
    yield tmp_path


def _parse_sse(text: str):
    events = []
    for block in text.strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        events.append((name, json.loads(data) if data else None))
    return events


def test_stream_endpoint_streams_trace_answer_done(monkeypatch, isolated):
    monkeypatch.setattr(
        ask_module, "build_graph_app", lambda: make_graph(monkeypatch)
    )
    resp = TestClient(app).post(
        "/api/v1/ask/stream", json={"question": "how do bounding boxes work"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    kinds = [name for name, _ in events]
    assert kinds[0] == "trace"
    assert "answer" in kinds
    assert kinds[-1] == "done"
    # Trace events carry node names in order.
    trace_nodes = [d["node"] for n, d in events if n == "trace"]
    assert trace_nodes == [
        "retrieve",
        "grade_documents",
        "generate",
        "check_hallucination",
    ]
    # The answer event carries the same payload as /ask.
    answer = next(d for n, d in events if n == "answer")
    assert answer["grounded"] is True
    assert "[2, 72, 72, 500, 200]" in answer["answer"]
    assert len(answer["citations"]) >= 1


def test_stream_rejects_empty_question(isolated):
    resp = TestClient(app).post("/api/v1/ask/stream", json={"question": "   "})
    assert resp.status_code == 400


def test_stream_answer_matches_non_streaming(monkeypatch, isolated):
    monkeypatch.setattr(
        ask_module, "build_graph_app", lambda: make_graph(monkeypatch)
    )
    client = TestClient(app)
    plain = client.post(
        "/api/v1/ask", json={"question": "how do bounding boxes work"}
    ).json()
    streamed = _parse_sse(
        client.post(
            "/api/v1/ask/stream", json={"question": "how do bounding boxes work"}
        ).text
    )
    answer = next(d for n, d in streamed if n == "answer")
    assert answer["answer"] == plain["answer"]
    assert answer["grounded"] == plain["grounded"]
    assert answer["trace"] == plain["trace"]


def test_feedback_writes_jsonl_line(isolated, tmp_path):
    resp = TestClient(app).post(
        "/api/v1/feedback",
        json={
            "query": "how do bounding boxes work",
            "answer": "They ground citations.",
            "rating": "up",
            "comment": "clear",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "recorded"}
    lines = (tmp_path / "feedback.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["query"] == "how do bounding boxes work"
    assert rec["rating"] == "up"
    assert rec["comment"] == "clear"
    assert "ts" in rec


def test_feedback_rejects_bad_rating(isolated):
    resp = TestClient(app).post(
        "/api/v1/feedback",
        json={"query": "q", "answer": "a", "rating": "meh"},
    )
    assert resp.status_code == 422


def test_request_logger_appends_ask_record(monkeypatch, isolated, tmp_path):
    monkeypatch.setattr(
        ask_module, "build_graph_app", lambda: make_graph(monkeypatch)
    )
    resp = TestClient(app).post(
        "/api/v1/ask", json={"question": "how do bounding boxes work"}
    )
    assert resp.status_code == 200
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["endpoint"] == "/api/v1/ask"
    assert rec["grounded"] is True
    assert rec["rewrites"] == 0
    assert isinstance(rec["latency_ms"], (int, float))
    assert "ts" in rec


def test_request_logger_appends_search_record(monkeypatch, isolated, tmp_path):
    monkeypatch.setattr(
        search_module,
        "build_retriever",
        lambda: StubRetriever(make_docs()),
    )
    resp = TestClient(app).post("/api/v1/search", json={"query": "bounding boxes"})
    assert resp.status_code == 200
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["endpoint"] == "/api/v1/search"
    assert rec["result_count"] == 2
    assert "latency_ms" in rec


def test_langfuse_off_runs_clean_with_no_keys(monkeypatch, isolated):
    tracer = LangfuseTracer()
    assert tracer.enabled is False
    with tracer.span("ask", {"question": "q"}) as span:
        assert span is None  # no-op, never raises
    monkeypatch.setattr(
        ask_module, "build_graph_app", lambda: make_graph(monkeypatch)
    )
    resp = TestClient(app).post(
        "/api/v1/ask", json={"question": "how do bounding boxes work"}
    )
    assert resp.status_code == 200
    assert resp.json()["grounded"] is True


def test_auth_disabled_when_env_unset(monkeypatch, isolated):
    monkeypatch.setattr(config_module.settings, "VERIRAG_API_KEY", "")
    resp = TestClient(app).post("/api/v1/ask", json={"question": "q"})
    # No 401: auth disabled means the request proceeds (graph will
    # answer honestly against an empty index).
    assert resp.status_code != 401


def test_auth_enforced_when_key_set(monkeypatch, isolated):
    monkeypatch.setattr(config_module.settings, "VERIRAG_API_KEY", "s3cret")
    monkeypatch.setattr(
        ask_module, "build_graph_app", lambda: make_graph(monkeypatch)
    )
    client = TestClient(app)
    denied = client.post("/api/v1/ask", json={"question": "q"})
    assert denied.status_code == 401
    allowed = client.post(
        "/api/v1/ask",
        json={"question": "q"},
        headers={"X-API-Key": "s3cret"},
    )
    assert allowed.status_code == 200


def test_rate_limiter_returns_429_when_exceeded(monkeypatch, isolated):
    monkeypatch.setattr(config_module.settings, "VERIRAG_RATE_LIMIT_PER_MIN", 2)
    monkeypatch.setattr(
        ask_module, "build_graph_app", lambda: make_graph(monkeypatch)
    )
    client = TestClient(app)
    assert client.post("/api/v1/ask", json={"question": "q"}).status_code == 200
    assert client.post("/api/v1/ask", json={"question": "q"}).status_code == 200
    limited = client.post("/api/v1/ask", json={"question": "q"})
    assert limited.status_code == 429
    assert limited.headers.get("Retry-After") == "60"


def test_stream_is_also_rate_limited(monkeypatch, isolated):
    monkeypatch.setattr(config_module.settings, "VERIRAG_RATE_LIMIT_PER_MIN", 1)
    monkeypatch.setattr(
        ask_module, "build_graph_app", lambda: make_graph(monkeypatch)
    )
    client = TestClient(app)
    assert (
        client.post(
            "/api/v1/ask/stream", json={"question": "q"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/ask/stream", json={"question": "q"}
        ).status_code
        == 429
    )
