"""Day 2 tests: PDF parsing with bbox, chunk grounding, ingest endpoint.

The endpoint test stubs out the LEANN build so unit tests stay fast and
offline; the real index build is exercised by scripts/verify_ingest.py
against a live LEANN index (see README).
"""

import io

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.api.v1 import ingest as ingest_module
from app.main import app
from app.services.chunker import chunk_pages
from app.services.parser import parse_pdf_bytes


def make_pdf() -> bytes:
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_textbox(
        (72, 72, 523, 720),
        "VeriRAG checks every claim against its source. "
        "Cross validation keeps model scores honest. " * 10,
    )
    page2 = doc.new_page()
    page2.insert_textbox(
        (72, 72, 523, 720),
        "Bounding boxes let a reader click a citation and see the "
        "exact passage in the PDF. " * 10,
    )
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_parse_keeps_page_and_bbox():
    parsed = parse_pdf_bytes(make_pdf(), source="sample.pdf")
    assert parsed.page_count == 2
    assert parsed.word_count > 50
    for page in parsed.pages:
        assert page.page in (1, 2)
        for word in page.words:
            x0, y0, x1, y1 = word.bbox
            assert 0 <= x0 < x1 and 0 <= y0 < y1
            assert word.text.strip()


def test_chunks_keep_page_bbox_metadata():
    parsed = parse_pdf_bytes(make_pdf(), source="sample.pdf")
    chunks = chunk_pages(parsed.pages, source="sample.pdf",
                         chunk_words=20, overlap_words=5)
    assert chunks
    ids = set()
    for chunk in chunks:
        assert chunk.page in (1, 2)
        x0, y0, x1, y1 = chunk.bbox
        assert 0 <= x0 < x1 and 0 <= y0 < y1
        assert chunk.text.strip()
        meta = chunk.metadata()
        assert meta["page"] == chunk.page
        assert meta["bbox"] == chunk.bbox
        assert meta["source"] == "sample.pdf"
        assert chunk.id not in ids
        ids.add(chunk.id)
    # Chunks never cross a page boundary: each page contributes chunks.
    assert {c.page for c in chunks} == {1, 2}


def test_chunk_pages_validates_args():
    parsed = parse_pdf_bytes(make_pdf(), source="sample.pdf")
    with pytest.raises(ValueError):
        chunk_pages(parsed.pages, "sample.pdf", chunk_words=0)
    with pytest.raises(ValueError):
        chunk_pages(parsed.pages, "sample.pdf", chunk_words=10,
                    overlap_words=10)


def test_ingest_endpoint_builds_chunks_and_calls_index(monkeypatch):
    calls = {}

    def fake_build(chunks, index_path, backend, embedding_model):
        calls["n"] = len(chunks)
        calls["index_path"] = index_path
        calls["metas"] = [c.metadata() for c in chunks]

    monkeypatch.setattr(ingest_module, "build_leann_index", fake_build)
    client = TestClient(app)
    resp = client.post(
        "/api/v1/ingest",
        files={"file": ("sample.pdf", make_pdf(), "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "sample.pdf"
    assert body["pages"] == 2
    assert body["chunks"] == calls["n"] > 0
    assert body["empty_pages"] == []
    assert all("page" in m and "bbox" in m for m in calls["metas"])


def test_ingest_rejects_non_pdf():
    client = TestClient(app)
    resp = client.post(
        "/api/v1/ingest",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_ingest_rejects_textless_pdf():
    doc = pymupdf.open()
    doc.new_page()
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    client = TestClient(app)
    resp = client.post(
        "/api/v1/ingest",
        files={"file": ("blank.pdf", buf.getvalue(), "application/pdf")},
    )
    assert resp.status_code == 400
