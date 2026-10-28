"""Day 3 live verification (not a unit test): ingest a sample PDF through
the real endpoint, then search it through the real /api/v1/search
endpoint (real LEANN dense search + BM25 + RRF, Cohere rerank inactive
without a key) and show that page/bbox metadata round-trips into
search results.

Usage (from backend/):
    TMPDIR=$HOME/.tmp-pip HF_HUB_DISABLE_XET=1 \
      ~/.venv-verirag/bin/python scripts/verify_search.py

Uses a throwaway index under data/verify_search_index so it never
clobbers the endpoint's configured index.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402

VERIFY_INDEX = "data/verify_search_index"


def make_pdf() -> bytes:
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_textbox(
        (72, 72, 523, 720),
        "Cross validation tests a model on several different slices of "
        "data instead of trusting one train test split. " * 6,
    )
    p2 = doc.new_page()
    p2.insert_textbox(
        (72, 72, 523, 720),
        "Bounding boxes let a reader click a citation and see the exact "
        "passage highlighted inside the PDF viewer. " * 6,
    )
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def main() -> None:
    settings.LEANN_INDEX_PATH = VERIFY_INDEX
    settings.COHERE_API_KEY = ""
    client = TestClient(app)
    resp = client.post(
        "/api/v1/ingest",
        files={"file": ("sample.pdf", make_pdf(), "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    print("ingest:", resp.json())

    for query, expect_page in (
        ("bounding boxes citation highlight", 2),
        ("cross validation slices", 1),
    ):
        started = time.time()
        resp = client.post(
            "/api/v1/search", json={"query": query, "top_k": 3}
        )
        elapsed = time.time() - started
        assert resp.status_code == 200, resp.text
        body = resp.json()
        print(f"\nquery: {query!r} -> {len(body['results'])} hits "
              f"in {elapsed:.1f}s, reranked={body['reranked']}")
        assert body["reranked"] is False, "no key set, rerank must be off"
        top = body["results"][0]
        print(
            f"  top: page={top['page']} bbox={top['bbox']} "
            f"rrf={top['rrf_score']:.4f} dense_rank={top['dense_rank']} "
            f"bm25_rank={top['bm25_rank']} text={top['text'][:80]!r}"
        )
        assert top["page"] == expect_page, (
            f"expected page {expect_page}, got {top['page']}"
        )
        assert len(top["bbox"]) == 4
        assert top["chunk_id"] and top["source"] == "sample.pdf"
    print("\nOK: hybrid search works end to end and metadata round-trips.")


if __name__ == "__main__":
    main()
