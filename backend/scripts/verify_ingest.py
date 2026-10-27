"""Day 2 live verification (not a unit test): ingest a sample PDF through
the real endpoint, then search the built LEANN index and show that
page/bbox metadata round-trips into search results.

Usage (from backend/):
    ~/.venv-verirag/bin/python scripts/verify_ingest.py

The script uses a throwaway index under data/verify_index so it never
clobbers the endpoint's configured index.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient
from leann import LeannSearcher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402

VERIFY_INDEX = "data/verify_index"


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
    client = TestClient(app)
    resp = client.post(
        "/api/v1/ingest",
        files={"file": ("sample.pdf", make_pdf(), "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    print("ingest response:", body)
    assert body["pages"] == 2 and body["chunks"] > 0
    # LEANN writes the index as files sharing the path prefix
    # (<path>.meta.json, <path>.passages.jsonl, ...), not a directory.
    assert Path(VERIFY_INDEX + ".meta.json").exists(), (
        "index files were not created under " + VERIFY_INDEX
    )

    searcher = LeannSearcher(VERIFY_INDEX)
    try:
        for query in ("bounding boxes citation highlight", "cross validation slices"):
            hits = searcher.search(query, top_k=3)
            print(f"\nquery: {query!r} -> {len(hits)} hits")
            for hit in hits:
                meta = hit.metadata or {}
                print(
                    f"  page={meta.get('page')} bbox={meta.get('bbox')} "
                    f"score={getattr(hit, 'score', None)} "
                    f"text={hit.text[:80]!r}"
                )
                assert meta.get("page") in (1, 2), "page metadata missing"
                bbox = meta.get("bbox")
                assert isinstance(bbox, (list, tuple)) and len(bbox) == 4, (
                    "bbox metadata missing or malformed: %r" % (bbox,)
                )
    finally:
        searcher.cleanup()
    print("\nOK: ingest endpoint works and page/bbox metadata round-trips.")


if __name__ == "__main__":
    main()
