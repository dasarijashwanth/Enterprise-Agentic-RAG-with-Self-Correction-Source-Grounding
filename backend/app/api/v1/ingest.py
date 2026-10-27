"""Ingest endpoint: upload a PDF, parse it, chunk it, build the index.

Day 2 scope only: parsing, chunking, and LEANN index build. No
retrieval, reranking, or agent code lives here; that is Day 3/4 work.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ...core.config import settings
from ...services.chunker import chunk_pages
from ...services.index import build_leann_index
from ...services.parser import parse_pdf_bytes

router = APIRouter()


class IngestResponse(BaseModel):
    filename: str
    pages: int
    words: int
    chunks: int
    empty_pages: list[int]
    index_path: str
    backend: str
    embedding_model: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a .pdf file")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        parsed = parse_pdf_bytes(data, source=file.filename)
    except Exception as exc:  # invalid/corrupt PDF bytes
        raise HTTPException(status_code=400, detail=f"Could not parse PDF: {exc}") from exc

    chunks = chunk_pages(
        parsed.pages,
        source=file.filename,
        chunk_words=settings.CHUNK_WORDS,
        overlap_words=settings.CHUNK_OVERLAP_WORDS,
    )
    if not chunks:
        raise HTTPException(
            status_code=400, detail="No extractable text found in this PDF"
        )

    build_leann_index(
        chunks,
        index_path=settings.LEANN_INDEX_PATH,
        backend=settings.LEANN_BACKEND,
        embedding_model=settings.LEANN_EMBEDDING_MODEL,
    )
    return IngestResponse(
        filename=file.filename,
        pages=parsed.page_count,
        words=parsed.word_count,
        chunks=len(chunks),
        empty_pages=[p.page for p in parsed.pages if not p.words],
        index_path=settings.LEANN_INDEX_PATH,
        backend=settings.LEANN_BACKEND,
        embedding_model=settings.LEANN_EMBEDDING_MODEL,
    )
