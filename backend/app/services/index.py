"""LEANN dense index build (HNSW backend, Day 2 deviation from Qdrant).

Each chunk is added with its grounding metadata (chunk_id, source,
page, bbox), so a later search result carries the exact passage
location with it. Re-ingesting replaces the previous index at the
configured path; Day 2 treats the index as one document collection.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from leann import LeannBuilder

from .chunker import Chunk


def build_leann_index(
    chunks: list[Chunk],
    index_path: str,
    backend: str = "hnsw",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> dict:
    """Embed chunks and build a LEANN index at ``index_path``."""
    if not chunks:
        raise ValueError("no chunks to index")
    path = Path(index_path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)

    builder = LeannBuilder(backend_name=backend, embedding_model=embedding_model)
    for chunk in chunks:
        builder.add_text(chunk.text, metadata=chunk.metadata())
    builder.build_index(str(path))
    return {
        "index_path": str(path),
        "chunks": len(chunks),
        "backend": backend,
        "embedding_model": embedding_model,
    }
