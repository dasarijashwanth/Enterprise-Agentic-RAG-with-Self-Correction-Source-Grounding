"""Chunking that preserves page/bbox grounding on every chunk.

Chunks are sliding word windows over a single page (never across a page
boundary, so a chunk's page is unambiguous). A chunk's bbox is the
union of its words' boxes, which is exactly the region a PDF viewer
highlights for a citation ``[Page, X1, Y1, X2, Y2]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .parser import PageWords


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int  # 1-based
    # [x0, y0, x1, y1] in PDF points.
    bbox: list[float]

    def metadata(self) -> dict:
        """Metadata stored alongside the chunk in the LEANN index."""
        return {
            "chunk_id": self.id,
            "source": self.source,
            "page": self.page,
            "bbox": self.bbox,
        }


def _union_bbox(boxes: list[list[float]]) -> list[float]:
    return [
        round(min(b[0] for b in boxes), 2),
        round(min(b[1] for b in boxes), 2),
        round(max(b[2] for b in boxes), 2),
        round(max(b[3] for b in boxes), 2),
    ]


def chunk_pages(
    pages: list[PageWords],
    source: str,
    chunk_words: int = 200,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Split parsed pages into overlapping word-window chunks."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must satisfy 0 <= overlap < chunk_words")

    chunks: list[Chunk] = []
    step = chunk_words - overlap_words
    for page in pages:
        words = page.words
        if not words:
            continue
        index = 0
        n = 0
        while index < len(words):
            window = words[index : index + chunk_words]
            chunks.append(
                Chunk(
                    id=f"{source}:p{page.page}:c{n}",
                    text=" ".join(w.text for w in window),
                    source=source,
                    page=page.page,
                    bbox=_union_bbox([w.bbox for w in window]),
                )
            )
            n += 1
            if index + chunk_words >= len(words):
                break
            index += step
    return chunks
