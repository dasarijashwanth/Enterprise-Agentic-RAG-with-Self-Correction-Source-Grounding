"""PDF parsing with page and bounding-box extraction (PyMuPDF).

Every word extracted from a page keeps its bounding box, so downstream
chunking can attach exact ``[page, x0, y0, x1, y1]`` grounding to each
chunk. Pages with no extractable text (scanned images) yield an empty
word list instead of an error; the caller decides how to report them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf


@dataclass
class Word:
    text: str
    # [x0, y0, x1, y1] in PDF points, top-left origin (PyMuPDF coords).
    bbox: list[float]


@dataclass
class PageWords:
    page: int  # 1-based page number, matching what a reader sees.
    words: list[Word] = field(default_factory=list)


@dataclass
class ParsedDocument:
    source: str
    pages: list[PageWords] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def word_count(self) -> int:
        return sum(len(p.words) for p in self.pages)


def parse_pdf_bytes(data: bytes, source: str) -> ParsedDocument:
    """Parse PDF bytes into per-page words with bounding boxes."""
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        pages: list[PageWords] = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            words = [
                Word(text=w[4], bbox=[round(float(c), 2) for c in w[:4]])
                for w in page.get_text("words")
                if w[4].strip()
            ]
            pages.append(PageWords(page=page_index + 1, words=words))
        return ParsedDocument(source=source, pages=pages)
    finally:
        doc.close()
