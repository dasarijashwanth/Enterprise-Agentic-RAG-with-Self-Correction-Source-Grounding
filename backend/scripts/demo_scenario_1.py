"""FDE demo scenario 1: contract Q&A with bbox citations.

End-to-end offline run over a REAL PDF: parses
backend/data/sample_contract.pdf with PyMuPDF (run
scripts/make_sample_pdf.py first if it does not exist), chunks it with
the page-safe chunker, and runs the LangGraph agent over the chunks.
The answer's [Page, X1, Y1, X2, Y2] citation points at the exact
termination clause on page 1.

Run: cd backend && python -m scripts.demo_scenario_1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import build_graph, run_question
from app.graph.components import content_tokens
from app.services import chunker as chunker_module
from app.services import parser as parser_module
from app.services.retriever import ChunkDoc
from eval.ragas_eval import EvalRetriever


def pdf_to_chunk_docs(pdf_path: Path, source: str) -> list[ChunkDoc]:
    parsed = parser_module.parse_pdf_bytes(pdf_path.read_bytes(), source)
    chunks = chunker_module.chunk_pages(parsed.pages, source)
    docs = []
    for i, ch in enumerate(chunks):
        docs.append(
            ChunkDoc(
                passage_id=str(i),
                text=ch.text,
                chunk_id=ch.id,
                source=ch.source,
                page=ch.page,
                bbox=ch.bbox,
            )
        )
    return docs


def main() -> int:
    pdf_path = Path(__file__).resolve().parent.parent / "data" / "sample_contract.pdf"
    if not pdf_path.exists():
        print("sample_contract.pdf missing; run: python -m scripts.make_sample_pdf")
        return 1
    docs = pdf_to_chunk_docs(pdf_path, "sample_contract.pdf")
    print(f"parsed {pdf_path.name}: {len(docs)} chunks")
    for d in docs:
        print(f"  {d.chunk_id} page={d.page} bbox={[round(v, 1) for v in d.bbox]}")

    graph = build_graph(EvalRetriever(docs))
    question = "what is the termination notice period"
    final = run_question(graph, question)

    print(f"\nquestion: {question}")
    print(f"grounded: {final['grounded']}  rewrites: {final['rewrite_count']}")
    print(f"trace: {' -> '.join(final['trace'])}")
    print(f"answer: {final['answer']}")
    print("\ncitations:")
    for c in final.get("claims", []):
        print(f"  [{c['page']}, {', '.join(f'{v:g}' for v in c['bbox'])}] {c['chunk_id']}")
    assert final["grounded"] is True
    assert "thirty (30) days" in final["answer"]
    print("\nOK: grounded answer cites the termination clause on page 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
