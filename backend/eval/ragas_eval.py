"""Day 7: Ragas-style evaluation harness for the VeriRAG agent.

What this measures (honest framing):
  - faithfulness: every claim in the answer must be supported by the
    chunk it cites. Measured with the deterministic offline checker,
    so it verifies the pipeline's grounding guarantee (extractive
    claims + hallucination check), NOT the quality of an LLM's
    reasoning. With the offline extractive generator this is 1.0 by
    construction; the value of the test is that it catches regressions
    (e.g. a component that starts fabricating claims).
  - citation precision / recall: do the emitted [Page, X1, Y1, X2, Y2]
    citations point at the chunks that genuinely answer the question?
  - context recall: did the relevant chunks survive retrieve + grade?
  - answer rate: fraction of answerable questions that come back
    grounded (not declined).
  - decline honesty: fraction of adversarial / out-of-corpus questions
    that decline honestly ("will not guess") instead of fabricating.
  - rewrite bound: the rewrite loop must terminate at <= MAX_REWRITES.

Everything is deterministic and offline: an overlap-based dense
searcher, a no-op reranker, and the offline grader/rewriter/generator/
checker. No network, no API keys. Run:

    cd backend && python -m eval.ragas_eval

Writes results to eval/results.json and prints a summary table.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import MAX_REWRITES, build_graph, run_question
from app.graph.components import (
    OfflineGrader,
    OfflineHallucinationChecker,
    content_tokens,
)
from app.services.retriever import ChunkDoc

CITATION_RE = re.compile(r"\[\d+, [\d.]+, [\d.]+, [\d.]+, [\d.]+\]")


def make_corpus() -> list[ChunkDoc]:
    """12 chunks, 3 topics. Every expected chunk_id is cited in QUESTIONS."""
    docs = [
        # ml-eval.pdf: model evaluation
        ChunkDoc(passage_id="0", source="ml-eval.pdf", page=1, chunk_id="ml-eval.pdf:p1:c0",
                 bbox=[72.0, 72.0, 500.0, 100.0],
                 text="Cross validation tests a model on several different slices of data"),
        ChunkDoc(passage_id="1", source="ml-eval.pdf", page=2, chunk_id="ml-eval.pdf:p2:c0",
                 bbox=[72.0, 72.0, 500.0, 120.0],
                 text="Overfitting shows up when training scores look great but validation folds disagree"),
        ChunkDoc(passage_id="2", source="ml-eval.pdf", page=3, chunk_id="ml-eval.pdf:p3:c0",
                 bbox=[72.0, 72.0, 500.0, 110.0],
                 text="Precision is the fraction of retrieved documents that are relevant; recall is the fraction of relevant documents that were retrieved"),
        ChunkDoc(passage_id="3", source="ml-eval.pdf", page=4, chunk_id="ml-eval.pdf:p4:c0",
                 bbox=[72.0, 72.0, 500.0, 100.0],
                 text="Ragas evaluates retrieval augmented generation with faithfulness and answer relevancy metrics"),
        # rag-guide.pdf: agent grounding
        ChunkDoc(passage_id="4", source="rag-guide.pdf", page=1, chunk_id="rag-guide.pdf:p1:c0",
                 bbox=[72.0, 72.0, 500.0, 200.0],
                 text="Bounding boxes let a reader click a citation and see the exact passage on the source page"),
        ChunkDoc(passage_id="5", source="rag-guide.pdf", page=2, chunk_id="rag-guide.pdf:p2:c0",
                 bbox=[72.0, 72.0, 500.0, 190.0],
                 text="Hybrid retrieval combines dense embeddings with BM25 keyword search through reciprocal rank fusion"),
        ChunkDoc(passage_id="6", source="rag-guide.pdf", page=3, chunk_id="rag-guide.pdf:p3:c0",
                 bbox=[72.0, 72.0, 500.0, 210.0],
                 text="The rewrite loop rephrases the query when no chunk grades relevant, then tries retrieval again, bounded at two attempts"),
        ChunkDoc(passage_id="7", source="rag-guide.pdf", page=4, chunk_id="rag-guide.pdf:p4:c0",
                 bbox=[72.0, 72.0, 500.0, 180.0],
                 text="Rate limiting protects the ask endpoints from abuse with a sliding window counter per client"),
        # gardening.pdf: distractor domain
        ChunkDoc(passage_id="8", source="gardening.pdf", page=1, chunk_id="gardening.pdf:p1:c0",
                 bbox=[72.0, 72.0, 400.0, 90.0],
                 text="Tomatoes need six hours of sun and consistent watering to grow well"),
        ChunkDoc(passage_id="9", source="gardening.pdf", page=2, chunk_id="gardening.pdf:p2:c0",
                 bbox=[72.0, 72.0, 400.0, 95.0],
                 text="Compost adds nitrogen and organic matter to tired soil"),
        # distractor chunks that share vocabulary with the eval topics
        ChunkDoc(passage_id="10", source="rag-guide.pdf", page=5, chunk_id="rag-guide.pdf:p5:c0",
                 bbox=[72.0, 72.0, 500.0, 170.0],
                 text="Streaming answers over server sent events keeps the chat interface responsive during long generations"),
        ChunkDoc(passage_id="11", source="ml-eval.pdf", page=5, chunk_id="ml-eval.pdf:p5:c0",
                 bbox=[72.0, 72.0, 500.0, 160.0],
                 text="Early stopping halts training when the validation loss stops improving across epochs"),
    ]
    return docs


# (question, expected chunk_ids, adversarial?)
QUESTIONS: list[tuple[str, list[str], bool]] = [
    ("what is cross validation", ["ml-eval.pdf:p1:c0"], False),
    ("how does overfitting show up in evaluation", ["ml-eval.pdf:p2:c0"], False),
    ("define precision and recall", ["ml-eval.pdf:p3:c0"], False),
    ("what is ragas", ["ml-eval.pdf:p4:c0"], False),
    ("how do bounding boxes work", ["rag-guide.pdf:p1:c0"], False),
    ("what is hybrid retrieval", ["rag-guide.pdf:p2:c0"], False),
    ("explain the rewrite loop in this agent", ["rag-guide.pdf:p3:c0"], False),
    ("how to grow tomatoes", ["gardening.pdf:p1:c0"], False),
    ("what is compost used for", ["gardening.pdf:p2:c0"], False),
    ("how does rate limiting work here", ["rag-guide.pdf:p4:c0"], False),
    ("quantum banana telephone", [], True),
    ("what is the capital of france", [], True),
    ("explain backpropagation from scratch", [], True),
    ("who directed the 1998 world cup final", [], True),
]


class OverlapDenseSearcher:
    """Deterministic stand-in for the LEANN dense index: ranks chunks
    by content-token overlap with the query."""

    def __init__(self, docs: list[ChunkDoc]):
        self._docs = docs

    def docs(self):
        return self._docs

    def search(self, query, top_k):
        q = set(content_tokens(query))

        class Hit:
            def __init__(self, chunk_id, score):
                self.metadata = {"chunk_id": chunk_id}
                self.score = score

        scored = []
        for d in self._docs:
            c = set(content_tokens(d.text))
            score = len(q & c) / len(q) if q else 0.0
            scored.append((d, score))
        scored.sort(key=lambda t: (-t[1], t[0].chunk_id))
        return [Hit(d.chunk_id, s) for d, s in scored[:top_k]]


class NoOpReranker:
    def rerank(self, query, docs, top_n):
        return docs, [None] * len(docs), False


class EvalRetriever:
    """Retriever facade matching HybridRetriever's search/docs contract."""

    def __init__(self, docs):
        self._docs = docs
        self._searcher = OverlapDenseSearcher(docs)
        self._reranker = NoOpReranker()

    def docs(self):
        return [d.result_dict() for d in self._docs]

    def search(self, query, top_k=5):
        hits = self._searcher.search(query, top_k * 2)
        by_id = {d.chunk_id: d for d in self._docs}
        docs = [by_id[h.metadata["chunk_id"]] for h in hits]
        docs, _, reranked = self._reranker.rerank(query, docs, top_k)
        return {
            "query": query,
            "results": [d.result_dict() for d in docs[:top_k]],
            "reranked": reranked,
        }


def evaluate_question(graph, question, expected, adversarial):
    t0 = time.perf_counter()
    final = run_question(graph, question)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    claims = final.get("claims", [])
    checker = OfflineHallucinationChecker()
    relevant = final.get("relevant", [])
    unsupported = checker.check(claims, relevant)

    cited = [c["chunk_id"] for c in claims]
    relevant_ids = [d.get("chunk_id") for d in relevant]
    expected_set = set(expected)

    faithfulness = (
        (len(claims) - len(unsupported)) / len(claims) if claims else 1.0
    )
    citation_precision = (
        len(set(cited) & expected_set) / len(set(cited)) if cited else 1.0
    )
    citation_recall = (
        len(set(cited) & expected_set) / len(expected_set) if expected_set else 1.0
    )
    context_recall = (
        len(set(relevant_ids) & expected_set) / len(expected_set)
        if expected_set
        else 1.0
    )
    grounded = bool(final.get("grounded"))
    honest_decline = (not grounded) and ("will not guess" in final.get("answer", ""))
    bbox_citations = len(CITATION_RE.findall(final.get("answer", "")))
    rewrites = int(final.get("rewrite_count", 0))

    return {
        "question": question,
        "adversarial": adversarial,
        "expected": expected,
        "grounded": grounded,
        "honest_decline": honest_decline,
        "n_claims": len(claims),
        "n_unsupported": len(unsupported),
        "faithfulness": round(faithfulness, 4),
        "citation_precision": round(citation_precision, 4),
        "citation_recall": round(citation_recall, 4),
        "context_recall": round(context_recall, 4),
        "cited": cited,
        "relevant": relevant_ids,
        "bbox_citations_in_answer": bbox_citations,
        "rewrites": rewrites,
        "rewrite_bound_ok": rewrites <= MAX_REWRITES,
        "latency_ms": round(latency_ms, 2),
    }


def main() -> int:
    docs = make_corpus()
    graph = build_graph(EvalRetriever(docs))

    per_q = [
        evaluate_question(graph, q, exp, adv) for q, exp, adv in QUESTIONS
    ]
    answerable = [r for r in per_q if not r["adversarial"]]
    adversarial = [r for r in per_q if r["adversarial"]]

    def mean(key, rows):
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else 0.0

    summary = {
        "n_questions": len(per_q),
        "n_answerable": len(answerable),
        "n_adversarial": len(adversarial),
        "faithfulness_mean": mean("faithfulness", per_q),
        "citation_precision_mean": mean("citation_precision", answerable),
        "citation_recall_mean": mean("citation_recall", answerable),
        "context_recall_mean": mean("context_recall", answerable),
        "answer_rate": round(
            sum(1 for r in answerable if r["grounded"]) / len(answerable), 4
        ),
        "decline_honesty_rate": round(
            sum(1 for r in adversarial if r["honest_decline"]) / len(adversarial),
            4,
        ),
        "rewrite_bound_ok_all": all(r["rewrite_bound_ok"] for r in per_q),
        "max_rewrites_observed": max(r["rewrites"] for r in per_q),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in per_q) / len(per_q), 2),
        "honesty_note": (
            "Deterministic offline harness: measures pipeline correctness "
            "(grounding guarantee, citation validity, rewrite bound), not "
            "LLM reasoning quality. No benchmark numbers are claimed; these "
            "are measured on this corpus with the offline components."
        ),
    }

    out = {"summary": summary, "questions": per_q}
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("Ragas-style eval (offline, deterministic) - VeriRAG Day 7")
    print(f"  questions: {summary['n_questions']} "
          f"({summary['n_answerable']} answerable, {summary['n_adversarial']} adversarial)")
    print(f"  faithfulness (mean):        {summary['faithfulness_mean']}")
    print(f"  citation precision (mean):  {summary['citation_precision_mean']}")
    print(f"  citation recall (mean):     {summary['citation_recall_mean']}")
    print(f"  context recall (mean):      {summary['context_recall_mean']}")
    print(f"  answer rate (answerable):   {summary['answer_rate']}")
    print(f"  decline honesty (advers.):  {summary['decline_honesty_rate']}")
    print(f"  rewrite bound ok (all):     {summary['rewrite_bound_ok_all']} "
          f"(max rewrites {summary['max_rewrites_observed']}, limit {MAX_REWRITES})")
    print(f"  avg latency:                {summary['avg_latency_ms']} ms")
    for r in per_q:
        mark = "OK " if (r["grounded"] or r["honest_decline"]) else "FAIL"
        print(f"  [{mark}] adv={int(r['adversarial'])} rewrites={r['rewrites']} "
              f"P={r['citation_precision']} R={r['citation_recall']} | {r['question'][:52]}")
    print(f"\nresults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
