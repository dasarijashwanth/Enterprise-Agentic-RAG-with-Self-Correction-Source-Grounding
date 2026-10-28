"""Hybrid retrieval: LEANN dense + BM25 + RRF + Cohere Rerank v3 (Day 3).

Pipeline for one query:
  1. LEANN dense search over the Day 2 index (semantic matches).
  2. BM25 search over the same chunks (exact-term matches), implemented
     here in pure Python over the index's ``<path>.passages.jsonl`` file,
     so retrieval works with no extra dependency and no network.
  3. Reciprocal rank fusion (RRF) merges the two rankings.
  4. Cohere Rerank v3 reorders the fused top candidates. When no
     ``COHERE_API_KEY`` is set (or the API call fails), retrieval
     degrades gracefully: the fused order is returned unchanged and
     ``reranked`` is False. Tests use a stub reranker, never the API.

Every result carries the chunk's grounding metadata (chunk_id, source,
page, bbox) straight through, so Day 4's agent can cite
``[Page, X1, Y1, X2, Y2]`` for any hit. No Qdrant anywhere.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

RRF_K = 60
DEFAULT_DENSE_K = 20
DEFAULT_BM25_K = 20
DEFAULT_RERANK_K = 20
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
COHERE_RERANK_MODEL = "rerank-v3.0"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; shared by BM25 and query parsing."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ChunkDoc:
    """One indexed chunk, as stored in ``<index>.passages.jsonl``."""

    passage_id: str
    text: str
    chunk_id: str
    source: str
    page: int
    bbox: list[float]

    def result_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "bbox": self.bbox,
            "text": self.text,
        }


def load_chunk_docs(index_path: str) -> list[ChunkDoc]:
    """Read chunk docs from the LEANN passages file.

    Returns [] when the index does not exist yet (clean empty-index
    behavior) or when no readable lines are present. Malformed lines
    are skipped, never fatal.
    """
    passages = Path(str(index_path) + ".passages.jsonl")
    if not passages.exists():
        return []
    docs: list[ChunkDoc] = []
    for line in passages.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            meta = row.get("metadata") or {}
            docs.append(
                ChunkDoc(
                    passage_id=str(row.get("id", "")),
                    text=row.get("text", ""),
                    chunk_id=str(meta.get("chunk_id", "")),
                    source=str(meta.get("source", "")),
                    page=int(meta.get("page", 0)),
                    bbox=[float(v) for v in meta.get("bbox", [])],
                )
            )
        except (ValueError, TypeError, KeyError):
            continue
    return docs


class BM25Index:
    """Small deterministic BM25 (k1=1.5, b=0.75) over ChunkDocs."""

    def __init__(self, docs: list[ChunkDoc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        total_len = 0
        for doc in docs:
            tokens = tokenize(doc.text)
            total_len += len(tokens)
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            self._tf.append(counts)
            for tok in counts:
                self._df[tok] = self._df.get(tok, 0) + 1
        self._avgdl = total_len / len(docs) if docs else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[ChunkDoc, float]]:
        """Return (doc, bm25_score) sorted by score desc, then chunk_id."""
        terms = tokenize(query)
        if not terms or not self.docs or top_k <= 0:
            return []
        scored: list[tuple[ChunkDoc, float]] = []
        for doc, counts in zip(self.docs, self._tf):
            dl = sum(counts.values())
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += self._idf(term) * tf * (self.k1 + 1) / denom
            if score > 0:
                scored.append((doc, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return scored[:top_k]


def rrf_fuse(
    rankings: list[list[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over rankings of chunk_ids (rank 1-based).

    A doc ranked in both lists outscores a doc ranked first in only one
    list (1/(k+1) < 2/(k+2) for k=60), which is exactly the behavior
    the fusion step is supposed to give: agreement beats a single hit.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class CohereReranker:
    """Cohere Rerank v3 client with graceful offline degradation.

    ``rerank`` returns (docs_in_new_order, reranked_flag). With no API
    key, or on any API failure, it returns the input order unchanged
    with ``reranked=False`` so retrieval keeps working offline. The
    key is read from the environment only, never logged or stored.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = COHERE_RERANK_MODEL,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout

    @property
    def active(self) -> bool:
        return bool(self.api_key)

    def rerank(
        self, query: str, docs: list[ChunkDoc], top_n: int
    ) -> tuple[list[ChunkDoc], list[float | None], bool]:
        """Rerank docs; returns (docs, rerank_scores, reranked_flag)."""
        if not self.active or not docs:
            return docs, [None] * len(docs), False
        try:
            resp = httpx.post(
                COHERE_RERANK_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [d.text for d in docs],
                    "top_n": min(top_n, len(docs)),
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            reranked = [docs[r["index"]] for r in results]
            scores = [float(r["relevance_score"]) for r in results]
            # Keep any docs the API dropped (should not happen) at the end.
            seen = {docs[r["index"]].chunk_id for r in results}
            for doc in docs:
                if doc.chunk_id not in seen:
                    reranked.append(doc)
                    scores.append(None)
            return reranked, scores, True
        except Exception:
            return docs, [None] * len(docs), False


@dataclass
class SearchResult:
    chunk_id: str
    source: str
    page: int
    bbox: list[float]
    text: str
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "bbox": self.bbox,
            "text": self.text,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "dense_score": self.dense_score,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
        }


class HybridRetriever:
    """LEANN dense + BM25 + RRF + Cohere rerank over one index.

    ``dense_searcher`` and ``reranker`` are injectable so tests can run
    fully offline: pass a fake searcher with a ``search(query, top_k)``
    method returning objects (or dicts) with ``metadata``/``score``,
    and a fake reranker with a ``rerank(query, docs, top_n)`` method
    returning ``(docs, scores, reranked_flag)``.
    """

    def __init__(
        self,
        index_path: str,
        dense_searcher=None,
        reranker=None,
        dense_k: int = DEFAULT_DENSE_K,
        bm25_k: int = DEFAULT_BM25_K,
        rerank_k: int = DEFAULT_RERANK_K,
    ):
        self.index_path = index_path
        self._dense_searcher = dense_searcher
        self._reranker = reranker
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rerank_k = rerank_k
        self._docs_cache: list[ChunkDoc] | None = None

    def docs(self) -> list[ChunkDoc]:
        if self._docs_cache is None:
            self._docs_cache = load_chunk_docs(self.index_path)
        return self._docs_cache

    def _dense_ranking(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Dense hits as (chunk_id, score) in rank order; [] if no index."""
        if self._dense_searcher is None:
            try:
                from leann import LeannSearcher

                searcher = LeannSearcher(self.index_path)
                try:
                    hits = searcher.search(query, top_k=top_k)
                finally:
                    searcher.cleanup()
            except Exception:
                return []
        else:
            hits = self._dense_searcher.search(query, top_k=top_k)
        ranking: list[tuple[str, float]] = []
        for hit in hits:
            meta = hit.metadata if hasattr(hit, "metadata") else hit.get("metadata", {})
            chunk_id = (meta or {}).get("chunk_id")
            score = hit.score if hasattr(hit, "score") else hit.get("score")
            if chunk_id:
                ranking.append((str(chunk_id), float(score) if score is not None else 0.0))
        return ranking

    def search(self, query: str, top_k: int = 5) -> dict:
        """Run the full hybrid pipeline; returns results + reranked flag."""
        if not query or not query.strip() or top_k <= 0:
            return {"query": query, "results": [], "reranked": False}
        docs = self.docs()
        if not docs:
            return {"query": query, "results": [], "reranked": False}
        by_id = {d.chunk_id: d for d in docs}

        dense = self._dense_ranking(query, self.dense_k)
        dense_rank = {cid: i + 1 for i, (cid, _) in enumerate(dense)}
        dense_score = dict(dense)

        bm25 = BM25Index(docs).search(query, self.bm25_k)
        bm25_rank = {d.chunk_id: i + 1 for i, (d, _) in enumerate(bm25)}
        bm25_score = {d.chunk_id: s for d, s in bm25}

        fused = rrf_fuse(
            [[cid for cid, _ in dense], [d.chunk_id for d, _ in bm25]]
        )
        fused_score = dict(fused)
        candidates = [
            by_id[cid] for cid, _ in fused if cid in by_id
        ][: self.rerank_k]

        reranker = self._reranker
        if reranker is None:
            from ..core.config import settings

            reranker = CohereReranker(api_key=settings.COHERE_API_KEY)
        ordered, rerank_scores, reranked = reranker.rerank(
            query, candidates, top_n=top_k
        )
        results: list[SearchResult] = []
        for doc, rscore in zip(ordered[:top_k], rerank_scores[:top_k]):
            results.append(
                SearchResult(
                    chunk_id=doc.chunk_id,
                    source=doc.source,
                    page=doc.page,
                    bbox=doc.bbox,
                    text=doc.text,
                    dense_rank=dense_rank.get(doc.chunk_id),
                    bm25_rank=bm25_rank.get(doc.chunk_id),
                    dense_score=dense_score.get(doc.chunk_id),
                    bm25_score=bm25_score.get(doc.chunk_id),
                    rrf_score=fused_score.get(doc.chunk_id, 0.0),
                    rerank_score=rscore,
                )
            )
        return {
            "query": query,
            "results": [r.to_dict() for r in results],
            "reranked": reranked,
        }
