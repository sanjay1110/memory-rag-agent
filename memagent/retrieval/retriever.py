"""Retrieval pipeline: query -> hybrid candidates -> rerank -> attributed results.

    query ──► embed ──► FAISS top-k ─┐
          └─► BM25 top-k ────────────┼─► RRF fusion ─► rerank (LLM or cross-feature) ─► RetrievedItem[]
                                     │                                                   (score + citation)
    (repeated per collection: knowledge, memories)

Context filtering, de-duplication and compaction happen downstream in
``memagent.context.optimizer`` so they apply uniformly to *everything* that
competes for the prompt (retrieved chunks, memories, working-memory items).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..text import STOPWORDS, tokenize
from ..embeddings import SYNONYMS
from .index import Candidate, HybridIndex


@dataclass
class RetrievalResult:
    query: str
    items: list[Candidate]
    trace: dict = field(default_factory=dict)


class CrossFeatureReranker:
    """Offline reranker that scores (query, passage) *pairs* on features the
    first-stage retrievers cannot see jointly:

    * dense cosine similarity (semantic closeness)
    * BM25 score, normalised within the candidate pool (exact term match)
    * query-term coverage incl. synonyms/stems (does the passage address *all*
      parts of the question, not just one keyword?)
    * heading match (does the section title name the topic?)

    Output is a calibrated-ish relevance in [0, 1] used for filtering.
    """

    weights = {"dense": 0.35, "sparse": 0.20, "coverage": 0.35, "heading": 0.10}

    def __init__(self, idf=None):
        # idf(term) -> float; coverage is IDF-weighted so rare, specific terms
        # ("downtime", "password") count more than generic ones ("data", "move").
        self.idf = idf or (lambda t: 1.0)

    @staticmethod
    def _expand(term: str) -> set[str]:
        syns = SYNONYMS.get(term, ())
        return {term, term[:5], *syns, *(s[:5] for s in syns)}

    def score(self, query: str, cands: list[Candidate], use_sparse: bool = True) -> list[float]:
        """``use_sparse=False`` scores candidates that did not come from a BM25
        search (e.g. pinned upfront context); the sparse weight is then
        redistributed proportionally over the other features."""
        w = dict(self.weights)
        if not use_sparse:
            rest = 1 - w.pop("sparse")
            w = {k: v / rest for k, v in w.items()} | {"sparse": 0.0}
        qterms = [t for t in tokenize(query) if t not in STOPWORDS]
        max_sparse = max((c.sparse_score for c in cands), default=0.0) or 1.0
        out = []
        for c in cands:
            body = c.meta.get("embed_text", c.text)
            btoks = set(tokenize(body))
            bstems = {t[:5] for t in btoks}
            weights = [self.idf(t) for t in qterms]
            covered = sum(wt for t, wt in zip(qterms, weights) if self._expand(t) & (btoks | bstems))
            coverage = covered / sum(weights) if qterms and sum(weights) else 0.0
            heading = set(tokenize(f"{c.meta.get('section', '')} {c.meta.get('title', '')}"))
            head = 1.0 if any(self._expand(t) & heading for t in qterms) else 0.0
            dense = max(0.0, min(1.0, (c.dense_score - 0.05) / 0.45))
            sparse = c.sparse_score / max_sparse
            s = w["dense"] * dense + w["sparse"] * sparse + w["coverage"] * coverage + w["heading"] * head
            c.signals.update(dense=round(dense, 3), sparse=round(sparse, 3), coverage=round(coverage, 3), heading=head)
            out.append(s)
        return out


class Retriever:
    def __init__(self, indexes: dict[str, HybridIndex], embedder, cfg, llm=None):
        self.indexes = indexes
        self.embedder = embedder
        self.cfg = cfg
        self.llm = llm
        kb = indexes.get("knowledge")
        self.cross = CrossFeatureReranker(idf=(lambda t: kb.bm25.idf(t)) if kb is not None else None)

    def retrieve(
        self,
        query: str,
        collections: tuple[str, ...] = ("knowledge",),
        top_n: int | None = None,
        use_llm_rerank: bool | None = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        top_n = top_n or self.cfg.rerank_top_n
        qvec = self.embedder.embed([query])[0]
        cands: list[Candidate] = []
        per_coll = {}
        for name in collections:
            idx = self.indexes.get(name)
            if idx is None or len(idx) == 0:
                continue
            found = idx.search(query, qvec, self.cfg.dense_k, self.cfg.sparse_k, self.cfg.rrf_k)
            per_coll[name] = len(found)
            cands.extend(found)
        cands.sort(key=lambda c: -c.fused_score)
        cands = cands[: max(top_n * 2, top_n)]
        t1 = time.perf_counter()

        cross = self.cross.score(query, cands) if cands else []
        llm_scores = None
        use_llm = self.cfg.llm_rerank if use_llm_rerank is None else use_llm_rerank
        if use_llm and self.llm is not None and cands:
            llm_scores = self.llm.rerank(query, [c.text for c in cands[:top_n]])
        for i, c in enumerate(cands):
            if llm_scores is not None and i < len(llm_scores):
                c.rerank_score = 0.7 * llm_scores[i] + 0.3 * cross[i]
                c.signals["llm"] = round(llm_scores[i], 3)
            else:
                c.rerank_score = cross[i]
            # memory-specific prior (importance x recency) computed by LongTermMemory
            prior = c.meta.get("_prior")
            if prior is not None:
                c.rerank_score *= 0.75 + 0.25 * prior
        cands.sort(key=lambda c: -c.rerank_score)
        t2 = time.perf_counter()
        return RetrievalResult(
            query,
            cands[:top_n],
            {
                "candidates_per_collection": per_coll,
                "reranker": "llm+cross" if llm_scores is not None else "cross-feature",
                "first_stage_ms": round((t1 - t0) * 1000, 2),
                "rerank_ms": round((t2 - t1) * 1000, 2),
            },
        )
