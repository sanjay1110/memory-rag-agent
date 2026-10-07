"""HybridIndex: one collection searchable two ways (dense FAISS + sparse BM25),
fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .vector_store import VectorStore


@dataclass
class Candidate:
    key: str
    text: str
    meta: dict
    collection: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float = 0.0
    signals: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        m = self.meta
        if self.collection == "knowledge":
            return f"{m.get('source')} § {m.get('section')}"
        return f"memory:{m.get('kind', 'fact')} (session {m.get('session_id', '?')}, {m.get('created_at', '')[:10]})"


class HybridIndex:
    def __init__(self, directory: Path, name: str, embedder):
        self.name = name
        self.embedder = embedder
        self.store = VectorStore(directory, name, embedder.dim)
        self.bm25 = BM25()
        for key, text, meta in self.store.all():
            self.bm25.add(key, meta.get("embed_text", text))

    def add(self, keys: list[str], texts: list[str], metas: list[dict], embed_texts: list[str] | None = None):
        embed_texts = embed_texts or texts
        vecs = self.embedder.embed(embed_texts)
        for m, et in zip(metas, embed_texts):
            m["embed_text"] = et
        existing = {k for k, _, _ in self.store.all()}
        self.store.upsert(keys, texts, vecs, metas)
        rebuild = any(k in existing for k in keys)
        if rebuild:
            self._rebuild_bm25()
        else:
            for k, et in zip(keys, embed_texts):
                self.bm25.add(k, et)
        return vecs

    def remove(self, key: str) -> None:
        if self.store.delete(key):
            self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self.bm25 = BM25()
        for key, text, meta in self.store.all():
            self.bm25.add(key, meta.get("embed_text", text))

    def search(self, query: str, qvec: np.ndarray, dense_k: int, sparse_k: int, rrf_k: int) -> list[Candidate]:
        pool: dict[str, Candidate] = {}
        for rank, (key, text, meta, score) in enumerate(self.store.search(qvec, dense_k)):
            c = pool.setdefault(key, Candidate(key, text, meta, self.name))
            c.dense_score, c.dense_rank = score, rank
        for rank, (key, score) in enumerate(self.bm25.search(query, sparse_k)):
            if key not in pool:
                got = self.store.get(key)
                if not got:
                    continue
                pool[key] = Candidate(key, got[1], got[2], self.name)
                v = self.store.get_vector(key)
                pool[key].dense_score = float(np.dot(v, qvec)) if v is not None else 0.0
            pool[key].sparse_score, pool[key].sparse_rank = score, rank
        for c in pool.values():
            c.fused_score = sum(1.0 / (rrf_k + r + 1) for r in (c.dense_rank, c.sparse_rank) if r is not None)
        return sorted(pool.values(), key=lambda c: -c.fused_score)

    def __len__(self) -> int:
        return len(self.store)
