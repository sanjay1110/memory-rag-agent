"""Embedding models.

Two interchangeable embedders share one interface (``embed(list[str]) -> np.ndarray``,
rows L2-normalised so inner product == cosine similarity):

* ``HashingEmbedder`` (default) - a fit-free, fully offline embedder.  It hashes
  word unigrams, word bigrams and character 3-5-grams into a fixed-width vector,
  applies sub-linear TF weighting and a small domain synonym expansion, then
  L2-normalises.  Character n-grams give robustness to morphology and typos
  ("deploying" ~ "deployment"), bigrams give some phrase sensitivity.  Because it
  needs no fitting, new memories can be embedded at any time and the FAISS index
  never has to be rebuilt.
* ``SentenceTransformerEmbedder`` - dense neural embeddings
  (``all-MiniLM-L6-v2`` by default).  Select it with
  ``MEMAGENT_EMBEDDER=sentence-transformers`` after ``pip install sentence-transformers``.

The retriever never depends on which one is active.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from .text import STOPWORDS, tokenize

# Light query/document expansion so the offline embedder captures a little of
# the paraphrase tolerance a neural model would give.  Kept deliberately small
# and domain-generic.
SYNONYMS = {
    "deploy": ["release", "rollout", "ship"],
    "release": ["deploy", "rollout"],
    "rollback": ["revert", "undo"],
    "revert": ["rollback"],
    "db": ["database", "postgres"],
    "database": ["db", "postgres"],
    "postgres": ["database", "db"],
    "secret": ["credential", "password", "key"],
    "credential": ["secret", "password"],
    "password": ["secret", "credential"],
    "outage": ["incident", "downtime"],
    "incident": ["outage", "sev"],
    "cost": ["spend", "budget", "price"],
    "budget": ["cost", "spend"],
    "limit": ["quota", "throttle"],
    "quota": ["limit"],
    "latency": ["slow", "p99", "response"],
    "slow": ["latency"],
    "alert": ["page", "alarm"],
    "login": ["auth", "authentication", "sso"],
    "auth": ["authentication", "login", "token"],
    "migrate": ["migration", "move", "port"],
    "migration": ["migrate"],
    "backup": ["snapshot", "restore"],
    "restore": ["backup", "recovery"],
    "undo": ["rollback", "revert"],
    "keep": ["retain", "retention"],
    "kept": ["retained", "retention"],
    "max": ["maximum"],
    "maximum": ["max"],
}


class HashingEmbedder:
    name = "hashing-ngram-v1"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    @staticmethod
    @lru_cache(maxsize=200_000)
    def _bucket(feature: str, dim: int) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def _features(self, text: str) -> dict[str, float]:
        words = [w for w in tokenize(text) if w not in STOPWORDS]
        feats: dict[str, float] = {}

        def add(f: str, w: float) -> None:
            feats[f] = feats.get(f, 0.0) + w

        for w in words:
            add("w:" + w, 1.0)
            stem = w[:6]
            add("s:" + stem, 0.6)
            for syn in SYNONYMS.get(w, ()):
                add("w:" + syn, 0.35)
                add("s:" + syn[:6], 0.25)
            padded = f"<{w}>"
            for n in (3, 4, 5):
                for i in range(len(padded) - n + 1):
                    add(f"c{n}:" + padded[i : i + n], 0.25)
        for a, b in zip(words, words[1:]):
            add(f"b:{a}_{b}", 0.8)
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, tf in self._features(text).items():
                idx, sign = self._bucket(feat, self.dim)
                out[row, idx] += sign * (1.0 + np.log(tf)) if tf >= 1 else sign * tf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = f"st-{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(kind: str = "hashing", dim: int = 1024):
    if kind in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder()
    return HashingEmbedder(dim)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
