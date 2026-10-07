"""Okapi BM25 sparse index (the lexical half of hybrid search).

Exact keyword matching rescues queries that dense vectors handle poorly:
identifiers (``NIMBUS_DB_POOL``), error codes (``E429``), version numbers.
"""

from __future__ import annotations

import math
from collections import Counter

from ..text import tokenize


def _stem(t: str) -> str:
    """Minimal plural folding so 'secrets' matches 'secret' (BM25 stays lexical)."""
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


class BM25:
    def __init__(self, k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.keys: list[str] = []
        self.docs: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()

    def add(self, key: str, text: str) -> None:
        toks = [_stem(t) for t in tokenize(text, drop_stopwords=True)]
        tf = Counter(toks)
        self.keys.append(key)
        self.docs.append(tf)
        self.lengths.append(len(toks))
        self.df.update(tf.keys())

    def idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(_stem(term), 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    @property
    def avgdl(self) -> float:
        return sum(self.lengths) / max(1, len(self.lengths))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = [_stem(t) for t in tokenize(query, drop_stopwords=True)]
        n = len(self.docs)
        if not n or not q:
            return []
        scores = []
        avgdl = self.avgdl
        for i, tf in enumerate(self.docs):
            s = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lengths[i] / avgdl))
            if s > 0:
                scores.append((self.keys[i], s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
