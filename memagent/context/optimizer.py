"""Context optimisation: filtering, de-duplication, compaction.

Everything that competes for the prompt passes through here, in this order:

1. **Filter**  - drop candidates below the relevance floor
   ``max(min_relevance, relative_floor x best_score)`` or with zero query-term
   coverage (a semantic false positive).
2. **Dedup**   - (a) containment: drop an item whose content words are >= 85%
   contained in a longer candidate (the longer one inherits the higher score);
   (b) exact duplicates (normalised content hash); (c) near-duplicates
   (cosine >= ``near_dup_threshold`` *or* token Jaccard >= 0.7), including
   items that duplicate something already loaded upfront.
3. **Select**  - keep the best ``final_k``.
4. **Compact** - if the survivors still exceed the token budget, each is
   reduced to its most query-relevant sentences (extractive compaction keeps
   the original wording so citations stay faithful), and if that is still not
   enough the lowest-ranked items are dropped.

``OptimiserReport`` records how many tokens each step removed; the evaluation
harness aggregates these into the context-efficiency metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..text import content_hash, count_tokens, jaccard, split_sentences, tokenize


def containment(small: str, big: str) -> float:
    """Share of ``small``'s content words that also occur in ``big``."""
    a, b = set(tokenize(small, True)), set(tokenize(big, True))
    return len(a & b) / len(a) if a else 0.0


@dataclass
class OptimiserReport:
    input_items: int = 0
    input_tokens: int = 0
    filtered: int = 0
    exact_dups: int = 0
    near_dups: int = 0
    dropped_by_k: int = 0
    compacted_items: int = 0
    dropped_by_budget: int = 0
    output_items: int = 0
    output_tokens: int = 0
    removed: list = field(default_factory=list)

    @property
    def reduction(self) -> float:
        return 0.0 if not self.input_tokens else 1 - self.output_tokens / self.input_tokens


class ContextOptimizer:
    def __init__(self, embedder, min_relevance=0.2, near_dup_threshold=0.85, relative_floor=0.5,
                 containment_threshold=0.85):
        self.embedder = embedder
        self.min_relevance = min_relevance          # absolute floor
        self.relative_floor = relative_floor        # keep only items >= this fraction of the best score
        self.near_dup_threshold = near_dup_threshold
        self.containment_threshold = containment_threshold

    # -------------------------------------------------------------- stages
    def filter(self, items, rep: OptimiserReport):
        kept = []
        best = max((c.rerank_score for c in items), default=0.0)
        floor = max(self.min_relevance, self.relative_floor * best)
        for c in items:
            cov = c.signals.get("coverage", 1.0)
            if c.rerank_score < floor or cov == 0:
                rep.filtered += 1
                rep.removed.append(("filtered", c.key, round(c.rerank_score, 3)))
            else:
                kept.append(c)
        return kept

    def dedup(self, items, rep: OptimiserReport, already_loaded: list[str] | None = None):
        already_loaded = already_loaded or []
        seen_hash = {content_hash(t) for t in already_loaded}
        kept, kept_vecs = [], []
        # Containment pass: an item whose content is (almost) entirely inside a
        # longer candidate is redundant even if their vectors differ (e.g. a FAQ
        # answer copied from a guide).  Keep the longer, more complete one.
        by_len = sorted(items, key=lambda c: -len(c.text))
        contained = set()
        for i, small in enumerate(by_len):
            for big in by_len[:i]:
                if big.key not in contained and containment(small.text, big.text) >= self.containment_threshold:
                    contained.add(small.key)
                    big.rerank_score = max(big.rerank_score, small.rerank_score)
                    rep.near_dups += 1
                    rep.removed.append(("contained_in", small.key, big.key))
                    break
        items = [c for c in items if c.key not in contained]
        loaded_vecs = list(self.embedder.embed(already_loaded)) if already_loaded else []
        texts = [c.text for c in items]
        vecs = self.embedder.embed(texts) if texts else np.zeros((0, 1))
        for c, v in sorted(zip(items, vecs), key=lambda x: -x[0].rerank_score):
            h = content_hash(c.text)
            if h in seen_hash:
                rep.exact_dups += 1
                rep.removed.append(("exact_dup", c.key))
                continue
            dup = False
            for other_v, other_t in zip(kept_vecs + loaded_vecs, [k.text for k in kept] + already_loaded):
                if float(np.dot(v, other_v)) >= self.near_dup_threshold or jaccard(c.text, other_t) >= 0.7:
                    dup = True
                    break
            if dup:
                rep.near_dups += 1
                rep.removed.append(("near_dup", c.key))
                continue
            seen_hash.add(h)
            kept.append(c)
            kept_vecs.append(v)
        return kept

    @staticmethod
    def compact_text(text: str, query: str, budget_tokens: int) -> str:
        sents = split_sentences(text)
        if count_tokens(text) <= budget_tokens or len(sents) <= 1:
            return text
        q = set(tokenize(query, True))
        ranked = sorted(range(len(sents)), key=lambda i: (-len(q & set(tokenize(sents[i], True))), i))
        chosen, used = [], 0
        for i in ranked:
            t = count_tokens(sents[i]) + (2 if chosen else 0)  # 2 ≈ cost of the " … " separator
            if used + t > budget_tokens and chosen:
                continue
            chosen.append(i)
            used += t
        return " … ".join(sents[i] for i in sorted(chosen))

    # ------------------------------------------------------------ pipeline
    def optimise(self, items, query: str, final_k: int, token_budget: int, already_loaded=None):
        rep = OptimiserReport(input_items=len(items), input_tokens=sum(count_tokens(c.text) for c in items))
        items = self.filter(items, rep)
        items = self.dedup(items, rep, already_loaded)
        items.sort(key=lambda c: -c.rerank_score)
        rep.dropped_by_k = max(0, len(items) - final_k)
        items = items[:final_k]

        total = sum(count_tokens(c.text) for c in items)
        if total > token_budget and items:
            per_item = max(40, token_budget // len(items))
            for c in items:
                if count_tokens(c.text) > per_item:
                    c.signals["original_tokens"] = count_tokens(c.text)
                    c.text = self.compact_text(c.text, query, per_item)
                    rep.compacted_items += 1
            while items and sum(count_tokens(c.text) for c in items) > token_budget:
                items.pop()
                rep.dropped_by_budget += 1
        rep.output_items = len(items)
        rep.output_tokens = sum(count_tokens(c.text) for c in items)
        return items, rep
