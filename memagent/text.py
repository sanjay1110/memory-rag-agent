"""Small text utilities shared by every module (tokenising, token counting,
sentence splitting, normalisation)."""

from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"[a-z0-9][a-z0-9_\-\.]*[a-z0-9]|[a-z0-9]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])")

# A compact, hand-picked list: generic lists (e.g. scikit-learn's) also drop
# words such as "move", "keep", "next" and "back" that carry meaning in
# operational questions, and measurably hurt retrieval on the eval set.
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    how i if in into is it its me my of on or our so that the their them then there these
    they this to was we were what when where which who why will with would you your about
    should any all also just than too very more most some such only own same other s t
    without hey hi hello ok okay please thanks""".split()
)


def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    toks = _WORD.findall(text.lower())
    toks = [t.strip(".") for t in toks if t.strip(".")]
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def count_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (~0.75 words per token)."""
    if not text:
        return 0
    return max(1, round(len(text.split()) * 1.33))


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def content_hash(text: str) -> str:
    return hashlib.sha1(normalise(text).encode()).hexdigest()[:16]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a, True)), set(tokenize(b, True))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def truncate_to_tokens(text: str, budget: int) -> str:
    """Truncate to roughly ``budget`` tokens, preserving line structure
    (whole lines first, then words inside the last line that fits)."""
    if count_tokens(text) <= budget:
        return text
    out, used = [], 0
    for line in text.splitlines():
        t = count_tokens(line)
        if used + t <= budget:
            out.append(line)
            used += t
            continue
        keep = int((budget - used) / 1.33)
        if keep > 3:
            out.append(" ".join(line.split()[:keep]) + " …")
        break
    return "\n".join(out)
