"""Tier 3 - Long-term memory (FAISS + SQLite, persistent across sessions).

Two collections share one embedder:

* ``knowledge`` - the document corpus (chunked Markdown), i.e. what the agent
  *knows*.  Written by ingestion, read by RAG.
* ``memories``  - what the agent has *learned about the user and the work*:
  facts, preferences, decisions and session episodes.  Written by the
  consolidation step at the end of every session (and by explicit
  ``remember`` calls), read by both upfront loading and JIT retrieval.

Memory quality controls applied on write:

* **Semantic de-duplication** - a new memory whose cosine similarity to an
  existing memory of the same user exceeds ``dedup_threshold`` is *merged*
  (access count and importance bumped) instead of stored twice.
* **Slot-based supersession** - memories with a ``slot`` (e.g.
  ``pref:language``) replace the previous value for that slot; the old value
  is kept in ``history`` so the change is auditable.

Scoring on read multiplies the reranker score by a prior
``importance x recency`` (exponential decay, configurable half-life) so stale,
low-importance memories fade without being deleted.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..retrieval.chunking import Chunk
from ..retrieval.index import HybridIndex
from .session import now


@dataclass
class MemoryRecord:
    key: str
    text: str
    kind: str
    importance: float
    meta: dict


class LongTermMemory:
    def __init__(self, directory: Path, embedder, dedup_threshold: float = 0.9, half_life_days: float = 30.0):
        self.embedder = embedder
        self.knowledge = HybridIndex(directory, "knowledge", embedder)
        self.memories = HybridIndex(directory, "memories", embedder)
        self.dedup_threshold = dedup_threshold
        self.half_life_days = half_life_days
        self.stats = {"written": 0, "merged": 0, "superseded": 0}

    # ------------------------------------------------------------ knowledge
    def ingest_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self.knowledge.add(
            [c.chunk_id for c in chunks],
            [c.text for c in chunks],
            [
                {"source": c.source, "title": c.title, "section": c.section, "position": c.position, **c.metadata}
                for c in chunks
            ],
            embed_texts=[c.embed_text for c in chunks],
        )
        return len(chunks)

    # ------------------------------------------------------------- memories
    def remember(
        self,
        user_id: str,
        text: str,
        kind: str = "fact",
        importance: float = 0.5,
        session_id: str = "",
        slot: str | None = None,
    ) -> dict:
        """Write a memory with dedup / supersession. Returns what happened."""
        text = text.strip()
        vec = self.embedder.embed([text])[0]

        if slot:
            for key, old_text, meta in self.memories.store.all():
                if meta.get("user_id") == user_id and meta.get("slot") == slot:
                    if old_text.strip().lower() == text.lower():
                        return self._touch(key, meta, importance, "merged")
                    meta.setdefault("history", []).append({"text": old_text, "until": now()})
                    meta.update(importance=max(meta["importance"], importance), updated_at=now(), session_id=session_id)
                    self.memories.add([key], [text], [meta])
                    self.stats["superseded"] += 1
                    return {"action": "superseded", "key": key, "previous": old_text}

        for key, old_text, meta, score in self.memories.store.search(vec, 5):
            if meta.get("user_id") == user_id and score >= self.dedup_threshold:
                return self._touch(key, meta, importance, "merged", similarity=score)

        key = "m-" + uuid.uuid4().hex[:10]
        meta = {
            "user_id": user_id,
            "kind": kind,
            "importance": importance,
            "created_at": now(),
            "updated_at": now(),
            "last_accessed": now(),
            "access_count": 0,
            "session_id": session_id,
            "section": kind,
            "title": "memory",
        }
        if slot:
            meta["slot"] = slot
        self.memories.add([key], [text], [meta])
        self.stats["written"] += 1
        return {"action": "written", "key": key}

    def _touch(self, key, meta, importance, action, similarity=None):
        meta["access_count"] = meta.get("access_count", 0) + 1
        meta["importance"] = min(1.0, max(meta.get("importance", 0.5), importance) + 0.05)
        meta["updated_at"] = now()
        self.memories.store.update_meta(key, meta)
        self.stats["merged"] += 1
        out = {"action": action, "key": key}
        if similarity is not None:
            out["similarity"] = round(similarity, 3)
        return out

    def prior(self, meta: dict) -> float:
        """importance x recency, both in [0,1]."""
        try:
            ts = datetime.fromisoformat(meta.get("updated_at") or meta.get("created_at"))
            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
        except Exception:
            age_days = 0.0
        recency = math.pow(0.5, age_days / self.half_life_days)
        return float(meta.get("importance", 0.5)) * (0.5 + 0.5 * recency)

    def mark_accessed(self, keys: list[str]) -> None:
        for key in keys:
            got = self.memories.store.get(key)
            if got:
                meta = got[2]
                meta["access_count"] = meta.get("access_count", 0) + 1
                meta["last_accessed"] = now()
                self.memories.store.update_meta(key, meta)

    def user_memories(self, user_id: str, kinds: tuple[str, ...] | None = None) -> list[MemoryRecord]:
        out = []
        for key, text, meta in self.memories.store.all():
            if meta.get("user_id") != user_id or (kinds and meta.get("kind") not in kinds):
                continue
            out.append(MemoryRecord(key, text, meta.get("kind", "fact"), meta.get("importance", 0.5), meta))
        return out

    def top_memories(self, user_id: str, n: int = 5, kinds=None) -> list[MemoryRecord]:
        mems = self.user_memories(user_id, kinds)
        mems.sort(key=lambda m: -self.prior(m.meta))
        return mems[:n]


# ---------------------------------------------------------------- extraction
_PATTERNS: list[tuple[re.Pattern, str, float, str | None]] = [
    (re.compile(r"\bmy name is ([A-Z][\w\-]+)", re.I), "fact", 0.9, "profile:name"),
    (re.compile(r"\bI(?:'d| would)? prefer (?:to )?(.+?)(?:[.!?]|$)", re.I), "preference", 0.8, None),
    (re.compile(r"\b(?:please )?(?:always|keep) (?:answers?|responses?|replies) (.+?)(?:[.!?]|$)", re.I), "preference", 0.8, "pref:style"),
    (re.compile(r"\bwe(?:'ve| have)? decided (?:to )?(.+?)(?:[.!?]|$)", re.I), "decision", 0.85, None),
    (re.compile(r"\blet'?s go with (.+?)(?:[.!?]|$)", re.I), "decision", 0.8, None),
    (re.compile(r"\bremember that (.+?)(?:[.!?]|$)", re.I), "fact", 0.9, None),
    (re.compile(r"\b(?:the )?deadline is (.+?)(?:[.!?]|$)", re.I), "fact", 0.85, "project:deadline"),
    (re.compile(r"\bI(?:'m| am) ((?:working on|migrating|building) .+?)(?:[.!?]|$)", re.I), "fact", 0.75, "project:current"),
    (re.compile(r"\bour (?:service|app|team|stack) (?:is|uses|runs) (.+?)(?:[.!?]|$)", re.I), "fact", 0.7, None),
    (re.compile(r"\b(?:we use|we're using|we are using) (.+?)(?:[.!?]|$)", re.I), "fact", 0.7, None),
    (re.compile(r"\bour region is ([\w\-]+)", re.I), "fact", 0.8, "project:region"),
]

_PREF_SLOTS = {
    "python": "pref:language", "go": "pref:language", "typescript": "pref:language", "java": "pref:language",
    "concise": "pref:style", "short": "pref:style", "detailed": "pref:style", "bullet": "pref:style",
}


def extract_memories(user_text: str) -> list[dict]:
    """Rule-based extraction of durable facts from a single user utterance.

    Kept rule-based (rather than LLM) so it is deterministic and testable; the
    agent can additionally run an LLM consolidation pass when a model is
    available.  Each result: {text, kind, importance, slot}.
    """
    found = []
    for pat, kind, imp, slot in _PATTERNS:
        for m in pat.finditer(user_text):
            val = m.group(1).strip().rstrip(",;")
            if len(val) < 2:
                continue
            if kind == "preference":
                text = f"User prefers {val}"
                if slot is None:
                    slot = next((s for w, s in _PREF_SLOTS.items() if re.search(rf"\b{w}\b", val, re.I)), None)
            elif kind == "decision":
                text = f"Decision: {val}"
            elif slot == "profile:name":
                text = f"User's name is {val}"
            elif slot == "project:deadline":
                text = f"Project deadline is {val}"
            elif slot == "project:current":
                text = f"User is {val}"
            elif slot == "project:region":
                text = f"User's deployment region is {val}"
            else:
                text = val[0].upper() + val[1:]
            found.append({"text": text, "kind": kind, "importance": imp, "slot": slot})
    return found
