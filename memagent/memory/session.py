"""Tier 2 - Session memory (SQLite-backed).

Maintains continuity *across interactions*:

* ``sessions``  - one row per session with a rolling summary
* ``messages``  - full conversation history (never deleted; compaction only
                  marks old turns as folded into the summary)
* ``users``     - user preferences (key/value, last write wins)
* ``tasks``     - multi-session task progress: title, status, ordered steps
                  with done flags, free-form notes and the last working-memory
                  snapshot so an interrupted workflow can be resumed exactly.

Compaction: once more than ``compact_after`` un-summarised turns accumulate,
everything older than the most recent ``recent_turns`` is summarised by the LLM
into ``sessions.summary`` and flagged ``compacted=1``.  The prompt therefore
carries *summary + last N turns* instead of the whole transcript, keeping the
session tier O(1) in prompt size.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..text import count_tokens


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionMemory:
    def __init__(self, db_path: Path, recent_turns: int = 6, compact_after: int = 10):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.recent_turns = recent_turns
        self.compact_after = compact_after
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, preferences TEXT DEFAULT '{}');
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, user_id TEXT, started_at TEXT, ended_at TEXT,
                summary TEXT DEFAULT '');
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, user_id TEXT, role TEXT,
                content TEXT, ts TEXT, tokens INTEGER, compacted INTEGER DEFAULT 0, meta TEXT DEFAULT '{}');
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY, user_id TEXT, title TEXT, status TEXT, steps TEXT,
                notes TEXT DEFAULT '', working_snapshot TEXT DEFAULT '{}', updated_at TEXT,
                session_id TEXT);
            """
        )
        self.db.commit()

    # -------------------------------------------------------------- sessions
    def start_session(self, user_id: str) -> str:
        sid = "s-" + uuid.uuid4().hex[:8]
        self.db.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
        self.db.execute(
            "INSERT INTO sessions (session_id, user_id, started_at) VALUES (?,?,?)", (sid, user_id, now())
        )
        self.db.commit()
        return sid

    def end_session(self, session_id: str, summary: str | None = None) -> None:
        if summary is not None:
            self.db.execute("UPDATE sessions SET summary=? WHERE session_id=?", (summary, session_id))
        self.db.execute("UPDATE sessions SET ended_at=? WHERE session_id=?", (now(), session_id))
        self.db.commit()

    def summary(self, session_id: str) -> str:
        r = self.db.execute("SELECT summary FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return r["summary"] if r else ""

    def previous_sessions(self, user_id: str, exclude: str | None = None, limit: int = 3) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM sessions WHERE user_id=? AND session_id!=? AND ended_at IS NOT NULL "
            "ORDER BY started_at DESC, rowid DESC LIMIT ?",
            (user_id, exclude or "", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- messages
    def add_message(self, session_id: str, user_id: str, role: str, content: str, meta: dict | None = None) -> None:
        self.db.execute(
            "INSERT INTO messages (session_id, user_id, role, content, ts, tokens, meta) VALUES (?,?,?,?,?,?,?)",
            (session_id, user_id, role, content, now(), count_tokens(content), json.dumps(meta or {})),
        )
        self.db.commit()

    def messages(self, session_id: str, include_compacted: bool = True) -> list[dict]:
        q = "SELECT * FROM messages WHERE session_id=?" + ("" if include_compacted else " AND compacted=0")
        return [dict(r) for r in self.db.execute(q + " ORDER BY id", (session_id,))]

    def recent(self, session_id: str, n: int | None = None) -> list[dict]:
        live = self.messages(session_id, include_compacted=False)
        return live[-(n or self.recent_turns):]

    def maybe_compact(self, session_id: str, llm) -> dict | None:
        """Fold old turns into the rolling summary when the live window is too long."""
        live = self.messages(session_id, include_compacted=False)
        if len(live) <= self.compact_after:
            return None
        to_fold = live[: -self.recent_turns]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in to_fold)
        prior = self.summary(session_id)
        new_summary = llm.summarise((f"Earlier summary: {prior}\n" if prior else "") + transcript, max_words=90)
        ids = [m["id"] for m in to_fold]
        self.db.execute(
            f"UPDATE messages SET compacted=1 WHERE id IN ({','.join('?' * len(ids))})", ids
        )
        self.db.execute("UPDATE sessions SET summary=? WHERE session_id=?", (new_summary, session_id))
        self.db.commit()
        before = sum(m["tokens"] for m in to_fold)
        return {"folded_turns": len(ids), "tokens_before": before, "tokens_after": count_tokens(new_summary)}

    # ----------------------------------------------------------- preferences
    def set_preference(self, user_id: str, key: str, value: str) -> None:
        prefs = self.preferences(user_id)
        prefs[key] = value
        self.db.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
        self.db.execute("UPDATE users SET preferences=? WHERE user_id=?", (json.dumps(prefs), user_id))
        self.db.commit()

    def preferences(self, user_id: str) -> dict:
        r = self.db.execute("SELECT preferences FROM users WHERE user_id=?", (user_id,)).fetchone()
        return json.loads(r["preferences"]) if r else {}

    # ----------------------------------------------------------------- tasks
    def upsert_task(
        self,
        user_id: str,
        title: str,
        steps: list[dict] | None = None,
        status: str = "in_progress",
        notes: str | None = None,
        session_id: str | None = None,
        working_snapshot: dict | None = None,
        task_id: str | None = None,
    ) -> str:
        existing = self.get_task(task_id) if task_id else self.find_task(user_id, title)
        tid = existing["task_id"] if existing else (task_id or "t-" + uuid.uuid4().hex[:6])
        steps = steps if steps is not None else (existing["steps"] if existing else [])
        notes = notes if notes is not None else (existing["notes"] if existing else "")
        snap = working_snapshot if working_snapshot is not None else (existing["working_snapshot"] if existing else {})
        self.db.execute(
            "INSERT OR REPLACE INTO tasks (task_id, user_id, title, status, steps, notes, working_snapshot, "
            "updated_at, session_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (tid, user_id, title, status, json.dumps(steps), notes, json.dumps(snap), now(), session_id),
        )
        self.db.commit()
        return tid

    def complete_step(self, task_id: str, step_index: int, note: str = "") -> None:
        t = self.get_task(task_id)
        if not t or step_index >= len(t["steps"]):
            return
        t["steps"][step_index]["done"] = True
        if note:
            t["steps"][step_index]["note"] = note
        status = "done" if all(s.get("done") for s in t["steps"]) else "in_progress"
        self.upsert_task(t["user_id"], t["title"], t["steps"], status, task_id=task_id, session_id=t["session_id"])

    def _row(self, r) -> dict:
        d = dict(r)
        d["steps"] = json.loads(d["steps"] or "[]")
        d["working_snapshot"] = json.loads(d["working_snapshot"] or "{}")
        return d

    def get_task(self, task_id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return self._row(r) if r else None

    def find_task(self, user_id: str, title: str) -> dict | None:
        r = self.db.execute("SELECT * FROM tasks WHERE user_id=? AND title=?", (user_id, title)).fetchone()
        return self._row(r) if r else None

    def open_tasks(self, user_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM tasks WHERE user_id=? AND status!='done' ORDER BY updated_at DESC", (user_id,)
        ).fetchall()
        return [self._row(r) for r in rows]

    @staticmethod
    def render_task(t: dict) -> str:
        done = sum(1 for s in t["steps"] if s.get("done"))
        lines = [f"- {t['title']} — {done}/{len(t['steps'])} steps done (status: {t['status']})"]
        nxt = next((s for s in t["steps"] if not s.get("done")), None)
        for s in t["steps"]:
            lines.append(f"    [{'x' if s.get('done') else ' '}] {s['name']}" + (f" — {s['note']}" if s.get("note") else ""))
        if nxt:
            lines.append(f"    next: {nxt['name']}")
        if t.get("notes"):
            lines.append(f"    notes: {t['notes']}")
        return "\n".join(lines)
