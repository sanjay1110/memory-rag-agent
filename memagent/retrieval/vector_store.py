"""Persistent FAISS vector store.

Vectors live in a FAISS ``IndexIDMap2(IndexFlatIP)`` (exact cosine search on
normalised vectors; supports delete-by-id).  Payloads (text + metadata) live in
a SQLite table keyed by the same integer id, so the store survives restarts
and can be inspected with any SQLite browser.

One ``VectorStore`` instance == one collection (e.g. ``knowledge`` or
``memories``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, directory: Path, name: str, dim: int):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.dim = dim
        self.index_path = self.dir / f"{name}.faiss"
        self.db = sqlite3.connect(self.dir / f"{name}.sqlite", check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, key TEXT UNIQUE, "
            "text TEXT, meta TEXT)"
        )
        self.db.commit()
        self._lock = threading.Lock()
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    # ------------------------------------------------------------------ write
    def upsert(self, keys: list[str], texts: list[str], vectors: np.ndarray, metas: list[dict]):
        with self._lock:
            ids = []
            for key, text, meta in zip(keys, texts, metas):
                row = self.db.execute("SELECT id FROM items WHERE key=?", (key,)).fetchone()
                if row:
                    self.index.remove_ids(np.array([row[0]], dtype=np.int64))
                    self.db.execute(
                        "UPDATE items SET text=?, meta=? WHERE id=?", (text, json.dumps(meta), row[0])
                    )
                    ids.append(row[0])
                else:
                    cur = self.db.execute(
                        "INSERT INTO items (key, text, meta) VALUES (?,?,?)", (key, text, json.dumps(meta))
                    )
                    ids.append(cur.lastrowid)
            self.index.add_with_ids(vectors.astype(np.float32), np.array(ids, dtype=np.int64))
            self.db.commit()
            self._persist()
            return ids

    def update_meta(self, key: str, meta: dict) -> None:
        with self._lock:
            self.db.execute("UPDATE items SET meta=? WHERE key=?", (json.dumps(meta), key))
            self.db.commit()

    def delete(self, key: str) -> bool:
        with self._lock:
            row = self.db.execute("SELECT id FROM items WHERE key=?", (key,)).fetchone()
            if not row:
                return False
            self.index.remove_ids(np.array([row[0]], dtype=np.int64))
            self.db.execute("DELETE FROM items WHERE id=?", (row[0],))
            self.db.commit()
            self._persist()
            return True

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))

    # ------------------------------------------------------------------- read
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, str, dict, float]]:
        if self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(vector.reshape(1, -1).astype(np.float32), min(k, self.index.ntotal))
        out = []
        for score, iid in zip(scores[0], ids[0]):
            if iid < 0:
                continue
            row = self.db.execute("SELECT key, text, meta FROM items WHERE id=?", (int(iid),)).fetchone()
            if row:
                out.append((row[0], row[1], json.loads(row[2]), float(score)))
        return out

    def get(self, key: str):
        row = self.db.execute("SELECT key, text, meta FROM items WHERE key=?", (key,)).fetchone()
        return (row[0], row[1], json.loads(row[2])) if row else None

    def get_vector(self, key: str) -> np.ndarray | None:
        row = self.db.execute("SELECT id FROM items WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        return self.index.reconstruct(int(row[0]))

    def all(self) -> list[tuple[str, str, dict]]:
        return [
            (k, t, json.loads(m))
            for k, t, m in self.db.execute("SELECT key, text, meta FROM items ORDER BY id")
        ]

    def __len__(self) -> int:
        return int(self.index.ntotal)
