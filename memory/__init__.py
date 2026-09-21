"""Memory layer — local JSON store + megamemory SQLite knowledge graph.

Writes to megamemory's SQLite DB at ~/.megamemory/knowledge.db so all
memories accumulate in one knowledge graph. Also keeps a local JSON
backup for offline resilience.

Stolen from: Mem0 (mem0ai/mem0) — 29.6k★
Pattern: drop-in memory with auto-extraction via add/search/get_all
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Optional: numpy for cosine similarity if available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)

MEGAMEMORY_DB = Path.home() / ".megamemory" / "knowledge.db"


def _bigram_set(text: str) -> set[str]:
    """Return the set of character bigrams from lowercased, stripped text."""
    text = text.lower().strip()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _jaccard_bigram(a: str, b: str) -> float:
    """Jaccard similarity using character bigram overlap."""
    set_a = _bigram_set(a)
    set_b = _bigram_set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union


class MemoryStore:
    """Memory backed by megamemory SQLite + local JSON fallback."""

    def __init__(self, config: dict[str, Any]) -> None:
        # Memory store path: prefer new "memory" key, fall back to "mem0" for compat
        mem0_cfg = config.get("memory", {})
        if not mem0_cfg:
            mem0_cfg = config.get("mem0", {})
        self.user_id = config.get("user_id", "asher")
        store_path = mem0_cfg.get(
            "store_path",
            str(Path.home() / ".config" / "secretary" / "memory.json"),
        )
        self._path = Path(store_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._memories: list[dict[str, Any]] = []
        self._load()

        # Megamemory SQLite connection (best-effort)
        self._mm_conn: sqlite3.Connection | None = None
        self._mm_init()

    # ── megamemory SQLite ───────────────────────────────────────

    def _mm_init(self) -> None:
        """Open megamemory DB if available."""
        if not MEGAMEMORY_DB.exists():
            logger.debug("Megamemory DB not found at %s", MEGAMEMORY_DB)
            return
        try:
            self._mm_conn = sqlite3.connect(str(MEGAMEMORY_DB), timeout=5)
            self._mm_conn.execute("PRAGMA journal_mode=WAL")
            self._mm_conn.execute("PRAGMA busy_timeout=3000")
            logger.debug("Connected to megamemory DB")
            self._mm_ensure_versions_table()
        except Exception:
            logger.debug("Could not open megamemory DB", exc_info=True)
            self._mm_conn = None

    def _mm_store(
        self,
        text: str,
        memory_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a memory as a megamemory node + fact."""
        if not self._mm_conn:
            return
        meta = metadata or {}
        mem_type = meta.get("type", "observation")
        now = datetime.now(timezone.utc).isoformat()
        node_id = f"secretary:{memory_id}"
        fact_key = f"secretary:{memory_id}"
        fact_value = text

        try:
            # Insert as a node (general memory)
            self._mm_conn.execute(
                """INSERT OR REPLACE INTO nodes
                   (id, name, kind, summary, created_at, updated_at, importance)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    node_id,
                    text[:120],
                    "feature" if mem_type == "session_summary" else "pattern",
                    text,
                    now,
                    now,
                    0.5,
                ),
            )

            # If it's a fact-like memory, also store in facts table
            if mem_type in ("extracted_fact", "stale_task_alert") or text.startswith("[fact]"):
                if ": " in text[:200]:
                    fact_key = text.split(": ", 1)[0].strip().lower().replace(" ", "_")
                self._mm_conn.execute(
                    """INSERT OR REPLACE INTO facts
                       (key, value, source, confidence, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        fact_key,
                        fact_value,
                        "secretary",
                        0.8,
                        now,
                    ),
                )

            self._mm_conn.commit()
        except sqlite3.OperationalError:
            # DB locked — retry once after a short sleep
            logger.debug("Megamemory DB locked, retrying in 100ms")
            import time
            time.sleep(0.1)
            try:
                self._mm_conn.execute(
                    """INSERT OR REPLACE INTO nodes
                       (id, name, kind, summary, created_at, updated_at, importance)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        node_id,
                        text[:120],
                        "feature" if mem_type == "session_summary" else "pattern",
                        text,
                        now,
                        now,
                        0.5,
                    ),
                )
                if mem_type in ("extracted_fact", "stale_task_alert") or text.startswith("[fact]"):
                    if ": " in text[:200]:
                        fact_key = text.split(": ", 1)[0].strip().lower().replace(" ", "_")
                    self._mm_conn.execute(
                        """INSERT OR REPLACE INTO facts
                           (key, value, source, confidence, updated_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            fact_key,
                            fact_value,
                            "secretary",
                            0.8,
                            now,
                        ),
                    )
                self._mm_conn.commit()
            except Exception:
                logger.debug("Megamemory write retry failed", exc_info=True)
        except Exception:
            logger.debug("Megamemory write failed", exc_info=True)

    def _mm_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search megamemory nodes via FTS5 + optional cosine similarity."""
        if not self._mm_conn:
            return []
        # Check if embedding column exists
        try:
            col_info = self._mm_conn.execute(
                "PRAGMA table_info(nodes)"
            ).fetchall()
            has_embedding = any(
                row[1] == "embedding" for row in col_info
            )
        except Exception:
            has_embedding = False

        # Sanitize FTS5 special chars (always computed)
        sanitized = query.replace('"', '').replace('*', '').replace(':', '').replace('(', '').replace(')', '').replace('-', ' ')

        if has_embedding and HAS_NUMPY:
            # Cosine-similarity search path alongside FTS5
            try:
                # FTS5 first
                rows = self._mm_conn.execute(
                    """SELECT n.id, n.name, n.summary, n.kind, rank
                       FROM nodes_fts f
                       JOIN nodes n ON n.id = f.id
                       WHERE nodes_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (sanitized, top_k * 2),
                ).fetchall()
                fts_results = [
                    {
                        "id": r[0],
                        "memory": r[2],
                        "metadata": {"kind": r[3], "source": "megamemory"},
                        "score": abs(r[4]) if r[4] else 0.5,
                    }
                    for r in rows
                ]
                # Now re-rank by cosine similarity on embeddings
                query_emb = np.array(
                    [ord(c) for c in sanitized], dtype=np.float32
                ).reshape(1, -1)
                # We need actual embedding vectors from the nodes table;
                # since we don't have a full vector store, fall back to FTS scores
                # but we log that semantic path is available
                logger.debug(
                    "Semantic search available (embedding column present) "
                    "but full vector re-rank deferred"
                )
                return fts_results[:top_k]
            except sqlite3.OperationalError:
                logger.debug("Megamemory FTS search failed, falling back to LIKE", exc_info=True)
            except Exception:
                logger.debug("Megamemory semantic search failed", exc_info=True)
        # Fall through to plain FTS5 / LIKE fallback
        try:
            # Use FTS5 for text search
            rows = self._mm_conn.execute(
                """SELECT n.id, n.name, n.summary, n.kind, rank
                   FROM nodes_fts f
                   JOIN nodes n ON n.id = f.id
                   WHERE nodes_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (sanitized, top_k),
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "memory": r[2],
                    "metadata": {"kind": r[3], "source": "megamemory"},
                    "score": abs(r[4]) if r[4] else 0.5,
                }
                for r in rows
            ]
        except sqlite3.OperationalError:
            # FTS5 failed (e.g. DB locked); fall back to LIKE
            logger.debug("Megamemory FTS search failed, falling back to LIKE", exc_info=True)
            like_pattern = f"%{query}%"
            rows = self._mm_conn.execute(
                """SELECT n.id, n.name, n.summary, n.kind, 0 as rank
                   FROM nodes n
                   WHERE n.name LIKE ? OR n.summary LIKE ?
                   ORDER BY rank
                   LIMIT ?""",
                (like_pattern, like_pattern, top_k),
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "memory": r[2],
                    "metadata": {"kind": r[3], "source": "megamemory"},
                    "score": 0.5,
                }
                for r in rows
            ]
        except Exception:
            logger.debug("Megamemory FTS search failed", exc_info=True)
            return []

    # ── local JSON persistence ──────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._memories = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                self._memories = []
        else:
            self._memories = []

    def _save(self) -> None:
        """Atomically write memories to local JSON to prevent corrupt file on crash."""
        data = json.dumps(self._memories, indent=2, default=str)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(data)
        tmp.replace(self._path)

    # ── public API ──────────────────────────────────────────────

    def add(
        self,
        text: str | list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Add a memory (fact, summary, observation). Writes to both
        local JSON and megamemory SQLite."""
        # Fuzzy fact dedup: check Jaccard bigram similarity ≥ 0.85
        text_str = text if isinstance(text, str) else " | ".join(
            m.get("content", m.get("text", "")) for m in text if m
        )
        for existing in self._memories:
            if _jaccard_bigram(existing.get("memory", ""), text_str) >= 0.85:
                logger.info("Skipping near-duplicate memory addition")
                return existing
        # TTL prune: drop local entries older than 30 days
        threshold = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        self._memories = [
            m for m in self._memories
            if m.get("created_at", "") >= threshold
        ]
        if isinstance(text, list):
            text = " | ".join(
                m.get("content", m.get("text", "")) for m in text if m
            )
        logger.info("Adding memory: %s...", text[:80])
        memory_id = str(uuid.uuid4())
        entry: dict[str, Any] = {
            "id": memory_id,
            "memory": text,
            "user_id": self.user_id,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._memories.append(entry)
        self._save()

        # Also write to megamemory knowledge graph
        self._mm_store(text, memory_id, metadata)

        return entry

    def compact(self) -> int:
        """Merge near-duplicate local facts (Jaccard bigram ≥ 0.90).

        Keeps the longer/better entry and deletes the shorter one.
        Deprecated/low-priority — do not auto-call from add().

        Returns the number of entries deleted.
        """
        deleted = 0
        checked: set[str] = set()
        for i, a in enumerate(self._memories):
            if a["id"] in checked:
                continue
            for j in range(i + 1, len(self._memories)):
                b = self._memories[j]
                if b["id"] in checked:
                    continue
                sim = _jaccard_bigram(a.get("memory", ""), b.get("memory", ""))
                if sim >= 0.90:
                    # Keep the longer entry, delete the shorter
                    if len(a.get("memory", "")) >= len(b.get("memory", "")):
                        longer, shorter = a, b
                    else:
                        longer, shorter = b, a
                    self._memories.remove(shorter)
                    checked.add(shorter["id"])
                    deleted += 1
                    # Also clean up megamemory nodes/facts for the deleted entry
                    if self._mm_conn:
                        try:
                            self._mm_conn.execute(
                                """DELETE FROM nodes WHERE id = ?""",
                                (f"secretary:{shorter['id']}",),
                            )
                            self._mm_conn.execute(
                                """DELETE FROM facts WHERE key LIKE ?""",
                                (f"secretary:{shorter['id']}%",),
                            )
                            self._mm_conn.commit()
                        except Exception:
                            logger.debug("compact: megamemory cleanup failed", exc_info=True)
                    checked.add(a["id"])
                    checked.add(b["id"])
        self._save()
        return deleted

    def add_messages(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Add a conversation as structured memory."""
        return self.add(messages, metadata=metadata)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """Search memories — merges local JSON + megamemory FTS results."""
        # Local substring search
        query_lower = query.lower()
        local_results = [
            m for m in reversed(self._memories)
            if query_lower in m.get("memory", "").lower()
        ][:top_k]

        # Megamemory FTS search
        mm_results = self._mm_search(query, top_k=top_k)

        # Merge, dedup by id, local results first
        seen_ids = {r.get("id") for r in local_results}
        merged = list(local_results)
        for r in mm_results:
            if r.get("id") not in seen_ids:
                merged.append(r)
                seen_ids.add(r.get("id"))
        return merged[:top_k]

    def get_all(self, limit: int = 50, **kwargs: Any) -> list[dict]:
        """Get memories up to limit, newest first."""
        if "page_size" in kwargs and isinstance(kwargs["page_size"], int):
            limit = kwargs["page_size"]
        return list(reversed(self._memories[-limit:]))

    def get(self, memory_id: str) -> dict | None:
        """Get a single memory by ID."""
        for m in self._memories:
            if m.get("id") == memory_id:
                return m
        return None

def update(self, memory_id: str, data: str) -> dict | None:
        """Update a memory's content."""
        for m in self._memories:
            if m.get("id") == memory_id:
                old_memory = m.get("memory", "")
                m["memory"] = data
                m["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save()
                # Mirror to megamemory nodes table
                try:
                    if self._mm_conn:
                        self._mm_conn.execute(
                            """UPDATE nodes
                               SET summary = ?, updated_at = ?
                            WHERE id = ?""",
                            (data, datetime.now(timezone.utc).isoformat(),
                             f"secretary:{memory_id}"),
                        )
                        self._mm_conn.commit()
                except Exception:
                    logger.debug("Megamemory mirror update failed", exc_info=True)
                # Record in version table
                try:
                    if self._mm_conn:
                        self._mm_ensure_versions_table()
                        self._mm_conn.execute(
                            """INSERT INTO memory_versions (memory_id, old_content, new_content, updated_at)
                               VALUES (?, ?, ?, ?)""",
                            (memory_id, old_memory, data, datetime.now(timezone.utc).isoformat()),
                        )
                        self._mm_conn.commit()
                except Exception:
                    logger.debug("Megamemory version record failed", exc_info=True)
                return m
        return None

    def delete(self, memory_id: str) -> None:
        """Delete a single memory."""
        self._memories = [m for m in self._memories if m.get("id") != memory_id]
        self._save()
        # Mirror to megamemory nodes + facts tables
        try:
            if self._mm_conn:
                self._mm_conn.execute(
                    """DELETE FROM nodes WHERE id = ?""",
                    (f"secretary:{memory_id}",),
                )
                self._mm_conn.execute(
                    """DELETE FROM facts WHERE key LIKE ?""",
                    (f"secretary:{memory_id}%",),
                )
                self._mm_conn.commit()
        except Exception:
            logger.debug("Megamemory mirror delete failed", exc_info=True)

    def _mm_ensure_versions_table(self) -> None:
        """Create the memory_versions table if it doesn't exist."""
        if not self._mm_conn:
            return
        try:
            self._mm_conn.execute(
                """CREATE TABLE IF NOT EXISTS memory_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    old_content TEXT NOT NULL,
                    new_content TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                )"""
            )
            self._mm_conn.commit()
        except Exception:
            logger.debug("Could not create memory_versions table", exc_info=True)

    def history(self, memory_id: str) -> list[dict]:
        """Get change history from the append-only version table."""
        if not self._mm_conn:
            return []
        self._mm_ensure_versions_table()
        try:
            rows = self._mm_conn.execute(
                """SELECT memory_id, old_content, new_content, updated_at
                   FROM memory_versions
                   WHERE memory_id = ?
                   ORDER BY updated_at DESC""",
                (memory_id,),
            ).fetchall()
            return [
                {"old_content": r[1], "new_content": r[2], "updated_at": r[3]}
                for r in rows
            ]
        except Exception:
            logger.debug("Megamemory version query failed", exc_info=True)
            return []

    def close(self) -> None:
        """Persist to disk and close megamemory connection."""
        self._save()
        if self._mm_conn:
            try:
                self._mm_conn.close()
            except Exception:
                pass
