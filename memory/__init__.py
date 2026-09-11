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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEGAMEMORY_DB = Path.home() / ".megamemory" / "knowledge.db"


class MemoryStore:
    """Memory backed by megamemory SQLite + local JSON fallback."""

    def __init__(self, config: dict[str, Any]) -> None:
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

        try:
            # Insert as a node (general memory)
            node_id = f"secretary:{memory_id}"
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
                fact_key = f"secretary:{memory_id}"
                fact_value = text
                # Try to extract a key from the text
                if ": " in text[:200]:
                    fact_key = text.split(": ", 1)[0].strip().lower().replace(" ", "_")
                    fact_value = text
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
            logger.debug("Megamemory write failed", exc_info=True)

    def _mm_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search megamemory nodes via FTS5."""
        if not self._mm_conn:
            return []
        try:
            # Use FTS5 for text search
            rows = self._mm_conn.execute(
                """SELECT n.id, n.name, n.summary, n.kind, rank
                   FROM nodes_fts f
                   JOIN nodes n ON n.id = f.id
                   WHERE nodes_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, top_k),
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
        self._path.write_text(json.dumps(self._memories, indent=2, default=str))

    # ── public API ──────────────────────────────────────────────

    def add(
        self,
        text: str | list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Add a memory (fact, summary, observation). Writes to both
        local JSON and megamemory SQLite."""
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
                m["memory"] = data
                m["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save()
                return m
        return None

    def delete(self, memory_id: str) -> None:
        """Delete a single memory."""
        self._memories = [m for m in self._memories if m.get("id") != memory_id]
        self._save()

    def history(self, memory_id: str) -> list[dict]:
        """Get change history (stub — local store has no versioning)."""
        m = self.get(memory_id)
        return [m] if m else []

    def close(self) -> None:
        """Persist to disk and close megamemory connection."""
        self._save()
        if self._mm_conn:
            try:
                self._mm_conn.close()
            except Exception:
                pass
