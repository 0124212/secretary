"""Memory layer — local JSON store + opencode serve inference.

No Mem0, no Qdrant, no OpenAI key needed. Stores memories in a
local JSON file and uses opencode serve for summarization/embeddings.

Stolen from: Mem0 (mem0ai/mem0) — 29.6k★
Pattern: drop-in memory with auto-extraction via add/search/get_all
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryStore:
    """Local JSON-file memory with opencode-backed search."""

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

    # ── persistence ──────────────────────────────────────────────

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

    # ── public API (same shape as before) ────────────────────────

    def add(
        self,
        text: str | list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Add a memory (fact, summary, observation)."""
        if isinstance(text, list):
            # conversation-style input → flatten
            text = " | ".join(
                m.get("content", m.get("text", "")) for m in text if m
            )
        logger.info("Adding memory: %s...", text[:80])
        entry: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "memory": text,
            "user_id": self.user_id,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._memories.append(entry)
        self._save()
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
        """Search memories — substring match, returns newest first."""
        query_lower = query.lower()
        results = [
            m for m in reversed(self._memories)
            if query_lower in m.get("memory", "").lower()
        ]
        return results[:top_k]

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
        """Persist to disk."""
        self._save()
