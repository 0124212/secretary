"""Processed-session ledger — idle debounce + exactly-once guard.

session.idle fires repeatedly per session, so the daemon must not
summarize on every event. The ledger records processed sessions in a
small JSON file; pending debounce tasks live in memory in main.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionLedger:
    """Persistent record of processed sessions."""

    def __init__(self, path: str | Path, retention_days: int = 30) -> None:
        self.path = Path(path).expanduser()
        self.retention_days = retention_days
        self._data: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt ledger file, starting fresh: %s", self.path)
            return {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2))
        except OSError:
            logger.warning("Cannot write ledger file: %s", self.path, exc_info=True)

    def is_processed(self, session_id: str) -> bool:
        """True if this session was already summarized and stored."""
        return self._data.get(session_id, {}).get("state") == "processed"

    def mark_processed(self, session_id: str) -> None:
        """Record a session as done. Never raises."""
        self._data[session_id] = {
            "state": "processed",
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.prune()
        self._save()

    def prune(self) -> int:
        """Drop entries older than retention_days. Returns count removed."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        stale = []
        for sid, entry in self._data.items():
            try:
                ts = datetime.fromisoformat(str(entry.get("ts", "")))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    stale.append(sid)
            except (ValueError, TypeError):
                stale.append(sid)
        for sid in stale:
            del self._data[sid]
        return len(stale)
