#!/usr/bin/env python3
"""Secretary daemon — main entry point.

A background process that acts as a coding session secretary:
  - Listens to opencode events (session lifecycle)
  - Summarizes completed sessions (LD-Agent pattern)
  - Tracks tasks via todo-for-ai (MCP)
  - Generates morning/evening briefings
  - Stores everything in Mem0 (persistent memory)

Architecture stolen from:
  - Mem0 (memory layer)
  - Milpa Agent (event-sourced session management)
  - LD-Agent (summarize → extract → generate pipeline)
  - Daily Briefing Agent (cron-based summaries)
  - todo-for-ai (task tracking)
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any

import yaml

# Allow running as `python main.py` from the project dir AND as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory import MemoryStore
from event_listener import EventListener
from ledger import SessionLedger
from opencode_client import OpenCodeClient, opencode_client_from_config
from summarizer import SessionSummarizer
from tasks import TaskTracker
from briefing import BriefingGenerator
from scheduler import SecretaryScheduler
from project_manager import ProjectOverseer, ProjectOverview

logger = logging.getLogger("secretary")


def load_config(path: str | Path = "~/.config/secretary/config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        # Fall back to project-local config.yaml (dev / first run)
        local = Path(__file__).resolve().parent / "config.yaml"
        if local.exists():
            config_path = local
        else:
            logger.error("Config not found: %s", config_path)
            sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    # Allow env var override for Vikunja token
    import os
    vikunja_token = os.environ.get("VIKUNJA_TOKEN")
    if vikunja_token:
        cfg.setdefault("projects", {}).setdefault("vikunja", {})["token"] = vikunja_token
    return cfg


def setup_logging(config: dict[str, Any]) -> None:
    """Configure logging from config (file handler is best-effort, rotated)."""
    from logging.handlers import RotatingFileHandler

    log_cfg = config.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    log_file = log_cfg.get("file")
    max_bytes = int(log_cfg.get("max_bytes_mb", 5)) * 1024 * 1024
    backups = int(log_cfg.get("backup_count", 3))

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(
                RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backups)
            )
        except OSError as exc:
            logging.getLogger("secretary").warning(
                "Cannot open log file %s (%s), console only", log_file, exc
            )

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


class SecretaryDaemon:
    """The main secretary daemon orchestrator."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.opencode = opencode_client_from_config(config)
        self.memory = MemoryStore(config)
        self.summarizer = SessionSummarizer(config)
        self.task_tracker = TaskTracker(config)
        self.briefing = BriefingGenerator(config, self.memory)
        self.scheduler = SecretaryScheduler(config)
        self.project_manager = ProjectOverseer(config)
        sum_cfg = config.get("summarizer", {})
        self.idle_debounce = int(sum_cfg.get("idle_debounce_seconds", 300))
        self.min_messages = int(sum_cfg.get("min_messages", 2))
        self.max_transcript_chars = int(sum_cfg.get("max_transcript_chars", 20000))
        self.ledger = SessionLedger(
            config.get("ledger_file", "~/.config/secretary/processed.json")
        )
        self._pending: dict[str, asyncio.Task] = {}
        self._stopped = False

        self.event_listener = EventListener(
            base_url=self.opencode.base_url,
            token=self.opencode.token,
            username=self.opencode.username,
            password=self.opencode.password,
            on_session_end=self._on_session_end,
            on_session_start=self._on_session_start,
        )

        self._setup_scheduler()

    def _setup_scheduler(self) -> None:
        """Register all cron jobs."""
        self.scheduler.register("stale_task_check", self._job_stale_task_check)
        self.scheduler.register("morning_briefing", self._job_morning_briefing)
        self.scheduler.register("evening_summary", self._job_evening_summary)
        self.scheduler.register("weekly_review", self._job_weekly_review)

    async def start(self) -> None:
        """Start all components."""
        logger.info("Secretary daemon starting...")

        # Ensure todo-for-ai project exists
        try:
            await self.task_tracker.ensure_project()
        except Exception:
            logger.warning("Could not connect to todo-for-ai, task tracking disabled")

        # Start scheduler
        self.scheduler.start()

        # Initial project registry refresh (best-effort)
        try:
            self.project_manager.refresh()
        except Exception:
            logger.debug("Initial project refresh failed", exc_info=True)

        # Start event listener (blocks until stopped)
        logger.info("Secretary daemon ready. Listening for events...")
        await self.event_listener.start()

    async def stop(self) -> None:
        """Gracefully stop all components (idempotent)."""
        if self._stopped:
            return
        self._stopped = True
        logger.info("Secretary daemon stopping...")
        for task in self._pending.values():
            task.cancel()
        self._pending.clear()
        self.scheduler.stop()
        await self.event_listener.stop()
        try:
            self.memory.close()
        except Exception:
            logger.debug("Memory close failed", exc_info=True)
        logger.info("Secretary daemon stopped.")

    # ── Event handlers ──────────────────────────────────────────────

    @staticmethod
    def _session_id(data: dict[str, Any]) -> str:
        for key in ("sessionID", "sessionId", "id"):
            val = data.get(key)
            if isinstance(val, str) and val:
                return val
        # session.status nests it
        inner = data.get("session", {})
        if isinstance(inner, dict):
            for key in ("sessionID", "id"):
                if isinstance(inner.get(key), str):
                    return inner[key]
        return "unknown"

    async def _on_session_end(self, data: dict[str, Any]) -> None:
        """Handle session.idle — debounce, then process exactly once."""
        session_id = self._session_id(data)
        if session_id == "unknown":
            return
        if session_id == self.opencode._secretary_session_id:
            # Ignore our own scratch session to avoid self-trigger loops
            return
        if self.ledger.is_processed(session_id):
            logger.debug("Session %s already processed, skipping", session_id)
            return
        # Restart the debounce window on every idle event
        old = self._pending.pop(session_id, None)
        if old is not None:
            old.cancel()
        self._pending[session_id] = asyncio.create_task(
            self._process_after_debounce(session_id, dict(data))
        )
        logger.debug("Debouncing session %s (%ds)", session_id, self.idle_debounce)

    async def _process_after_debounce(
        self, session_id: str, event_data: dict[str, Any], retries: int = 3
    ) -> None:
        """Wait out the debounce window, then summarize/store exactly once."""
        try:
            await asyncio.sleep(self.idle_debounce)
            # Still busy? Give it more time (bounded retries).
            for _ in range(retries):
                try:
                    if not await self.opencode.is_session_busy(session_id):
                        break
                except Exception:
                    break  # status endpoint down — process anyway
                logger.debug("Session %s busy, waiting another window", session_id)
                await asyncio.sleep(self.idle_debounce)

            if self.ledger.is_processed(session_id):
                return

            # Cost guard: skip trivial sessions (best-effort count)
            try:
                messages = await self.opencode.list_messages(session_id, limit=5)
                if len(messages) < self.min_messages:
                    logger.info(
                        "Session %s too short (%d msgs), skipping",
                        session_id, len(messages),
                    )
                    self.ledger.mark_processed(session_id)
                    return
            except Exception:
                logger.debug("Could not count messages for %s", session_id, exc_info=True)
                messages = []

            # Enrich the event with the transcript tail (best-effort)
            session_data: dict[str, Any] = dict(event_data)
            try:
                transcript = await self.opencode.get_transcript_text(
                    session_id, max_chars=self.max_transcript_chars
                )
                if transcript:
                    session_data["transcript"] = transcript
            except Exception:
                logger.debug("Could not fetch transcript for %s", session_id, exc_info=True)

            # Full pipeline: summarize → extract → store
            result = await self.summarizer.process_session(session_data)

            summary = result.get("summary", {})
            if not isinstance(summary, dict):
                summary = {"summary": str(summary)}
            summary_text = summary.get("summary") or json.dumps(summary, default=str)
            self.memory.add(
                f"[session:{session_id}] {summary_text}",
                metadata={"type": "session_summary", "session_id": session_id},
            )

            for fact in result.get("facts", []):
                if isinstance(fact, str) and fact.strip() and not self._fact_known(fact):
                    self.memory.add(
                        fact,
                        metadata={"type": "extracted_fact", "session_id": session_id},
                    )

            action_items = summary.get("action_items", []) or []
            action_items = [a for a in action_items if isinstance(a, str) and a.strip()]
            if action_items:
                existing = await self.task_tracker.list_titles()
                fresh = [a for a in action_items if a.casefold().strip() not in existing]
                if fresh:
                    await self.task_tracker.add_tasks_from_session(
                        fresh,
                        tags=["auto-extracted", f"session:{session_id}"],
                    )
                    logger.info("Created %d tasks from session %s", len(fresh), session_id)
                if len(fresh) != len(action_items):
                    logger.info(
                        "Skipped %d duplicate tasks for session %s",
                        len(action_items) - len(fresh), session_id,
                    )

            self.ledger.mark_processed(session_id)
            logger.info("Session %s processed", session_id)

            # Run overseer: detect problems and trigger actions
            try:
                actions = await self.project_manager.check_and_act(self.opencode)
                if actions:
                    logger.info(
                        "Overseer triggered %d action(s): %s",
                        len(actions),
                        "; ".join(f"{a.project}: {a.action}" for a in actions),
                    )
            except Exception:
                logger.debug("Overseer check_and_act failed", exc_info=True)

        except asyncio.CancelledError:
            logger.debug("Debounced processing cancelled for %s", session_id)
            raise
        except Exception:
            # Leave unprocessed: a later idle event or --backfill will retry
            logger.exception("Failed to process session %s", session_id)
        finally:
            self._pending.pop(session_id, None)

    def _fact_known(self, fact: str) -> bool:
        """Search-before-add: skip facts already in memory (best-effort)."""
        try:
            for hit in self.memory.search(fact, top_k=3):
                if not isinstance(hit, dict):
                    continue
                if hit.get("memory", "").strip().casefold() == fact.strip().casefold():
                    return True
                if float(hit.get("score", 0) or 0) >= 0.95:
                    return True
        except Exception:
            logger.debug("Fact dedupe search failed", exc_info=True)
        return False

    async def _on_session_start(self, data: dict[str, Any]) -> None:
        """Handle session.created event."""
        session_id = self._session_id(data)
        logger.info("Session started: %s", session_id)
        # Could log to memory, but sessions start frequently — keep it quiet

    # ── Scheduled jobs ──────────────────────────────────────────────

    async def _job_stale_task_check(self) -> None:
        """Check for stale tasks and log warnings."""
        try:
            stale = await self.task_tracker.get_stale_tasks(hours=24)
            if stale:
                logger.warning("Found %d stale tasks (>24h old)", len(stale))
                for task in stale[:5]:
                    logger.warning(
                        "  Stale: [%s] %s (%.1fh old)",
                        task.get("id"),
                        task.get("title"),
                        task.get("age_hours", 0),
                    )
                # Store in memory for briefing context
                self.memory.add(
                    f"Stale task alert: {len(stale)} tasks older than 24h. "
                    f"Oldest: {stale[0].get('title', 'unknown')} ({stale[0].get('age_hours', 0)}h)",
                    metadata={"type": "stale_task_alert"},
                )
        except Exception:
            logger.exception("Stale task check failed")

    async def _job_morning_briefing(self) -> None:
        """Generate, store, and deliver morning briefing."""
        try:
            briefing = await self.briefing.morning_briefing()
            await self.briefing.store_briefing(briefing, "morning")
            await self.briefing.deliver(briefing, "morning")
            logger.info("Morning briefing complete")
        except Exception:
            logger.exception("Morning briefing failed")

    async def _job_evening_summary(self) -> None:
        """Generate, store, and deliver evening summary."""
        try:
            summary = await self.briefing.evening_summary()
            await self.briefing.store_briefing(summary, "evening")
            await self.briefing.deliver(summary, "evening")
            logger.info("Evening summary complete")
        except Exception:
            logger.exception("Evening summary failed")

    async def _job_weekly_review(self) -> None:
        """Generate, store, and deliver weekly review."""
        try:
            review = await self.briefing.weekly_review()
            await self.briefing.store_briefing(review, "weekly")
            await self.briefing.deliver(review, "weekly")
            logger.info("Weekly review complete")
        except Exception:
            logger.exception("Weekly review failed")

    async def backfill(self, limit: int = 20) -> int:
        """Process up to `limit` recent unprocessed sessions. Returns count."""
        try:
            sessions = await self.opencode.list_sessions()
        except Exception:
            logger.error("Backfill failed: cannot reach opencode", exc_info=True)
            return 0
        done = 0
        for s in sessions[:limit]:
            if not isinstance(s, dict):
                continue
            sid = self._session_id(s)
            if sid == "unknown" or sid == self.opencode._secretary_session_id:
                continue
            if self.ledger.is_processed(sid):
                continue
            logger.info("Backfilling session %s", sid)
            await self._process_after_debounce(sid, s)
            if self.ledger.is_processed(sid):
                done += 1
        return done


# ── CLI helpers ─────────────────────────────────────────────────────

def check_connections(config: dict[str, Any]) -> int:
    """Probe all dependencies, print a report, return exit code."""
    import httpx

    ok = True

    def report(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"[{'OK' if good else 'FAIL'}] {name} {detail}")
        if not good:
            ok = False

    async def _probe() -> None:
        client_oc = opencode_client_from_config(config)
        try:
            health = await client_oc.health()
            report("opencode", True, json.dumps(health, default=str)[:80])
        except Exception as exc:
            report("opencode", False, str(exc)[:100])

        mem0_cfg = config.get("mem0", {})
        qh, qp = mem0_cfg.get("qdrant_host", "localhost"), mem0_cfg.get("qdrant_port", 6333)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"http://{qh}:{qp}/healthz")
                r.raise_for_status()
                rc = await client.get(
                    f"http://{qh}:{qp}/collections/{mem0_cfg.get('collection_name', 'secretary_memory')}"
                )
                detail = "collection exists" if rc.status_code == 200 else "collection missing (auto-created on first use)"
                report("qdrant", True, detail)
        except Exception as exc:
            report("qdrant", False, str(exc)[:100])

        try:
            tracker = TaskTracker(config)
            pid = await tracker.ensure_project()
            report("tasks", True, f"mode={'server' if pid != 'local' else 'local-fallback'}")
        except Exception as exc:
            report("tasks", False, str(exc)[:100])

        vik_cfg = config.get("projects", {}).get("vikunja", {})
        if vik_cfg.get("token"):
            try:
                async with httpx.AsyncClient(timeout=vik_cfg.get("timeout_seconds", 15)) as vc:
                    vr = await vc.get(
                        f"{vik_cfg['url']}/projects",
                        headers={"Authorization": f"Bearer {vik_cfg['token']}"},
                    )
                    vr.raise_for_status()
                    projects = vr.json()
                    report("vikunja", True, f"{len(projects)} projects")
            except Exception as exc:
                report("vikunja", False, str(exc)[:100])
        else:
            report("vikunja", False, "no token configured")

    asyncio.run(_probe())
    return 0 if ok else 1


# ── Entry point ─────────────────────────────────────────────────────

async def _run_daemon(config: dict[str, Any]) -> None:
    daemon = SecretaryDaemon(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig, functools.partial(asyncio.create_task, daemon.stop())
            )
        except NotImplementedError:
            # Windows / non-main-thread fallback
            signal.signal(sig, lambda *_: asyncio.create_task(daemon.stop()))

    await daemon.start()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Secretary daemon")
    parser.add_argument("--check", action="store_true", help="probe dependencies and exit")
    parser.add_argument("--backfill", type=int, metavar="N", default=0,
                        help="process up to N recent unprocessed sessions and exit")
    parser.add_argument("--no-debounce", action="store_true",
                        help="with --backfill: skip the idle debounce wait")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)

    if args.check:
        sys.exit(check_connections(config))

    if args.backfill:
        async def _backfill() -> None:
            daemon = SecretaryDaemon(config)
            if args.no_debounce:
                daemon.idle_debounce = 0
            n = await daemon.backfill(args.backfill)
            print(f"Backfilled {n} session(s)")
            try:
                daemon.memory.close()
            except Exception:
                pass

        asyncio.run(_backfill())
        return

    asyncio.run(_run_daemon(config))


if __name__ == "__main__":
    main()
