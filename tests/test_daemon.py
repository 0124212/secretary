"""Secretary daemon regression tests (stdlib unittest, no extra deps).

Run: .venv/bin/python -m unittest discover -s tests -v
Covers the bugs fixed during review: event routing, JSON parsing,
mem0 call shapes, task fallback, ledger, scheduler, transcript caps.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestEventDispatch(unittest.TestCase):
    def test_routing(self):
        from event_listener import EventListener

        calls: list = []

        async def on_end(d): calls.append(("end", d))

        async def on_start(d): calls.append(("start", d))

        el = EventListener("http://x", on_session_end=on_end, on_session_start=on_start)

        async def run():
            await el._dispatch("session.idle", '{"sessionID": "a"}')
            await el._dispatch("session.completed", '{"id": "b"}')
            await el._dispatch("session.status", '{"sessionID": "c", "status": "idle"}')
            await el._dispatch("session.status", '{"sessionID": "d", "status": "busy"}')
            await el._dispatch("session.created", '{"sessionID": "e"}')
            await el._dispatch("message.updated", '{"x": 1}')

        asyncio.run(run())
        self.assertEqual([c[0] for c in calls], ["end", "end", "end", "start"])


class TestJsonParsing(unittest.TestCase):
    def test_variants(self):
        from summarizer import SessionSummarizer as SS

        self.assertEqual(SS._parse_json('{"a": 1}'), {"a": 1})
        self.assertEqual(SS._parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(SS._parse_json('here:\n["x", "y"]'), ["x", "y"])
        self.assertEqual(SS._parse_json('pre ```json\n{"b": 2}\n``` post'), {"b": 2})
        self.assertIn("raw", SS._parse_json("not json at all {{{"))


class TestOpencodeClient(unittest.TestCase):
    def test_extract_text(self):
        from opencode_client import OpenCodeClient as OC

        self.assertEqual(OC.extract_text({"parts": [{"text": "hi"}]}), "hi")
        self.assertEqual(OC.extract_text({"text": "direct"}), "direct")
        self.assertEqual(OC.extract_text("plain"), "plain")

    def test_transcript_truncation(self):
        from opencode_client import OpenCodeClient as OC

        client = OC("http://x")
        msgs = [{"info": {"role": "user"}, "parts": [{"text": "A" * 1000}]}]

        async def run():
            with patch.object(client, "list_messages", new=AsyncMock(return_value=msgs)):
                full = await client.get_transcript_text("s")
                self.assertIn("A" * 100, full)
                capped = await client.get_transcript_text("s", max_chars=100)
                self.assertLessEqual(len(capped), 100 + 60)
                self.assertIn("truncated", capped)

        asyncio.run(run())


class TestMemoryShapes(unittest.TestCase):
    def test_calls(self):
        import tempfile
        from memory import MemoryStore

        with tempfile.TemporaryDirectory() as tmp:
            store_path = str(Path(tmp) / "memory.json")
            ms = MemoryStore({"mem0": {"store_path": store_path}, "user_id": "u"})
            # Override megamemory connection to avoid needing the real DB in tests
            ms._mm_conn = None

            ms.add("test memory", metadata={"type": "test"})
            results = ms.search("test")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["memory"], "test memory")
            self.assertEqual(results[0]["user_id"], "u")
            all_mems = ms.get_all(limit=50)
            self.assertEqual(len(all_mems), 1)
            ms.update(results[0]["id"], "updated memory")
            updated = ms.get(results[0]["id"])
            self.assertEqual(updated["memory"], "updated memory")


class TestTaskFallback(unittest.TestCase):
    def test_offline_roundtrip(self):
        from tasks import TaskTracker

        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"todo_for_ai": {
                "base_url": "http://127.0.0.1:9/nope",
                "fallback_file": str(Path(tmp) / "tasks.json"),
            }}
            tt = TaskTracker(cfg)

            async def run():
                self.assertEqual(await tt.ensure_project(), "local")
                t = await tt.create_task("Do thing", tags=["a"])
                self.assertTrue(t["id"].startswith("local-"))
                self.assertEqual(len(await tt.list_tasks(status_filter=["todo"])), 1)
                self.assertEqual(await tt.list_titles(), {"do thing"})
                await tt.update_task(t["id"], status="done")
                self.assertEqual(await tt.list_tasks(status_filter=["todo"]), [])

            asyncio.run(run())


class TestLedger(unittest.TestCase):
    def test_mark_and_prune(self):
        from ledger import SessionLedger

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "proc.json"
            ledger = SessionLedger(p, retention_days=30)
            self.assertFalse(ledger.is_processed("s1"))
            ledger.mark_processed("s1")
            self.assertTrue(ledger.is_processed("s1"))
            # Persists across instances
            self.assertTrue(SessionLedger(p).is_processed("s1"))
            # Old entries pruned
            data = json.loads(p.read_text())
            data["old"] = {"state": "processed", "ts": "2000-01-01T00:00:00+00:00"}
            p.write_text(json.dumps(data))
            ledger2 = SessionLedger(p, retention_days=30)
            self.assertEqual(ledger2.prune(), 1)
            self.assertFalse(ledger2.is_processed("old"))


class TestScheduler(unittest.TestCase):
    def test_crons_and_timezone(self):
        from scheduler import SecretaryScheduler

        cfg = {"scheduler": {
            "timezone": "UTC",
            "stale_task_check": "0 * * * *",
            "morning_briefing": "0 9 * * *",
            "evening_summary": "0 18 * * *",
            "weekly_review": "0 10 * * 0",
        }}
        sched = SecretaryScheduler(cfg)

        async def noop(): pass

        for name in ("stale_task_check", "morning_briefing",
                     "evening_summary", "weekly_review"):
            sched.register(name, noop)

        async def run():
            sched.start()
            for name in ("stale_task_check", "morning_briefing",
                         "evening_summary", "weekly_review"):
                self.assertIsNotNone(sched.get_next_run(name))
            sched.stop()

        asyncio.run(run())

    def test_bad_timezone(self):
        from scheduler import SecretaryScheduler

        with self.assertRaises(ValueError):
            SecretaryScheduler({"scheduler": {"timezone": "Not/AZone"}})

    def test_bad_cron(self):
        from scheduler import SecretaryScheduler

        with self.assertRaises(ValueError):
            SecretaryScheduler._parse_cron("not a cron")


class TestMainHelpers(unittest.TestCase):
    def test_session_id(self):
        import main

        sid = main.SecretaryDaemon._session_id
        self.assertEqual(sid({"sessionID": "a"}), "a")
        self.assertEqual(sid({"id": "b"}), "b")
        self.assertEqual(sid({"session": {"id": "c"}}), "c")
        self.assertEqual(sid({}), "unknown")

    def test_config_fallback(self):
        import main

        cfg = main.load_config("/nonexistent/path.yaml")
        self.assertIn("opencode", cfg)
        self.assertIn("idle_debounce_seconds", cfg.get("summarizer", {}))


if __name__ == "__main__":
    unittest.main()
