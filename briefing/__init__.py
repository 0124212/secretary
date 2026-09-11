"""Briefing generator — morning and evening summaries.

Queries Mem0 for recent memories and generates structured briefings.

Stolen from: Daily Briefing Agent (gokborayilmaz/daily-briefing-agent)
Pattern: background thread pulls data → generates summary → stores output
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from memory import MemoryStore
from opencode_client import OpenCodeClient, opencode_client_from_config

logger = logging.getLogger(__name__)

MORNING_PROMPT = """You are a morning briefing secretary. Generate a concise morning briefing based on recent session data.

Include:
1. **Yesterday's Highlights** — key things accomplished
2. **Open Tasks** — items still pending
3. **Today's Suggested Focus** — what to work on next
4. **Anything Stale** — tasks or items that need attention

Data:
{data}

Format as clean markdown. Be concise — aim for a quick 30-second read."""

EVENING_PROMPT = """You are an evening summary secretary. Generate an end-of-day wrap-up.

Include:
1. **What Was Done** — accomplishments today
2. **What's Left** — remaining tasks
3. **Patterns** — recurring issues or wins
4. **Tomorrow's Prep** — things to pick up tomorrow

Data:
{data}

Format as clean markdown. Be concise."""


class BriefingGenerator:
    """Generates morning and evening briefings from memory."""

    def __init__(self, config: dict[str, Any], memory: MemoryStore) -> None:
        self.memory = memory
        self.client = opencode_client_from_config(config)
        self.model = config.get("summarizer", {}).get("model", "")
        self.webhook_url = config.get("briefing", {}).get("webhook_url", "")
        self.store_in_memos = config.get("briefing", {}).get("store_in_memos", True)
        memos_cfg = config.get("memos", {})
        self.memos_url = str(memos_cfg.get("url", "http://localhost:5230")).rstrip("/")
        self.memos_token = memos_cfg.get("token", "")
        self.memos_visibility = config.get("briefing", {}).get("memos_visibility", "PRIVATE")

    async def morning_briefing(self) -> str:
        """Generate morning briefing from yesterday's memories."""
        logger.info("Generating morning briefing")

        # Gather recent memories
        memories = self.memory.get_all(page=1, page_size=50)
        facts = [m.get("memory", "") for m in memories]

        # Also get recent search results
        recent = self.memory.search("recent session work", top_k=10)

        data = {
            "recent_memories": [r.get("memory", "") for r in recent],
            "all_facts": facts,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

        prompt = MORNING_PROMPT.format(data=json.dumps(data, indent=2))
        briefing = await self._prompt(prompt)

        logger.info("Morning briefing generated (%d chars)", len(briefing))
        return briefing

    async def evening_summary(self) -> str:
        """Generate end-of-day summary."""
        logger.info("Generating evening summary")

        memories = self.memory.get_all(page=1, page_size=50)
        recent = self.memory.search("today session", top_k=10)

        data = {
            "recent_memories": [r.get("memory", "") for r in recent],
            "all_facts": [m.get("memory", "") for m in memories],
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

        prompt = EVENING_PROMPT.format(data=json.dumps(data, indent=2))
        summary = await self._prompt(prompt)

        logger.info("Evening summary generated (%d chars)", len(summary))
        return summary

    async def weekly_review(self) -> str:
        """Generate weekly review from accumulated data."""
        logger.info("Generating weekly review")

        # Pull a lot of memories for weekly context
        memories = self.memory.get_all(page=1, page_size=200)
        facts = [m.get("memory", "") for m in memories]

        data = {
            "all_facts": facts,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "period": "past week",
        }

        prompt = (
            "You are a weekly review secretary. Generate a concise weekly review.\n"
            "Include: accomplishments, recurring patterns, areas needing attention, "
            "and priorities for next week.\n\n"
            f"Data:\n{json.dumps(data, indent=2)}\n\n"
            "Format as clean markdown."
        )
        review = await self._prompt(prompt)

        logger.info("Weekly review generated (%d chars)", len(review))
        return review

    async def store_briefing(self, content: str, kind: str) -> None:
        """Store briefing as a memory (capped at 2000 chars)."""
        self.memory.add(
            f"[{kind}] {content[:2000]}",
            metadata={"type": "briefing", "kind": kind},
        )
        logger.info("Stored %s briefing in Mem0", kind)

    async def _prompt(self, prompt: str) -> str:
        """Send a prompt to opencode for inference."""
        return await self.client.prompt(prompt, model=self.model)

    async def deliver(self, content: str, kind: str) -> None:
        """Deliver a briefing: webhook POST and/or Memos note. Best-effort."""
        if self.webhook_url:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        self.webhook_url,
                        json={
                            "kind": kind,
                            "content": content,
                            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        },
                    )
                    resp.raise_for_status()
                logger.info("Delivered %s briefing to webhook", kind)
            except Exception:
                logger.warning("Webhook delivery failed for %s briefing", kind, exc_info=True)

        if self.store_in_memos:
            try:
                import httpx

                headers = {"Content-Type": "application/json"}
                if self.memos_token:
                    headers["Authorization"] = f"Bearer {self.memos_token}"
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{self.memos_url}/api/v1/memos",
                        json={"content": content, "visibility": self.memos_visibility},
                        headers=headers,
                    )
                    resp.raise_for_status()
                logger.info("Stored %s briefing in Memos", kind)
            except Exception:
                logger.warning("Memos delivery failed for %s briefing", kind, exc_info=True)
