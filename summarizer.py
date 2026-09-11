"""Summarizer — session → summary → extract → store.

Three-stage pipeline (stolen from LD-Agent pattern):
  1. Summarize: compress session into structured summary
  2. Extract: pull out action items, facts, decisions
  3. Store: persist to Mem0 memory layer (done by caller in main.py)

LLM inference goes through OpencodeClient (POST /session + 
POST /session/:id/message), not a /prompt endpoint (which doesn't exist).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from opencode_client import OpenCodeClient, opencode_client_from_config

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """You are a session secretary. Summarize this coding session.

Output JSON with these fields:
- "title": one-line summary (max 10 words)
- "summary": 2-4 sentence overview of what happened
- "decisions": list of decisions made (strings)
- "action_items": list of TODO items extracted (strings)
- "files_changed": list of files that were modified (paths)
- "next_steps": suggested follow-up actions (strings)

Session data:
{session_data}

Respond with ONLY valid JSON."""

EXTRACT_FACTS_PROMPT = """Extract key facts from this session summary that should be remembered long-term.
Focus on:
- User preferences discovered
- Project architecture decisions
- API keys, config values, endpoints
- Patterns that worked or failed
- Corrections the user made

Output JSON array of strings, each a single fact.
Summary:
{summary}

Respond with ONLY valid JSON array."""


class SessionSummarizer:
    """Summarizes completed sessions via opencode prompts."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.client = opencode_client_from_config(config)
        self.model = config.get("summarizer", {}).get("model", "")

    async def summarize_session(self, session_data: dict[str, Any]) -> dict[str, Any]:
        """Stage 1: Summarize a completed session."""
        prompt = SUMMARIZE_PROMPT.format(
            session_data=json.dumps(session_data, indent=2, default=str)
        )
        raw = await self.client.prompt(prompt, model=self.model)
        parsed = self._parse_json(raw)
        if isinstance(parsed, dict):
            return parsed
        logger.warning("Summarizer returned non-dict, wrapping: %s", str(parsed)[:200])
        return {"title": "session summary", "summary": str(parsed),
                "decisions": [], "action_items": [],
                "files_changed": [], "next_steps": []}

    async def extract_facts(self, summary: str) -> list[str]:
        """Stage 2: Extract long-term facts from summary."""
        prompt = EXTRACT_FACTS_PROMPT.format(summary=summary)
        raw = await self.client.prompt(prompt, model=self.model)
        result = self._parse_json(raw)
        return result if isinstance(result, list) else []

    async def process_session(self, session_data: dict[str, Any]) -> dict[str, Any]:
        """Full pipeline: summarize → extract → return structured result."""
        # Stage 1: Summarize
        summary = await self.summarize_session(session_data)
        logger.info("Session summarized: %s", summary.get("title", "unknown"))

        # Stage 2: Extract facts
        facts = await self.extract_facts(json.dumps(summary, default=str))
        logger.info("Extracted %d long-term facts", len(facts))

        return {
            "summary": summary,
            "facts": facts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Extract JSON from LLM response, handling markdown fences and preamble."""
        import re

        cleaned = text.strip()
        # Prefer the first ``` fenced block if present
        fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        elif cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        # Fall back to the first {...} or [...] span
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        for start, end in (("{", "}"), ("[", "]")):
            s, e = cleaned.find(start), cleaned.rfind(end)
            if 0 <= s < e:
                try:
                    return json.loads(cleaned[s:e + 1])
                except json.JSONDecodeError:
                    continue
        logger.warning("Failed to parse JSON from: %s", cleaned[:200])
        return {"raw": text.strip()}
