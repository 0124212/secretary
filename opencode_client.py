"""OpenCode REST client — real endpoints, no guessing.

Verified against opencode serve docs (opencode.ai/docs/server):
  - GET  /global/health          health check
  - GET  /global/event           SSE stream (fallback: /event)
  - GET  /session                list sessions
  - POST /session                create session {title?}
  - GET  /session/:id            session details
  - GET  /session/:id/message    list messages (transcript)
  - POST /session/:id/message    send prompt, waits for response
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def build_auth_headers(
    token: str = "", username: str = "", password: str = ""
) -> dict[str, str]:
    """Authorization header for Bearer or Basic auth (Basic wins if set)."""
    if username:
        raw = f"{username}:{password}".encode()
        return {"Authorization": "Basic " + base64.b64encode(raw).decode()}
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def opencode_client_from_config(config: dict[str, Any]) -> "OpenCodeClient":
    """Build a client from config, resolving secrets from env when asked.

    opencode:
      url: http://localhost:4097
      token: "<literal>"            # Bearer, or
      token_env: "OPENCODE_TOKEN"   # Bearer from env, or
      username: "ak"                # Basic auth (needs password/password_env)
      password_env: "OPENCODE_SERVER_PASSWORD"
    """
    oc = config.get("opencode", {})
    token = oc.get("token", "") or os.environ.get(oc.get("token_env", ""), "")
    username = oc.get("username", "")
    password = oc.get("password", "") or os.environ.get(oc.get("password_env", ""), "")
    return OpenCodeClient(
        base_url=oc.get("url", "http://localhost:4097"),
        token=token,
        username=username,
        password=password,
    )


class OpenCodeClient:
    """Thin async client for opencode serve."""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        username: str = "",
        password: str = "",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.username = username
        self.password = password
        self.timeout = timeout
        self._secretary_session_id: str | None = None

    @property
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        h.update(build_auth_headers(self.token, self.username, self.password))
        return h

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def health(self) -> dict[str, Any]:
        """GET /global/health — raises if server is down."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(self._url("/global/health"), headers=self._headers)
            resp.raise_for_status()
            return resp.json()

    async def list_sessions(self) -> list[dict[str, Any]]:
        """GET /session — list all sessions."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(self._url("/session"), headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("sessions", [])

    async def create_session(self, title: str = "") -> dict[str, Any]:
        """POST /session — create a new session."""
        payload: dict[str, Any] = {}
        if title:
            payload["title"] = title
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self._url("/session"), json=payload, headers=self._headers
            )
            resp.raise_for_status()
            return resp.json()

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """GET /session/:id — session details."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                self._url(f"/session/{session_id}"), headers=self._headers
            )
            resp.raise_for_status()
            return resp.json()

    async def session_status(self) -> dict[str, Any]:
        """GET /session/status — busy/idle per session. {} when unknown."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(self._url("/session/status"), headers=self._headers)
                resp.raise_for_status()
                data = resp.json()
                return data if isinstance(data, dict) else {}
        except Exception:
            logger.debug("session/status unavailable", exc_info=True)
            return {}

    async def is_session_busy(self, session_id: str) -> bool:
        """True if the session is currently busy (debounce should wait)."""
        statuses = await self.session_status()
        entry = statuses.get(session_id, {})
        if isinstance(entry, dict):
            return entry.get("status", entry.get("type", "")) == "busy"
        return str(entry) == "busy"

    async def list_messages(self, session_id: str, limit: int = 100) -> list[dict]:
        """GET /session/:id/message — full transcript."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                self._url(f"/session/{session_id}/message"),
                params={"limit": limit},
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            return data.get("messages", data.get("data", []))

    async def send_message(
        self,
        session_id: str,
        text: str,
        model: str = "",
    ) -> dict[str, Any]:
        """POST /session/:id/message — send prompt, wait for response.

        Body per docs: { parts } where parts is a list of message parts.
        Returns { info, parts } for the assistant reply.
        """
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        payload: dict[str, Any] = {"parts": parts}
        if model:
            payload["model"] = model
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._url(f"/session/{session_id}/message"),
                json=payload,
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def ensure_secretary_session(self, title: str = "secretary") -> str:
        """Find or create the daemon's own scratch session for LLM prompts."""
        if self._secretary_session_id:
            return self._secretary_session_id
        sessions = await self.list_sessions()
        for s in sessions:
            if isinstance(s, dict) and s.get("title") == title:
                self._secretary_session_id = s.get("id", s.get("sessionID", ""))
                if self._secretary_session_id:
                    return self._secretary_session_id
        created = await self.create_session(title=title)
        self._secretary_session_id = created.get("id", created.get("sessionID", ""))
        logger.info("Created secretary session: %s", self._secretary_session_id)
        return self._secretary_session_id

    async def prompt(self, text: str, model: str = "") -> str:
        """Send a prompt via the secretary session, return assistant text."""
        session_id = await self.ensure_secretary_session()
        result = await self.send_message(session_id, text, model=model)
        return self.extract_text(result)

    @staticmethod
    def extract_text(result: Any) -> str:
        """Pull human-readable text out of a message response."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            parts = result.get("parts", [])
            texts = []
            for p in parts if isinstance(parts, list) else []:
                if isinstance(p, dict):
                    if isinstance(p.get("text"), str):
                        texts.append(p["text"])
                    # nested part payloads
                    inner = p.get("part", {})
                    if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                        texts.append(inner["text"])
            if texts:
                return "\n".join(texts)
            for key in ("text", "response", "content"):
                if isinstance(result.get(key), str):
                    return result[key]
            return json.dumps(result)
        return str(result)

    async def get_transcript_text(
        self, session_id: str, limit: int = 100, max_chars: int = 0
    ) -> str:
        """Fetch a session transcript and flatten it to plain text.

        max_chars > 0 keeps the tail (most recent) with a truncation note.
        """
        messages = await self.list_messages(session_id, limit=limit)
        lines: list[str] = []
        for m in messages:
            if not isinstance(m, dict):
                lines.append(str(m))
                continue
            info = m.get("info", m)
            role = info.get("role", m.get("role", "?")) if isinstance(info, dict) else "?"
            parts = m.get("parts", [])
            text_bits = []
            for p in parts if isinstance(parts, list) else []:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    text_bits.append(p["text"])
            if text_bits:
                lines.append(f"[{role}] " + "\n".join(text_bits))
        text = "\n\n".join(lines)
        if max_chars > 0 and len(text) > max_chars:
            text = "[...earlier transcript truncated...]\n\n" + text[-max_chars:]
        return text

    async def delete_session(self, session_id: str) -> bool:
        """DELETE /session/:id — remove a session. Returns True on success."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.delete(
                    self._url(f"/session/{session_id}"), headers=self._headers
                )
                resp.raise_for_status()
                return True
        except Exception:
            logger.debug("Failed to delete session %s", session_id, exc_info=True)
            return False

    async def wait_for_healthy(self, retries: int = 5, delay: float = 2.0) -> bool:
        """Poll health endpoint; True if server answers within retries."""
        for _ in range(retries):
            try:
                await self.health()
                return True
            except Exception:
                await asyncio.sleep(delay)
        return False
