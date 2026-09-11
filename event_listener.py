"""Event listener — SSE stream from opencode serve.

Connects to the opencode global event stream and dispatches handlers
for session lifecycle events.

Verified endpoints (opencode.ai/docs/server):
  - GET /global/event (current), fallback GET /event (legacy)
Verified event types: server.connected, session.created, session.idle,
  session.status, message.updated. There is no session.completed event —
  session.idle (agent finished) is the end-of-session trigger.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine

import httpx

logger = logging.getLogger(__name__)

# Reconnect backoff
MIN_RECONNECT_DELAY = 1.0
MAX_RECONNECT_DELAY = 60.0


class EventListener:
    """Listens to opencode SSE events and dispatches to handlers."""

    def __init__(
        self,
        base_url: str = "http://localhost:4097",
        token: str = "",
        username: str = "",
        password: str = "",
        on_session_end: Callable[[dict], Coroutine[Any, Any, None]] | None = None,
        on_session_start: Callable[[dict], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.username = username
        self.password = password
        self.on_session_end = on_session_end
        self.on_session_start = on_session_start
        self._running = False
        self._client: httpx.AsyncClient | None = None

    @property
    def _headers(self) -> dict[str, str]:
        from opencode_client import build_auth_headers

        h = {"Accept": "text/event-stream"}
        h.update(build_auth_headers(self.token, self.username, self.password))
        return h

    async def start(self) -> None:
        """Start listening in a loop with reconnection."""
        self._running = True
        delay = MIN_RECONNECT_DELAY

        while self._running:
            try:
                await self._listen()
                delay = MIN_RECONNECT_DELAY  # reset on clean disconnect
            except httpx.ConnectError:
                logger.warning("Cannot connect to %s, retrying in %.1fs", self.base_url, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)
            except Exception:
                logger.exception("Event listener error, reconnecting in %.1fs", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)

    async def stop(self) -> None:
        """Gracefully stop."""
        self._running = False
        if self._client:
            await self._client.aclose()

    EVENT_URLS = ("/global/event", "/event")

    async def _listen(self) -> None:
        """Connect to SSE endpoint and process events."""
        last_error: Exception | None = None
        for path in self.EVENT_URLS:
            url = f"{self.base_url}{path}"
            try:
                await self._stream_url(url)
                return  # clean disconnect, no fallback needed
            except httpx.HTTPStatusError as exc:
                # 404 = wrong path, try next; anything else re-raises
                if exc.response.status_code == 404:
                    logger.info("Event endpoint %s not found, trying fallback", path)
                    last_error = exc
                    continue
                raise
        if last_error:
            raise last_error

    async def _stream_url(self, url: str) -> None:
        """Stream a single SSE URL until disconnect."""
        logger.info("Connecting to event stream: %s", url)

        async with httpx.AsyncClient(timeout=None) as client:
            self._client = client
            async with client.stream("GET", url, headers=self._headers) as response:
                response.raise_for_status()
                event_type = None
                data_lines: list[str] = []

                async for line in response.aiter_lines():
                    if not self._running:
                        break
                    line = line.strip()

                    if not line:
                        # Empty line = dispatch event
                        if event_type and data_lines:
                            raw_data = "\n".join(data_lines)
                            await self._dispatch(event_type, raw_data)
                        event_type = None
                        data_lines = []
                    elif line.startswith("event:"):
                        event_type = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:"):].strip())

    async def _dispatch(self, event_type: str, raw_data: str) -> None:
        """Route event to the appropriate handler."""
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            logger.debug("Non-JSON event %s: %s", event_type, raw_data[:200])
            data = {"raw": raw_data}

        logger.debug("Event: %s — %s", event_type, json.dumps(data, default=str)[:200])

        # session.idle = agent finished responding → end-of-session trigger.
        # session.completed kept for backward compat (older forks emit it).
        # session.status with idle status is another valid end signal.
        if event_type in ("session.idle", "session.completed"):
            if self.on_session_end:
                await self.on_session_end(data)
        elif event_type == "session.status":
            status = data.get("status", data.get("type", ""))
            if status == "idle" and self.on_session_end:
                await self.on_session_end(data)
            else:
                logger.debug("Session status event: %s", status)
        elif event_type == "session.created" and self.on_session_start:
            await self.on_session_start(data)
        elif event_type == "server.connected":
            logger.info("Connected to opencode server")
        else:
            logger.debug("Unhandled event type: %s", event_type)
