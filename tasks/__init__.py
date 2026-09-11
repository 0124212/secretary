"""Task tracker — todo-for-ai client with local JSON fallback.

Manages action items, TODOs, and project tasks extracted from sessions.
Talks to the todo-for-ai REST API when reachable; otherwise stores
tasks in a local JSON file so the daemon keeps working offline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# How long to trust a failed server probe before retrying (seconds)
SERVER_RETRY_COOLDOWN = 60.0


class TaskTracker:
    """Client for todo-for-ai REST API with local-file fallback."""

    def __init__(self, config: dict[str, Any]) -> None:
        todo_cfg = config.get("todo_for_ai", {})
        self.base_url = todo_cfg.get("base_url", "http://localhost:50110/todo-for-ai/api/v1")
        self.api_token = todo_cfg.get("api_token", "")
        self.project_name = todo_cfg.get("project_name", "secretary")
        self.project_id: int | None = todo_cfg.get("project_id")
        self.fallback_file = Path(
            todo_cfg.get("fallback_file", "~/.config/secretary/tasks.json")
        ).expanduser()
        self._server_available: bool | None = None
        self._server_failed_at: float = 0.0

    @property
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_token:
            h["Authorization"] = f"Bearer {self.api_token}"
        return h

    async def ensure_project(self) -> int | str:
        """Get or create the secretary project. Returns project ID.

        Returns "local" when the server is unreachable (fallback mode).
        Never raises — falls back to the local JSON store.
        """
        if self.project_id:
            return self.project_id

        import time
        if (
            self._server_available is False
            and time.monotonic() - self._server_failed_at < SERVER_RETRY_COOLDOWN
        ):
            return "local"  # recent failure, skip re-probing

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/projects",
                    headers=self._headers,
                )
                if resp.status_code == 200:
                    for proj in resp.json().get("data", []):
                        if proj.get("name") == self.project_name:
                            self.project_id = proj["id"]
                            self._server_available = True
                            logger.info("Found project '%s' (id=%s)", self.project_name, self.project_id)
                            return self.project_id

                resp = await client.post(
                    f"{self.base_url}/projects",
                    json={"name": self.project_name, "description": "Auto-managed by secretary daemon"},
                    headers=self._headers,
                )
                resp.raise_for_status()
                self.project_id = resp.json()["id"]
                self._server_available = True
                logger.info("Created project '%s' (id=%s)", self.project_name, self.project_id)
                return self.project_id
        except Exception as exc:
            logger.warning("todo-for-ai unreachable (%s), using local task store", exc)
            import time
            self._server_available = False
            self._server_failed_at = time.monotonic()
            return "local"

    # ── local JSON fallback ─────────────────────────────────────

    def _load_local(self) -> list[dict]:
        if not self.fallback_file.exists():
            return []
        try:
            return json.loads(self.fallback_file.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt local task file, starting fresh: %s", self.fallback_file)
            return []

    def _save_local(self, tasks: list[dict]) -> None:
        self.fallback_file.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_file.write_text(json.dumps(tasks, indent=2))

    def _local_create(self, title: str, content: str, priority: str,
                      tags: list[str] | None) -> dict:
        tasks = self._load_local()
        task = {
            "id": f"local-{len(tasks) + 1}",
            "project": self.project_name,
            "title": title,
            "content": content,
            "priority": priority,
            "status": "todo",
            "tags": tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        tasks.append(task)
        self._save_local(tasks)
        logger.info("Created local task: %s (%s)", title, task["id"])
        return task

    def _local_list(self, status_filter: list[str] | None) -> list[dict]:
        tasks = self._load_local()
        if status_filter:
            tasks = [t for t in tasks if t.get("status") in status_filter]
        return tasks

    def list_local(self, status_filter: list[str] | None = None) -> list[dict]:
        """Synchronous local-only task list (used by project registry)."""
        return self._local_list(status_filter)

    def _local_update(self, task_id: int | str, payload: dict) -> dict:
        tasks = self._load_local()
        for t in tasks:
            if str(t.get("id")) == str(task_id):
                t.update(payload)
                self._save_local(tasks)
                return t
        raise KeyError(f"Local task not found: {task_id}")

    # ── public API (server first, local fallback) ───────────────────

    async def create_task(
        self,
        title: str,
        content: str = "",
        priority: str = "medium",
        due_date: str | None = None,
        tags: list[str] | None = None,
        is_ai_task: bool = True,
    ) -> dict:
        """Create a new task (falls back to local store offline)."""
        project_id = await self.ensure_project()
        if project_id == "local":
            return self._local_create(title, content, priority, tags)
        payload: dict[str, Any] = {
            "project_id": project_id,
            "title": title,
            "content": content,
            "priority": priority,
            "status": "todo",
            "is_ai_task": is_ai_task,
        }
        if due_date:
            payload["due_date"] = due_date
        if tags:
            payload["tags"] = tags

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/tasks",
                    json=payload,
                    headers=self._headers,
                )
                resp.raise_for_status()
                task = resp.json()
                logger.info("Created task: %s (id=%s)", title, task.get("id"))
                return task
        except Exception as exc:
            logger.warning("create_task via server failed (%s), using local store", exc)
            return self._local_create(title, content, priority, tags)

    async def list_tasks(
        self,
        status_filter: list[str] | None = None,
    ) -> list[dict]:
        """List tasks in the secretary project (local fallback offline)."""
        project_id = await self.ensure_project()
        if project_id == "local":
            return self._local_list(status_filter)
        params: dict[str, Any] = {"project_id": project_id}
        if status_filter:
            params["status"] = ",".join(status_filter)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{self.base_url}/tasks",
                    params=params,
                    headers=self._headers,
                )
                resp.raise_for_status()
                return resp.json().get("data", [])
        except Exception as exc:
            logger.warning("list_tasks via server failed (%s), using local store", exc)
            return self._local_list(status_filter)

    async def update_task(
        self,
        task_id: int,
        status: str | None = None,
        title: str | None = None,
        content: str | None = None,
        priority: str | None = None,
    ) -> dict:
        """Update an existing task."""
        payload: dict[str, Any] = {}
        if status is not None:
            payload["status"] = status
        if title is not None:
            payload["title"] = title
        if content is not None:
            payload["content"] = content
        if priority is not None:
            payload["priority"] = priority

        if isinstance(task_id, str) and task_id.startswith("local-"):
            return self._local_update(task_id, payload)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.patch(
                    f"{self.base_url}/tasks/{task_id}",
                    json=payload,
                    headers=self._headers,
                )
                resp.raise_for_status()
                logger.info("Updated task %s: %s", task_id, payload)
                return resp.json()
        except KeyError:
            raise
        except Exception as exc:
            logger.warning("update_task via server failed (%s), using local store", exc)
            return self._local_update(task_id, payload)

    async def list_titles(self) -> set[str]:
        """Normalized titles of all known tasks (for dedupe). Never raises."""
        try:
            tasks = await self.list_tasks()
        except Exception:
            logger.debug("list_titles failed", exc_info=True)
            return set()
        return {
            str(t.get("title", "")).casefold().strip()
            for t in tasks if t.get("title")
        }

    async def get_stale_tasks(self, hours: int = 24) -> list[dict]:
        """Find tasks that have been in 'todo' status for too long."""
        tasks = await self.list_tasks(status_filter=["todo", "in_progress"])
        stale = []
        now = datetime.now(timezone.utc)
        for task in tasks:
            created = task.get("created_at")
            if created:
                try:
                    created_dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                    if created_dt.tzinfo is None:
                        created_dt = created_dt.replace(tzinfo=timezone.utc)
                    age_hours = (now - created_dt).total_seconds() / 3600
                    if age_hours > hours:
                        task["age_hours"] = round(age_hours, 1)
                        stale.append(task)
                except (ValueError, TypeError):
                    pass
        return stale

    async def add_tasks_from_session(
        self,
        action_items: list[str],
        tags: list[str] | None = None,
    ) -> list[dict]:
        """Batch create tasks from extracted action items."""
        created = []
        for item in action_items:
            task = await self.create_task(
                title=item,
                tags=tags or ["auto-extracted"],
                priority="medium",
            )
            created.append(task)
        return created
