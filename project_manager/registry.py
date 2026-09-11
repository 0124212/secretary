"""Project registry backed by Vikunja as the single source of truth."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class VikunjaSignals:
    open_tasks: int = 0
    done_tasks: int = 0
    overdue_tasks: int = 0
    oldest_task_days: float | None = None
    latest_task_update_days: float | None = None
    labels: list[str] = field(default_factory=list)


@dataclass
class ProjectRecord:
    id: str
    name: str
    kind: str  # vikunja
    description: str
    vikunja_id: int
    is_archived: bool
    signals: VikunjaSignals = field(default_factory=VikunjaSignals)


class ProjectRegistry:
    """Fetches projects and their tasks directly from Vikunja."""

    def __init__(self, config: dict[str, Any]) -> None:
        proj_cfg = config.get("projects", {})
        vikunja_cfg = proj_cfg.get("vikunja", {})
        self._base_url = vikunja_cfg.get("url", "https://vikunja.junilab.xyz/api/v1").rstrip("/")
        self._token = vikunja_cfg.get("token", "")
        self._timeout = float(vikunja_cfg.get("timeout_seconds", 15))
        self._ignore_projects = {n.lower() for n in proj_cfg.get("ignore_projects", [])}
        self._max_projects = int(proj_cfg.get("max_projects", 50))

    def discover(self) -> list[ProjectRecord]:
        """Fetch all non-archived Vikunja projects, then fetch tasks per project."""
        projects = self._fetch_projects()
        for rec in projects:
            try:
                rec.signals = self._fetch_task_signals(rec.vikunja_id)
            except Exception:
                logger.debug("Failed to fetch tasks for project %s", rec.name, exc_info=True)
                rec.signals = VikunjaSignals()
        return projects[: self._max_projects]

    def get_project(self, vikunja_id: int) -> ProjectRecord | None:
        """Fetch a single project by Vikunja ID."""
        try:
            data = self._get(f"/projects/{vikunja_id}")
        except Exception:
            return None
        if not data or data.get("is_archived"):
            return None
        rec = ProjectRecord(
            id=f"vikunja:{data['id']}",
            name=data.get("title", f"project-{data['id']}"),
            kind="vikunja",
            description=data.get("description", ""),
            vikunja_id=data["id"],
            is_archived=data.get("is_archived", False),
        )
        try:
            rec.signals = self._fetch_task_signals(vikunja_id)
        except Exception:
            pass
        return rec

    def list_task_titles(self, vikunja_id: int, done: bool | None = None) -> list[str]:
        """Return task titles for a project. done=None returns all."""
        try:
            params: dict[str, Any] = {"page": 1, "per_page": 200}
            if done is not None:
                params["filter"] = f"done={'true' if done else 'false'}"
            tasks_data = self._get(f"/projects/{vikunja_id}/tasks", params=params)
            if isinstance(tasks_data, dict):
                tasks_data = tasks_data.get("tasks", [])
            return [t.get("title", "") for t in (tasks_data or []) if isinstance(t, dict)]
        except Exception:
            return []

    def create_task(self, vikunja_id: int, title: str, description: str = "") -> dict[str, Any] | None:
        """Create a task in a Vikunja project. Returns created task dict or None."""
        try:
            result = self._post(f"/projects/{vikunja_id}/tasks", json={"title": title, "description": description})
            logger.info("Created task '%s' in project %d", title, vikunja_id)
            return result
        except Exception:
            logger.exception("Failed to create task '%s' in project %d", title, vikunja_id)
            return None

    def close_task(self, vikunja_id: int, task_id: int) -> bool:
        """Mark a task as done. Returns success."""
        try:
            self._post(f"/projects/{vikunja_id}/tasks/{task_id}", method="PATCH", json={"done": True})
            logger.info("Closed task %d in project %d", task_id, vikunja_id)
            return True
        except Exception:
            logger.exception("Failed to close task %d in project %d", task_id, vikunja_id)
            return False

    # ── Internal helpers ──────────────────────────────────────────────

    def _fetch_projects(self) -> list[ProjectRecord]:
        """List all non-archived projects from Vikunja."""
        all_projects: list[ProjectRecord] = []
        page = 1
        while len(all_projects) < self._max_projects:
            try:
                data = self._get("/projects", params={"page": page, "per_page": 50})
            except Exception:
                logger.warning("Failed to fetch Vikunja projects page %d", page, exc_info=True)
                break
            if not data:
                break
            projects_list = data if isinstance(data, list) else data.get("projects", [])
            if not projects_list:
                break
            for p in projects_list:
                title = p.get("title", "")
                if p.get("is_archived"):
                    continue
                if title.lower() in self._ignore_projects:
                    continue
                all_projects.append(
                    ProjectRecord(
                        id=f"vikunja:{p['id']}",
                        name=title,
                        kind="vikunja",
                        description=p.get("description", ""),
                        vikunja_id=p["id"],
                        is_archived=p.get("is_archived", False),
                    )
                )
            if len(projects_list) < 50:
                break
            page += 1
        return all_projects

    def _fetch_task_signals(self, project_id: int) -> VikunjaSignals:
        """Fetch all tasks for a project and compute aggregate signals."""
        now = datetime.now(timezone.utc)
        open_count = 0
        done_count = 0
        overdue = 0
        oldest_days: float | None = None
        latest_update_days: float | None = None
        labels_seen: set[str] = set()

        page = 1
        while True:
            try:
                tasks_data = self._get(f"/projects/{project_id}/tasks", params={"page": page, "per_page": 200})
            except Exception:
                break
            if isinstance(tasks_data, dict):
                tasks_list = tasks_data.get("tasks", [])
            elif isinstance(tasks_data, list):
                tasks_list = tasks_data
            else:
                break
            if not tasks_list:
                break

            for t in tasks_list:
                if not isinstance(t, dict):
                    continue
                if t.get("done"):
                    done_count += 1
                else:
                    open_count += 1
                    due = t.get("due_date")
                    if due and isinstance(due, str) and due != "0001-01-01T00:00:00Z":
                        try:
                            due_dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
                            if due_dt < now:
                                overdue += 1
                        except Exception:
                            pass

                    created = t.get("created")
                    if created and isinstance(created, str):
                        try:
                            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                            age_days = (now - created_dt).total_seconds() / 86400
                            if oldest_days is None or age_days > oldest_days:
                                oldest_days = age_days
                        except Exception:
                            pass

                updated = t.get("updated")
                if updated and isinstance(updated, str):
                    try:
                        updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        since_update = (now - updated_dt).total_seconds() / 86400
                        if latest_update_days is None or since_update < latest_update_days:
                            latest_update_days = since_update
                    except Exception:
                        pass

                for label in (t.get("labels") or []):
                    if isinstance(label, dict):
                        labels_seen.add(label.get("title", ""))

            if len(tasks_list) < 200:
                break
            page += 1

        return VikunjaSignals(
            open_tasks=open_count,
            done_tasks=done_count,
            overdue_tasks=overdue,
            oldest_task_days=oldest_days,
            latest_task_update_days=latest_update_days,
            labels=sorted(labels_seen - {""}),
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base_url}{path}"
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        r = httpx.get(url, headers=headers, params=params, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, method: str = "POST", json: dict[str, Any] | None = None) -> Any:
        url = f"{self._base_url}{path}"
        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"} if self._token else {"Content-Type": "application/json"}
        r = httpx.request(method, url, headers=headers, json=json, timeout=self._timeout)
        r.raise_for_status()
        return r.json()
