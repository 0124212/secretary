"""Project overseer — scans Vikunja, detects problems, takes autonomous action."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .health import ProjectHealthEvaluator, health_summary
from .registry import ProjectRecord, ProjectRegistry, VikunjaSignals

logger = logging.getLogger(__name__)


@dataclass
class ProjectOverview:
    generated_at: str
    projects: list[dict[str, Any]]
    red_projects: list[dict[str, Any]]
    actions_today: int


@dataclass
class ActionTaken:
    project: str
    action: str
    detail: str


class ProjectOverseer:
    """Scans projects, detects problems, takes autonomous action via opencode."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.registry = ProjectRegistry(config)
        self.health = ProjectHealthEvaluator(config.get("projects", {}).get("health_rules", {}))
        self._overview: ProjectOverview | None = None
        self._actions_today: list[ActionTaken] = []
        self._max_actions = int(config.get("projects", {}).get("max_actions_per_day", 10))

    @property
    def overview(self) -> ProjectOverview | None:
        return self._overview

    def scan(self) -> ProjectOverview:
        """Refresh projects from Vikunja, compute health, return overview."""
        projects = self.registry.discover()
        rows = health_summary(projects, self.config.get("projects", {}).get("health_rules", {}))
        red = [r for r in rows if r["level"] == "red"]
        self._overview = ProjectOverview(
            generated_at=datetime.now(timezone.utc).isoformat(),
            projects=rows,
            red_projects=red,
            actions_today=len(self._actions_today),
        )
        return self._overview

    def _build_prompt(self, overview: ProjectOverview) -> str | None:
        """Build an opencode prompt describing the problem and what action to take."""
        if not overview.red_projects:
            return None

        lines = [
            "You are the secretary overseer. These Vikunja projects need action:",
            "",
        ]
        for p in overview.red_projects:
            reasons = "; ".join(p.get("reasons", []))
            lines.append(f"Project: {p['name']} (health score: {p['score']:.0f})")
            lines.append(f"  Problem: {reasons}")
            lines.append(f"  Open tasks: {p['signals'].get('open_tasks', 0)}")
            lines.append(f"  Overdue tasks: {p['signals'].get('overdue_tasks', 0)}")
            lines.append(f"  Oldest task: {p['signals'].get('oldest_task_days', 'unknown')} days")
            lines.append("")

        lines.append("Take ONE concrete action for the most critical project:")
        lines.append("- If tasks are overdue: describe which task to close or update")
        lines.append("- If project is stale: describe what to create to move it forward")
        lines.append("- If too many open tasks: describe which to close as won't-do")
        lines.append("")
        lines.append("Respond with: ACTION:<project_name>:<what_you_did>")
        return "\n".join(lines)

    async def check_and_act(self, opencode_client: Any) -> list[ActionTaken]:
        """Full cycle: scan → detect → take action via opencode MCP."""
        if len(self._actions_today) >= self._max_actions:
            logger.info("Hit daily action limit (%d), skipping", self._max_actions)
            return []

        overview = self.scan()
        prompt = self._build_prompt(overview)
        if not prompt:
            logger.debug("All projects healthy, no action needed")
            return []

        actions: list[ActionTaken] = []
        try:
            response = await opencode_client.prompt(prompt)
            # Parse ACTION: lines from response
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("ACTION:"):
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        action = ActionTaken(
                            project=parts[1].strip(),
                            action="opencode",
                            detail=parts[2].strip(),
                        )
                        actions.append(action)
                        self._actions_today.append(action)

            if not actions:
                # opencode didn't produce parseable ACTION lines, log the response
                logger.info("Overseer prompt sent, response: %s", response[:200])
                self._actions_today.append(ActionTaken(
                    project=", ".join(p["name"] for p in overview.red_projects),
                    action="advisory",
                    detail=response[:200],
                ))

        except Exception:
            logger.exception("Failed to trigger opencode action")

        return actions

    def summarize_for_briefing(self, overview: ProjectOverview | None = None) -> str:
        """One-liner for the morning briefing."""
        ov = overview or self._overview
        if not ov:
            return "Project overview: not yet scanned."
        red_count = len(ov.red_projects)
        total = len(ov.projects)
        if red_count == 0:
            return f"Projects: {total} total, all healthy."
        names = ", ".join(p["name"] for p in ov.red_projects[:3])
        return f"Projects: {total} total, {red_count} need attention ({names})."
