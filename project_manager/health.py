"""Project health evaluator based on Vikunja signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .registry import ProjectRecord, VikunjaSignals


@dataclass
class HealthVerdict:
    score: float  # 0-100
    level: str  # green | yellow | red
    reasons: list[str]


def health_summary(projects: list[ProjectRecord], rules: dict[str, Any]) -> list[dict[str, Any]]:
    evaluator = ProjectHealthEvaluator(rules)
    out = []
    for p in projects:
        h = evaluator.evaluate(p)
        out.append(
            {
                "id": p.id,
                "name": p.name,
                "kind": p.kind,
                "vikunja_id": p.vikunja_id,
                "score": h.score,
                "level": h.level,
                "reasons": h.reasons,
                "signals": {
                    "open_tasks": p.signals.open_tasks,
                    "done_tasks": p.signals.done_tasks,
                    "overdue_tasks": p.signals.overdue_tasks,
                    "oldest_task_days": p.signals.oldest_task_days,
                    "latest_task_update_days": p.signals.latest_task_update_days,
                    "labels": p.signals.labels,
                },
            }
        )
    return sorted(out, key=lambda x: ({"red": 0, "yellow": 1, "green": 2}[x["level"]], -x["score"]))


class ProjectHealthEvaluator:
    def __init__(self, rules: dict[str, Any]) -> None:
        self.rules = rules

    def evaluate(self, project: ProjectRecord) -> HealthVerdict:
        score = 100.0
        reasons: list[str] = []
        sig = project.signals

        max_open = int(self.rules.get("max_open_tasks", 20))
        overdue_stale_days = int(self.rules.get("overdue_stale_days", 14))
        max_inactive_days = int(self.rules.get("max_inactive_days", 30))

        # Overdue tasks are a strong negative signal
        if sig.overdue_tasks > 0:
            penalty = min(30, 5 * sig.overdue_tasks)
            score -= penalty
            reasons.append(f"overdue_tasks={sig.overdue_tasks}")

        # Too many open tasks suggests project is losing focus
        if sig.open_tasks > max_open:
            score -= 10
            reasons.append(f"open_tasks={sig.open_tasks}")

        # No tasks at all is slightly suspicious
        if sig.open_tasks == 0 and sig.done_tasks == 0:
            score -= 5
            reasons.append("empty_project")

        # Stale oldest task — something has been sitting for too long
        if sig.oldest_task_days is not None and sig.oldest_task_days > overdue_stale_days:
            score -= min(15, (sig.oldest_task_days - overdue_stale_days) / 2)
            reasons.append(f"oldest_task={sig.oldest_task_days:.0f}d")

        # No recent task activity
        if sig.latest_task_update_days is not None and sig.latest_task_update_days > max_inactive_days:
            score -= min(20, (sig.latest_task_update_days - max_inactive_days) / 3)
            reasons.append(f"inactive={sig.latest_task_update_days:.0f}d")

        # Ratio check: if done/open < 0.3, project may be stalled
        if sig.open_tasks > 5:
            ratio = sig.done_tasks / max(1, sig.open_tasks)
            if ratio < 0.3:
                score -= 10
                reasons.append(f"low_completion_ratio={ratio:.2f}")

        score = max(0.0, min(100.0, score))
        level = "green" if score >= 80 else ("yellow" if score >= 55 else "red")
        return HealthVerdict(score=score, level=level, reasons=reasons)
