"""Scheduler — APScheduler cron jobs.

Runs periodic tasks: stale task checks, daily briefings, weekly reviews.

Stolen from: APScheduler (standard) + Daily Briefing Agent pattern
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

# Type alias for async job functions
AsyncJob = Callable[[], Coroutine[Any, Any, None]]


class SecretaryScheduler:
    """Manages cron-scheduled jobs for the secretary daemon."""

    def __init__(self, config: dict[str, Any]) -> None:
        sched_cfg = config.get("scheduler", {})
        self.schedules = {
            "stale_task_check": sched_cfg.get("stale_task_check", "0 * * * *"),
            "morning_briefing": sched_cfg.get("morning_briefing", "0 9 * * *"),
            "evening_summary": sched_cfg.get("evening_summary", "0 18 * * *"),
            "weekly_review": sched_cfg.get("weekly_review", "0 10 * * 0"),

        }
        self.timezone = sched_cfg.get("timezone", "UTC")
        try:
            self._scheduler = AsyncIOScheduler(timezone=self.timezone)
        except Exception as exc:
            raise ValueError(f"Invalid scheduler timezone {self.timezone!r}: {exc}") from exc
        self._jobs: dict[str, AsyncJob] = {}

    def register(self, name: str, func: AsyncJob) -> None:
        """Register a named job function."""
        self._jobs[name] = func
        logger.info("Registered job: %s", name)

    def start(self) -> None:
        """Start the scheduler with all registered jobs."""
        for name, func in self._jobs.items():
            cron_str = self.schedules.get(name)
            if not cron_str:
                logger.warning("No schedule for job '%s', skipping", name)
                continue

            trigger = self._parse_cron(cron_str)
            self._scheduler.add_job(
                func,
                trigger=trigger,
                id=name,
                name=name,
                replace_existing=True,
            )
            logger.info("Scheduled job '%s': %s", name, cron_str)

        self._scheduler.start()
        logger.info("Scheduler started with %d jobs", len(self._jobs))

    def stop(self) -> None:
        """Gracefully stop the scheduler (idempotent)."""
        try:
            if self._scheduler.running:
                self._scheduler.shutdown(wait=False)
                logger.info("Scheduler stopped")
        except Exception:
            logger.debug("Scheduler shutdown failed", exc_info=True)

    def get_next_run(self, job_name: str) -> str | None:
        """Get next run time for a job."""
        job = self._scheduler.get_job(job_name)
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
        return None

    @staticmethod
    def _parse_cron(cron_str: str) -> CronTrigger:
        """Parse a 5-field cron string into CronTrigger.

        Format: minute hour day_of_month month day_of_week
        """
        parts = cron_str.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron string (expected 5 fields): {cron_str}")

        return CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )
