"""Project Overseer — scans Vikunja, detects problems, takes autonomous action."""

from __future__ import annotations

import logging
from typing import Any

from .registry import ProjectRecord, ProjectRegistry, VikunjaSignals
from .health import ProjectHealthEvaluator, health_summary
from .manager import ProjectOverseer, ProjectOverview, ActionTaken

logger = logging.getLogger(__name__)

__all__ = [
    "ProjectRecord",
    "ProjectRegistry",
    "VikunjaSignals",
    "ProjectHealthEvaluator",
    "health_summary",
    "ProjectOverseer",
    "ProjectOverview",
    "ActionTaken",
]
