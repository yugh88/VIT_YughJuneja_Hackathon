"""Disk-space guard for jobs that write large files (project rule since Phase 3).

Before a job that may write more than ``STOP_ABOVE_GB``, it stops if free space would
fall below ``FLOOR_GB``; smaller jobs only warn. Experiments write to ``scratch/``,
which ``make clean-scratch`` empties.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from engine.config import REPO_ROOT

log = logging.getLogger(__name__)
FLOOR_GB = 15.0
STOP_ABOVE_GB = 2.0
SCRATCH = REPO_ROOT / "scratch"


class DiskSpaceError(RuntimeError):
    pass


def free_gb(path: Path = REPO_ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def require_space(job: str, expected_write_gb: float, path: Path = REPO_ROOT) -> None:
    free = free_gb(path)
    after = free - expected_write_gb
    if after >= FLOOR_GB:
        return
    msg = f"{job}: ~{expected_write_gb:.1f} GB to write, {free:.1f} GB free -> {after:.1f} GB left (floor {FLOOR_GB} GB)"
    if expected_write_gb > STOP_ABOVE_GB:
        raise DiskSpaceError(msg + ". Stopped: free space first (make clean-scratch) or confirm.")
    log.warning(msg)
