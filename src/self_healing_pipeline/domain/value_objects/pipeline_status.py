"""Repair episode lifecycle status value object."""

from enum import Enum


class RepairEpisodeStatus(str, Enum):
    """Lifecycle state of a repair episode."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
