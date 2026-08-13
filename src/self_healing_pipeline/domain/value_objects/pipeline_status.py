"""Repair episode lifecycle status value object."""

from enum import Enum


class RepairEpisodeStatus(str, Enum):
    """Lifecycle state of a repair episode.

    `REJECTED`: a human explicitly declined a proposed multi-failure
    repair before `apply` ran. Distinct from `FAILED` (validation/apply/
    reverify itself did not succeed) — `REJECTED` means apply never ran.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
