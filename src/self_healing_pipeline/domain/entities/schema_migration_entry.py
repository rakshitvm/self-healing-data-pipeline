"""A single, append-only migration history entry.

This is the exact shape written to the source-controlled migration JSON
(Tier 2 requirement 6) and mirrored into PostgreSQL. One entry is written
per repair attempt — `APPLIED` or `REJECTED` (or `ESCALATED`/`INVALID`/
`FAILED`) — never overwritten, so `episode_id` + `created_at` together
give a full, reproducible history of every decision made.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition
from self_healing_pipeline.domain.value_objects.column_diff import ColumnDiff
from self_healing_pipeline.domain.value_objects.schema_repair_prescription import (
    SchemaRepairPrescription,
)
from self_healing_pipeline.domain.value_objects.schema_repair_result import SchemaRepairStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SchemaMigrationEntry(BaseModel):
    """Immutable, append-only record of one schema repair attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    entry_id: UUID = Field(default_factory=uuid4)
    episode_id: UUID
    table: str
    baseline_version: int
    current_schema: tuple[ColumnDefinition, ...]
    diff: ColumnDiff
    prescription: SchemaRepairPrescription | None
    confidence: float | None
    status: SchemaRepairStatus
    applied: bool
    human_approved: bool | None
    verification_message: str | None
    trace_id: str | None
    mlflow_run_id: str | None
    token_usage: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=_utcnow)
