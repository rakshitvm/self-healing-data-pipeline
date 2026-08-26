"""Outcome of a single Tier 2 schema repair episode."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.value_objects.column_diff import ColumnDiff
from self_healing_pipeline.domain.value_objects.schema_repair_prescription import (
    SchemaRepairPrescription,
)


class SchemaRepairStatus(str, Enum):
    """Terminal state of a schema repair episode.

    `HEALTHY`: no drift detected, nothing else ran. `ESCALATED`: below
    the confidence threshold — still requires (and may still receive)
    human approval, but is flagged for extra visibility. `REJECTED`: a
    human explicitly declined. `SUCCEEDED`/`FAILED`: human approved and
    apply+verify did/did not succeed. `INVALID`: the assembled
    prescription failed structural validation before ever reaching a
    human.
    """

    HEALTHY = "healthy"
    ESCALATED = "escalated"
    REJECTED = "rejected"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INVALID = "invalid"


class SchemaRepairResult(BaseModel):
    """Immutable outcome of a schema repair episode."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    status: SchemaRepairStatus
    diff: ColumnDiff | None = None
    prescription: SchemaRepairPrescription | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    human_approved: bool | None = None
    validation_errors: list[str] = Field(default_factory=list)
    message: str | None = None
    output_path: str | None = None

    @property
    def success(self) -> bool:
        return self.status is SchemaRepairStatus.SUCCEEDED

    @property
    def applied(self) -> bool:
        return self.status is SchemaRepairStatus.SUCCEEDED
