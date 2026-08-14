"""Repair result value object.

Represents the outcome of a single repair attempt. It is an immutable
value object returned by application-layer use cases; it holds no
infrastructure or repair-execution logic itself.
"""

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams


class RepairResult(BaseModel):
    """Immutable outcome of a repair attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    success: bool
    applied: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    prescription: CsvRepairParams | None = None
    validation_errors: list[str] = Field(default_factory=list)
    message: str | None = None
    source_path: str | None = None
    output_path: str | None = None
