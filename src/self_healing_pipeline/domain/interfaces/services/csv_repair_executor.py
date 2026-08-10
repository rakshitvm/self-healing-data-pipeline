"""CSV repair execution abstraction.

Defines the boundary between a CSV repair agent's decision logic (which
`CsvRepairParams` prescription to try) and the mechanics of actually
attempting to read a CSV file with that prescription. Agents depend only
on `CsvRepairExecutor`; a concrete implementation backed by pandas,
Spark, Databricks, or any other engine can be substituted later without
the agent, this abstraction, or any domain model changing (Open/Closed,
Dependency Inversion).
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams


class CsvExecutionOutcome(BaseModel):
    """Result of attempting to read a CSV file with a given prescription."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    success: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    validation_errors: list[str] = Field(default_factory=list)
    message: str | None = None


@runtime_checkable
class CsvRepairExecutor(Protocol):
    """Attempts to read a CSV file using a candidate prescription.

    Implementations perform the actual CSV manipulation; this protocol is
    the only thing a CSV repair agent depends on to do so.
    """

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        """Attempt to read `file_path` using `params`; report the outcome."""
        ...
