"""CSV repair execution abstraction.

Defines the boundary between a CSV repair agent's decision logic (which
`CsvRepairParams` prescription to try) and the mechanics of actually
repairing a CSV file with that prescription. Agents depend only on
`CsvRepairExecutor`; a concrete implementation backed by pandas, Spark,
Databricks, or any other engine can be substituted later without the
agent, this abstraction, or any domain model changing (Open/Closed,
Dependency Inversion).
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams


class CsvExecutionOutcome(BaseModel):
    """Result of attempting to apply or verify a repair prescription
    against a CSV file."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    success: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    validation_errors: list[str] = Field(default_factory=list)
    message: str | None = None


@runtime_checkable
class CsvRepairExecutor(Protocol):
    """Applies a candidate prescription to a CSV file and independently
    verifies the result.

    `execute` and `verify` are deliberately separate, mirroring Tier 2's
    `SchemaExecutor`: `execute` *mutates* `file_path` — it only ever
    reports success after physically rewriting the file into canonical
    form — while `verify` never writes anything; it independently
    re-inspects the (now-rewritten) file to confirm the repair actually
    holds. Implementations perform the actual CSV manipulation; this
    protocol is the only thing a CSV repair agent depends on to do so.
    """

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        """Apply `params` to `file_path`, physically rewriting it into
        canonical CSV form. Only reports `success=True` once that
        rewrite has actually happened — never on a successful read alone."""
        ...

    def verify(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        """Independently confirm `file_path` now holds a valid, sane CSV.
        Read-only: must never write to `file_path`."""
        ...
