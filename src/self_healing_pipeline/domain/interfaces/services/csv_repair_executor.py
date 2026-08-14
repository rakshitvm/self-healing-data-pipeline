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
    against a CSV file.

    `output_path` is only ever set by a successful `execute()`: the path
    of the *separate* repaired file it created. It is never the same
    path as the source file that was read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    success: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    validation_errors: list[str] = Field(default_factory=list)
    message: str | None = None
    output_path: str | None = None


@runtime_checkable
class CsvRepairExecutor(Protocol):
    """Applies a candidate prescription to a CSV file and independently
    verifies the result.

    `execute` and `verify` are deliberately separate, mirroring Tier 2's
    `SchemaExecutor` — with one further constraint specific to Tier 1:
    **the source file passed to `execute` is never modified.** `execute`
    reads `file_path` (the source, read-only) and, on success, writes a
    *separate* repaired file, reporting its path as `output_path` on the
    returned `CsvExecutionOutcome`. `verify` never writes anything; it is
    given whatever path should be independently re-checked (in practice,
    the `output_path` `execute` just produced, not the source) and
    confirms that file holds a valid, sane CSV. Implementations perform
    the actual CSV manipulation; this protocol is the only thing a CSV
    repair agent depends on to do so.
    """

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        """Read `file_path` (the source; never modified) and, if `params`
        resolves it, write a separate repaired file, reporting its path
        as `output_path`. Only reports `success=True` once that new file
        has actually been written — never on a successful read alone."""
        ...

    def verify(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        """Independently confirm `file_path` now holds a valid, sane CSV.
        Read-only: must never write to `file_path` (or any other file)."""
        ...
