"""Schema repair execution abstraction — applies operations to real data."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.value_objects.schema_repair_operations import (
    SchemaRepairOperation,
)


class SchemaExecutionOutcome(BaseModel):
    """Result of attempting to apply ordered operations to a file.

    `output_path` is only ever set by a successful `execute()`: the path
    of the *separate* repaired file it created. It is never the same
    path as the source file that was read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    success: bool
    validation_errors: list[str] = Field(default_factory=list)
    message: str | None = None
    output_path: str | None = None


@runtime_checkable
class SchemaExecutor(Protocol):
    """Applies (already-ordered) schema repair operations to a data file.

    `execute` and `verify` are deliberately separate, mirroring Tier 1's
    `CsvRepairExecutor`: **the source file passed to `execute` is never
    modified.** `execute` reads `file_path` (the source, read-only) and,
    on success, writes a *separate* repaired file, reporting its path as
    `output_path` on the returned `SchemaExecutionOutcome`. `verify`
    never writes anything; it is given whatever path should be
    independently re-checked (in practice, the `output_path` `execute`
    just produced, not the source) and confirms each operation's target
    end-state genuinely holds there.
    """

    def execute(
        self, file_path: str, operations: tuple[SchemaRepairOperation, ...]
    ) -> SchemaExecutionOutcome:
        """Read `file_path` (the source; never modified) and, if the
        operations apply cleanly, write a separate repaired file,
        reporting its path as `output_path`."""
        ...

    def verify(
        self, file_path: str, operations: tuple[SchemaRepairOperation, ...]
    ) -> SchemaExecutionOutcome:
        """Independently confirm `operations`' target end-state now holds
        at `file_path` (in practice, `execute`'s `output_path`)."""
        ...
