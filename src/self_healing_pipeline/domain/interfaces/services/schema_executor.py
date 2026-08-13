"""Schema repair execution abstraction — applies operations to real data."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.value_objects.schema_repair_operations import (
    SchemaRepairOperation,
)


class SchemaExecutionOutcome(BaseModel):
    """Result of attempting to apply ordered operations to a file."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    success: bool
    validation_errors: list[str] = Field(default_factory=list)
    message: str | None = None


@runtime_checkable
class SchemaExecutor(Protocol):
    """Applies (already-ordered) schema repair operations to a data file.

    `execute` and `verify` are deliberately separate: unlike Tier 1's
    read-only CSV repair (where re-running the same call is naturally
    idempotent), schema operations *mutate* the file — rename/cast/drop/
    add_default all change it in place. `verify` never re-applies
    anything; it independently re-inspects the (now-modified) file and
    confirms each operation's target end-state genuinely holds.
    """

    def execute(
        self, file_path: str, operations: tuple[SchemaRepairOperation, ...]
    ) -> SchemaExecutionOutcome:
        """Apply `operations`, in the given order, to `file_path` (mutating it)."""
        ...

    def verify(
        self, file_path: str, operations: tuple[SchemaRepairOperation, ...]
    ) -> SchemaExecutionOutcome:
        """Independently confirm `operations`' target end-state now holds."""
        ...
