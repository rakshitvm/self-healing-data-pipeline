"""Versioned schema baseline persistence abstraction."""

from typing import Protocol, runtime_checkable

from self_healing_pipeline.domain.entities.schema_definition import SchemaBaseline


@runtime_checkable
class SchemaBaselineStore(Protocol):
    """Loads the current (latest) source-controlled baseline for a table."""

    def load_latest(self, table: str) -> SchemaBaseline | None:
        """Return `table`'s latest baseline, or `None` if none exists yet."""
        ...
