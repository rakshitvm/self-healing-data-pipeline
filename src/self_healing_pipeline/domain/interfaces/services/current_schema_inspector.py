"""Deterministic current-schema inspection abstraction.

Analogous to `CsvFailureDetector`: a pure, deterministic read of a real
data file's actual structure (no LLM, no randomness). Concrete
implementations (e.g. pandas-backed) live in infrastructure.
"""

from typing import Protocol, runtime_checkable

from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition


@runtime_checkable
class CurrentSchemaInspector(Protocol):
    """Infers the actual, current column structure of a data file."""

    def inspect(self, file_path: str) -> tuple[ColumnDefinition, ...]:
        """Return the current columns (name, type, nullable) of `file_path`."""
        ...
