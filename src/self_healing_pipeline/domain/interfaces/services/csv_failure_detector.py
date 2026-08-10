"""CSV failure detection abstraction.

Defines the boundary between "does this CSV file have a Tier 1 failure,
and which one" and however that determination is actually made.
Consumers depend only on `CsvFailureDetector`; a concrete implementation
(stdlib-based, pandas-based, or otherwise) can be substituted without any
domain model changing (Open/Closed, Dependency Inversion).
"""

from typing import Protocol, runtime_checkable

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass


@runtime_checkable
class CsvFailureDetector(Protocol):
    """Determines which Tier 1 CSV failure dimension, if any, applies."""

    def detect(self, file_path: str) -> FailureClass | None:
        """Inspect `file_path` and return the failure it exhibits, if any."""
        ...
