"""Tier 1 CSV failure-mode exceptions.

Each exception below corresponds to exactly one Tier 1 CSV failure
dimension defined by `FailureClass`. Fixing `FAILURE_CLASS` at the class
level (rather than requiring callers to pass it) attaches failure
identity to the exception's type, so a router can dispatch on type,
`isinstance` against `CsvRepairError`/`PipelineError`, or the
`failure_class` instance attribute — whichever suits it — without this
module knowing anything about the router or any concrete repair agent.

Future Tier 2/3 failure modes are added the same way: a new subclass of
`PipelineError` (or `CsvRepairError`, if CSV-related) with its own
`FAILURE_CLASS`. No existing class, and no router, needs to change.
"""

from typing import ClassVar

from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass


class CsvRepairError(PipelineError):
    """Base class for all Tier 1 CSV ingestion failure modes.

    Not intended to be raised directly; raise one of its dimension-specific
    subclasses instead. Each subclass fixes `FAILURE_CLASS`, a class-level
    constant, which is copied into the `failure_class` instance attribute
    (defined on `PipelineError`) at construction time.
    """

    FAILURE_CLASS: ClassVar[FailureClass]

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        table_name: str | None = None,
        file_path: str | None = None,
    ) -> None:
        super().__init__(
            message,
            failure_class=type(self).FAILURE_CLASS,
            source=source,
            table_name=table_name,
            file_path=file_path,
        )


class WrongDelimiterError(CsvRepairError):
    """Raised when a CSV file was parsed using an incorrect delimiter."""

    FAILURE_CLASS = FailureClass.WRONG_DELIMITER


class WrongEncodingError(CsvRepairError):
    """Raised when a CSV file was decoded using an incorrect text encoding."""

    FAILURE_CLASS = FailureClass.WRONG_ENCODING


class HeaderDetectionError(CsvRepairError):
    """Raised when a CSV file's header row could not be reliably detected."""

    FAILURE_CLASS = FailureClass.HEADER_DETECTION


class EngineSelectionError(CsvRepairError):
    """Raised when no available parsing engine could successfully read a CSV file."""

    FAILURE_CLASS = FailureClass.ENGINE_SELECTION


class SingleColumnMalformationError(CsvRepairError):
    """Raised when a CSV file silently collapsed into a single column."""

    FAILURE_CLASS = FailureClass.SINGLE_COLUMN_MALFORMATION


class MixedDelimiterError(CsvRepairError):
    """Raised when a minority of rows use a different delimiter than the
    file's established one (e.g. one semicolon-delimited row in an
    otherwise comma-delimited file)."""

    FAILURE_CLASS = FailureClass.MIXED_DELIMITER
