"""Human-in-the-loop approval abstraction for multi-failure CSV repair.

A single-failure CSV repair auto-applies exactly as it always has —
this gate exists only for the multi-failure case, where more than one
independent Tier 1 dimension is being repaired by one combined
prescription at once. Deliberately a separate, CSV-shaped Protocol from
Tier 2's `HumanApprovalPort`/`ApprovalRequest` (which are shaped around
`ColumnDiff`/`SchemaRepairPrescription` and don't fit CSV's data model)
rather than forcing an inappropriate coupling between the two —  but the
same design principle applies: framework-agnostic (no Click), so a CLI
prompt today can be swapped for a UI/API mechanism later with no change
to the workflow that calls it.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.mixed_delimiter_row_repair import (
    MixedDelimiterRowRepair,
)


class CsvApprovalRequest(BaseModel):
    """Everything a human (or any approval mechanism) needs to decide.

    `mixed_delimiter_rows` mirrors `prescription.mixed_delimiter_rows`
    exactly — surfaced as its own field so an approval port can display
    per-row evidence without reaching into `prescription`. Empty for
    every failure class except `MIXED_DELIMITER`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    file_path: str
    failure_classes: frozenset[FailureClass]
    prescription: CsvRepairParams
    mixed_delimiter_rows: tuple[MixedDelimiterRowRepair, ...] = ()


@runtime_checkable
class CsvHumanApprovalPort(Protocol):
    """Requests explicit human approval before a multi-failure repair may apply."""

    def request_approval(self, request: CsvApprovalRequest) -> bool:
        """Return `True` iff a human explicitly approved `request`."""
        ...
