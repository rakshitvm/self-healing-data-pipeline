"""Human-in-the-loop approval abstraction.

The LLM may analyze, propose, recommend, and assign confidence — it may
never authorize a data-changing repair. Every prescription, regardless
of confidence, must pass through this port before `apply`. Deliberately
framework-agnostic (no Click, no HTTP) so a CLI prompt today can be
swapped for a UI/API/queue-based approval mechanism later without any
change to the workflow that calls it.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from self_healing_pipeline.domain.value_objects.column_diff import ColumnDiff
from self_healing_pipeline.domain.value_objects.schema_repair_prescription import (
    SchemaRepairPrescription,
)


class ApprovalRequest(BaseModel):
    """Everything a human (or any approval mechanism) needs to decide."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    table: str
    diff: ColumnDiff
    prescription: SchemaRepairPrescription
    confidence: float
    escalated: bool


@runtime_checkable
class HumanApprovalPort(Protocol):
    """Requests explicit human approval before a repair may be applied."""

    def request_approval(self, request: ApprovalRequest) -> bool:
        """Return `True` iff a human explicitly approved `request`."""
        ...
