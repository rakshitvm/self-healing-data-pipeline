"""Provider-agnostic CSV repair proposal abstraction.

Defines the reasoning boundary between "what repair parameters should we
try" and whatever actually decides that — an LLM call (Azure OpenAI or
any other provider), a rule-based heuristic, or anything else. Callers
depend only on `CsvRepairProposalPort`; no concrete provider is coupled
into the domain or application layers here (Open/Closed, Dependency
Inversion).

The return value is a raw, UNVALIDATED payload. A proposal must never be
treated as a usable `CsvRepairParams` until it has been validated through
that model — this port only proposes, it does not validate.
"""

from typing import Any, Protocol, runtime_checkable

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass


@runtime_checkable
class CsvRepairProposalPort(Protocol):
    """Proposes a raw CSV repair parameter payload for a detected failure."""

    def propose(
        self, *, failure_class: FailureClass, sample: str, file_path: str
    ) -> dict[str, Any]:
        """Propose a raw (unvalidated) candidate for `CsvRepairParams`."""
        ...
