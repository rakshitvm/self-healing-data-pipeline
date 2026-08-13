"""Provider-agnostic repair run tracking abstraction.

MLflow (or any other experiment-tracking system) is observability, not
the audit-of-record — `RepairAuditStore` / PostgreSQL remains the sole
authoritative record of what happened. Tracking through `RepairRunTracker`
is strictly best-effort: a tracking failure must never fail, roll back,
or invalidate an otherwise successful repair. It must also never be
silently mistaken for success — every method reports whether it actually
worked via `TrackingOutcome`, in the same spirit as `CsvExecutionOutcome`.
"""

from typing import Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass


class TrackingOutcome(BaseModel):
    """Result of a single best-effort tracking operation."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    success: bool
    run_id: str | None = None
    error: str | None = None


@runtime_checkable
class RepairRunTracker(Protocol):
    """Records repair-episode observability data in an experiment tracker."""

    def start_run(self, *, episode_id: UUID, failure_class: FailureClass) -> TrackingOutcome:
        """Start a tracked run for one repair episode."""
        ...

    def log_metrics(
        self,
        run_id: str,
        *,
        latency_ms: int | None = None,
        token_usage: int | None = None,
        trace_id: str | None = None,
    ) -> TrackingOutcome:
        """Log whatever metrics are available for `run_id`.

        `trace_id`, when provided, lets the implementation derive real
        token usage/cost from already-captured LLM span data (e.g.
        MLflow's own OpenAI autolog) — never an invented or estimated
        figure. `token_usage` remains available as a direct override for
        a caller that already has a real value in hand.
        """
        ...

    def end_run(self, run_id: str, *, status: str) -> TrackingOutcome:
        """Terminate the tracked run for `run_id` with a final status."""
        ...
