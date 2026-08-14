"""LangGraph-backed `RepairAgent` — the canonical Tier 1 CSV repair handler.

Thin adapter between `ErrorRouter` (which only knows the `RepairAgent`
Protocol: `handle(error) -> RepairResult`) and the existing, unmodified
`csv_repair_workflow` `StateGraph` plus its audited wrapper. It contains
no repair logic of its own — it does not duplicate the LangGraph
workflow, and it is not a second CSV repair implementation. Its entire
job is: build the initial state for `error.file_path`, run the canonical
workflow through `run_audited_csv_repair` (which drives PostgreSQL audit
and best-effort MLflow tracking exactly as already implemented), and
translate the resulting `CsvRepairWorkflowState` back into the
`RepairResult` `ErrorRouter` expects.

Note: the workflow's own `diagnose` node independently re-detects the
failure via `CsvFailureDetector` — it does not trust `error`'s type. This
is intentional, not an oversight: `error` only tells us *that* something
upstream believed there was a failure (and supplies `file_path`); the
workflow's own diagnosis is the authoritative, up-to-date check, exactly
as it is for every other caller of the workflow.
"""

from typing import Any

from langgraph.graph.state import CompiledStateGraph

from self_healing_pipeline.application.orchestration.audited_csv_repair import (
    run_audited_csv_repair,
)
from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    DEFAULT_MAX_RETRIES,
    CsvRepairWorkflowState,
    build_initial_state,
)
from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.interfaces.repositories.repair_audit_store import (
    RepairAuditStore,
)
from self_healing_pipeline.domain.interfaces.services.repair_run_tracker import RepairRunTracker
from self_healing_pipeline.domain.interfaces.services.repair_trace_tracer import RepairTraceTracer
from self_healing_pipeline.domain.value_objects.repair_result import RepairResult


class LangGraphCsvRepairAgent:
    """Concrete `RepairAgent` that routes to the canonical LangGraph workflow."""

    def __init__(
        self,
        graph: CompiledStateGraph[CsvRepairWorkflowState, None, Any, Any],
        *,
        audit_store: RepairAuditStore,
        run_tracker: RepairRunTracker | None = None,
        trace_tracer: RepairTraceTracer | None = None,
        source: str = "local",
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._graph = graph
        self._audit_store = audit_store
        self._run_tracker = run_tracker
        self._trace_tracer = trace_tracer
        self._source = source
        self._max_retries = max_retries

    def handle(self, error: PipelineError) -> RepairResult:
        """Repair the CSV failure described by `error` via the LangGraph workflow."""
        if error.file_path is None:
            return RepairResult(
                success=False,
                applied=False,
                validation_errors=["missing_file_path"],
                message=f"Cannot repair {type(error).__name__}: no file_path was provided.",
            )

        initial_state = build_initial_state(error.file_path, max_retries=self._max_retries)
        final_state = run_audited_csv_repair(
            self._graph,
            initial_state,
            audit_store=self._audit_store,
            source=self._source,
            run_tracker=self._run_tracker,
            trace_tracer=self._trace_tracer,
        )

        if final_state["repair_result"] is not None:
            return final_state["repair_result"]

        if final_state["failure_class"] is None:
            return RepairResult(
                success=True,
                applied=False,
                source_path=error.file_path,
                message="No CSV failure detected; the file already parses correctly.",
            )

        return RepairResult(
            success=False,
            applied=False,
            validation_errors=final_state["validation_errors"],
            source_path=error.file_path,
            message=(
                f"Repair for {final_state['failure_class'].value} did not reach "
                "the apply step."
            ),
        )
