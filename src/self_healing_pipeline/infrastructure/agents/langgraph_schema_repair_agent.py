"""LangGraph-backed `RepairAgent` — the Tier 2 schema-drift repair handler.

Thin adapter between `ErrorRouter` (which only knows the `RepairAgent`
Protocol: `handle(error) -> RepairResult`) and the existing, unmodified
`schema_repair_workflow` `StateGraph` plus its audited wrapper
(`run_audited_schema_repair`) — mirroring `LangGraphCsvRepairAgent`'s
role for Tier 1 exactly. It contains no schema-diff or repair logic of
its own: `SchemaBaselineStore`, `CurrentSchemaInspector`,
`compute_column_diff`, the rename-resolution subgraph, confidence
gating, human approval, apply, and verify are all exactly what
`build_schema_repair_workflow` already wires up — this class only
translates `PipelineError` -> initial state, runs the canonical
workflow, and translates its `SchemaRepairResult` back into the
`RepairResult` shape `ErrorRouter`/the CLI's cloud-integration step
already understand, so no downstream code needs to know Tier 2 exists.

Status -> `RepairResult` mapping (deliberately mirrors Tier 1's own
healthy/failure conventions so the exact same CLI branching applies to
both tiers with no special-casing):

- `HEALTHY` (no drift): `success=True, applied=False` — same shape
  `LangGraphCsvRepairAgent` already returns for an already-healthy CSV.
  The cloud-integration step is skipped, exactly as it already is for a
  healthy Tier 1 file today.
- `SUCCEEDED` (approved, applied, verified): `success=True,
  applied=True, output_path=result.output_path`. Mirroring Tier 1
  exactly, `PandasSchemaExecutor` never touches the source file: the
  repaired file is written to a separate `repaired/<name>` path
  alongside it, reported here as `output_path`.
- `REJECTED` / `FAILED` / `INVALID` (including "no baseline found"):
  `success=False, applied=False` — the cloud step is skipped and the
  CLI exits non-zero, exactly as an unsuccessful Tier 1 repair already
  does. No Azure upload or Databricks trigger can be reached from here.
"""

from typing import Any
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from self_healing_pipeline.application.orchestration.audited_schema_repair import (
    run_audited_schema_repair,
)
from self_healing_pipeline.application.orchestration.schema_repair_workflow import (
    SchemaRepairWorkflowState,
    build_initial_schema_repair_state,
    result_from_final_state,
)
from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.interfaces.repositories.schema_migration_history_store import (
    SchemaMigrationHistoryStore,
)
from self_healing_pipeline.domain.interfaces.services.repair_trace_tracer import RepairTraceTracer
from self_healing_pipeline.domain.value_objects.repair_result import RepairResult
from self_healing_pipeline.domain.value_objects.schema_repair_result import SchemaRepairStatus


class LangGraphSchemaRepairAgent:
    """Concrete `RepairAgent` that routes Tier 2 schema-drift errors to the
    canonical Tier 2 LangGraph workflow."""

    def __init__(
        self,
        graph: CompiledStateGraph[SchemaRepairWorkflowState, None, Any, Any],
        *,
        history_store: SchemaMigrationHistoryStore,
        trace_tracer: RepairTraceTracer | None = None,
    ) -> None:
        self._graph = graph
        self._history_store = history_store
        self._trace_tracer = trace_tracer

    def handle(self, error: PipelineError) -> RepairResult:
        """Check `error.table_name`'s schema for drift via the LangGraph
        workflow, and repair it (subject to human approval) if found."""
        if error.table_name is None or error.file_path is None:
            return RepairResult(
                success=False,
                applied=False,
                validation_errors=["missing_table_name_or_file_path"],
                message=(
                    f"Cannot check schema drift for {type(error).__name__}: "
                    "both table_name and file_path are required."
                ),
            )

        initial_state = build_initial_schema_repair_state(
            episode_id=uuid4(), table=error.table_name, file_path=error.file_path
        )
        final_state = run_audited_schema_repair(
            self._graph,
            initial_state,
            history_store=self._history_store,
            trace_tracer=self._trace_tracer,
        )
        result = result_from_final_state(final_state)

        if result.status is SchemaRepairStatus.HEALTHY:
            return RepairResult(
                success=True,
                applied=False,
                source_path=error.file_path,
                message=result.message or "No schema drift detected.",
            )

        if result.status is SchemaRepairStatus.SUCCEEDED:
            return RepairResult(
                success=True,
                applied=True,
                confidence=result.confidence,
                source_path=error.file_path,
                output_path=result.output_path,
                message=result.message,
            )

        return RepairResult(
            success=False,
            applied=False,
            confidence=result.confidence,
            validation_errors=result.validation_errors,
            source_path=error.file_path,
            message=result.message,
        )
