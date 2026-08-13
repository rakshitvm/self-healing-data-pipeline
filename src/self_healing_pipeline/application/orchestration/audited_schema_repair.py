"""Migration-history-recording wrapper around `schema_repair_workflow`.

Mirrors Tier 1's `audited_csv_repair.py` shape (run the graph, then
record what happened) but for Tier 2: the audit-of-record is the
append-only `SchemaMigrationHistoryStore` (source-controlled JSON,
Tier 2 requirement 6) rather than PostgreSQL — PostgreSQL remains a
best-effort mirror when the concrete store also writes there (see
`infrastructure/schema/postgres_schema_migration_store.py`), exactly as
MLflow tracking is best-effort relative to Tier 1's PostgreSQL audit.

A healthy (no-drift) episode records nothing — there was no repair
attempt to audit, mirroring Tier 1's identical rule for healthy CSVs.
"""

from typing import Any, cast

from langgraph.graph.state import CompiledStateGraph

from self_healing_pipeline.application.orchestration.schema_repair_workflow import (
    SchemaRepairWorkflowState,
)
from self_healing_pipeline.domain.entities.schema_migration_entry import SchemaMigrationEntry
from self_healing_pipeline.domain.interfaces.repositories.schema_migration_history_store import (
    SchemaMigrationHistoryStore,
)
from self_healing_pipeline.domain.interfaces.services.repair_trace_tracer import RepairTraceTracer
from self_healing_pipeline.domain.value_objects.column_diff import ColumnDiff
from self_healing_pipeline.domain.value_objects.schema_repair_prescription import (
    SchemaRepairPrescription,
)
from self_healing_pipeline.domain.value_objects.schema_repair_result import SchemaRepairStatus


def run_audited_schema_repair(
    graph: CompiledStateGraph[SchemaRepairWorkflowState, None, Any, Any],
    initial_state: SchemaRepairWorkflowState,
    *,
    history_store: SchemaMigrationHistoryStore,
    trace_tracer: RepairTraceTracer | None = None,
) -> SchemaRepairWorkflowState:
    """Run `graph` to completion, then record migration history if a repair was attempted.

    `trace_tracer` (e.g. `MlflowRepairTraceTracer`, reused unmodified from
    Tier 1 — its interface has no CSV coupling) is optional and strictly
    best-effort, guaranteeing `graph.invoke` runs exactly once regardless
    of tracing outcome.
    """
    if trace_tracer is not None:
        final_state, trace_id = trace_tracer.trace_invocation(
            lambda: graph.invoke(initial_state), episode_id=initial_state["episode_id"]
        )
        final_state = cast(SchemaRepairWorkflowState, final_state)
    else:
        final_state = cast(SchemaRepairWorkflowState, graph.invoke(initial_state))
        trace_id = None

    status = final_state["status"] or SchemaRepairStatus.FAILED
    if status is SchemaRepairStatus.HEALTHY:
        return final_state  # no drift: nothing to record, mirrors Tier 1

    prescription: SchemaRepairPrescription | None = None
    if final_state["validated_operations"] is not None:
        prescription = SchemaRepairPrescription(
            table=final_state["table"],
            operations=tuple(final_state["validated_operations"]),
            confidence=final_state["confidence"] or 0.0,
        )

    # Real, already-captured usage from MLflow's own OpenAI autolog (see
    # trace_llm_usage) — never fabricated; `None` when tracing is off or
    # this episode's trace has no CHAT_MODEL span (e.g. the healthy or
    # no-rename-candidate paths, which never call the LLM at all).
    token_usage = None
    if trace_id is not None and trace_tracer is not None:
        token_usage = trace_tracer.get_llm_usage(trace_id)

    entry = SchemaMigrationEntry(
        episode_id=final_state["episode_id"],
        table=final_state["table"],
        baseline_version=final_state["baseline"].version if final_state["baseline"] else 0,
        current_schema=final_state["current_schema"],
        diff=final_state["diff"] or ColumnDiff(),
        prescription=prescription,
        confidence=final_state["confidence"],
        status=status,
        applied=status is SchemaRepairStatus.SUCCEEDED,
        human_approved=final_state["human_approved"],
        verification_message=final_state["message"],
        trace_id=trace_id,
        mlflow_run_id=None,
        token_usage=token_usage,
    )
    history_store.record(entry)

    return final_state
