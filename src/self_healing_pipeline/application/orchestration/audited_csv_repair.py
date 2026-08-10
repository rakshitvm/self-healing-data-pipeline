"""Audit-recording wrapper around the Tier 1 CSV repair workflow.

Runs the existing, unmodified `csv_repair_workflow` graph (Ticket 007)
and, once it has finished, records what happened through a
`RepairAuditStore`. This is a pure post-processing integration point: no
node inside the workflow itself calls the audit store, and no routing
decision is influenced by it. `ErrorRouter` remains responsible for
routing, the LangGraph workflow remains responsible for orchestration,
and `CsvRepairAgent` / `CsvRepairExecutor` remain responsible for repair
behavior — this module only records what already happened.

A healthy CSV (no `failure_class` detected) never creates an episode:
there was no repair attempt to audit.

Limitation: the workflow's state holds only the *latest* proposal/
validation/repair/verification outcome, not a per-retry history, so a
retried episode is audited by its final attempt's outcome rather than
one event per retry. Recording a full per-retry trail would require
either accumulating history inside the graph's state or calling the
audit store from within individual graph nodes — both are deferred, to
keep this integration point the smallest one that satisfies "every
attempt [episode] is audited" without modifying the workflow itself.
"""

from datetime import datetime, timezone
from typing import Any, cast

from langgraph.graph.state import CompiledStateGraph

from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    CsvRepairWorkflowState,
)
from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.domain.interfaces.repositories.repair_audit_store import (
    RepairAuditStore,
)
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.pipeline_context import PipelineContext
from self_healing_pipeline.domain.value_objects.pipeline_status import RepairEpisodeStatus


def _error_type(failure_class: FailureClass | None) -> str:
    return failure_class.value if failure_class is not None else "unknown"


def _propose_event(state: CsvRepairWorkflowState) -> RepairEvent:
    return RepairEvent(
        episode_id=state["episode_id"],
        node="propose",
        status="proposed",
        error_type=_error_type(state["failure_class"]),
        payload={"raw_proposal": state["proposed_params"]},
    )


def _validate_event(state: CsvRepairWorkflowState) -> RepairEvent:
    validated = state["validated_params"]
    return RepairEvent(
        episode_id=state["episode_id"],
        node="validate",
        status="valid" if validated is not None else "invalid",
        error_type=_error_type(state["failure_class"]),
        prescription=validated,
        payload={"validation_errors": state["validation_errors"]},
    )


def _apply_event(state: CsvRepairWorkflowState) -> RepairEvent:
    result = state["repair_result"]
    assert result is not None  # only called when repair_result was set
    return RepairEvent(
        episode_id=state["episode_id"],
        node="apply",
        status="applied" if result.success else "apply_failed",
        error_type=_error_type(state["failure_class"]),
        applied=result.applied,
        confidence=result.confidence,
        prescription=result.prescription,
        payload={"message": result.message, "validation_errors": result.validation_errors},
    )


def _reverify_event(state: CsvRepairWorkflowState) -> RepairEvent:
    outcome = state["verification_result"]
    assert outcome is not None  # only called when verification_result was set
    return RepairEvent(
        episode_id=state["episode_id"],
        node="reverify",
        status="verified" if outcome.success else "verification_failed",
        error_type=_error_type(state["failure_class"]),
        confidence=outcome.confidence,
        payload={"message": outcome.message, "validation_errors": outcome.validation_errors},
    )


def run_audited_csv_repair(
    graph: CompiledStateGraph[CsvRepairWorkflowState, None, Any, Any],
    initial_state: CsvRepairWorkflowState,
    *,
    audit_store: RepairAuditStore,
    source: str = "local",
) -> CsvRepairWorkflowState:
    """Run `graph` to completion, then record an audit trail if a repair was attempted.

    Returns the workflow's final state unchanged. Persistence failures
    are not caught: if `audit_store` raises, this function raises too.
    """
    final_state = cast(CsvRepairWorkflowState, graph.invoke(initial_state))
    failure_class = final_state["failure_class"]

    if failure_class is None:
        return final_state  # healthy path: no repair attempt, no audit episode

    context = PipelineContext(
        episode_id=final_state["episode_id"],
        source=source,
        file_path=final_state["file_path"],
        failure_class=failure_class,
    )
    episode = RepairEpisode(
        episode_id=final_state["episode_id"],
        pipeline_context=context,
        status=RepairEpisodeStatus.IN_PROGRESS,
    )
    audit_store.start_episode(episode)

    if final_state["proposed_params"] is not None:
        audit_store.record_event(_propose_event(final_state))
    audit_store.record_event(_validate_event(final_state))
    if final_state["repair_result"] is not None:
        audit_store.record_event(_apply_event(final_state))
    if final_state["verification_result"] is not None:
        audit_store.record_event(_reverify_event(final_state))

    now = datetime.now(timezone.utc)
    episode.status = final_state["status"]
    episode.is_active = False
    episode.last_active_at = now
    episode.completed_at = now
    audit_store.complete_episode(episode)

    return final_state
