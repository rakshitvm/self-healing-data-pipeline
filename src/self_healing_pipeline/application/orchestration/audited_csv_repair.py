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

Ticket 010 adds an optional `run_tracker` (`RepairRunTracker`, e.g. an
MLflow-backed implementation) purely as observability, layered on top of
the same post-hoc integration point — no change to the paragraphs above.
MLflow is never the audit-of-record: `audit_store` (PostgreSQL) remains
authoritative regardless of whether tracking succeeds. Tracking is
strictly best-effort — every `RepairRunTracker` call is wrapped so a
tracking failure can never fail, roll back, or invalidate an otherwise
successful repair; if tracking fails, that failure is instead recorded
into the audit trail itself (`payload["mlflow_tracking_error"]` on every
event of the episode), so it is never silently mistaken for success.
Because the workflow itself is not touched, there is no per-node timing
available; `latency_ms` measures the *entire* `graph.invoke()` call and
is attributed identically to every event of the episode, not to a single
node. `token_usage` is left unpopulated: `AzureOpenAIProposalProvider`
(Ticket 008) does not expose the LLM response's token usage through the
fixed `CsvRepairProposalPort.propose() -> dict[str, Any]` contract, and
per Ticket 010's constraints that provider's request/response behavior
must not change to obtain it.

Ticket 013 adds an optional `trace_tracer` (`RepairTraceTracer`, e.g.
`MlflowRepairTraceTracer`) alongside `run_tracker` — a second, separate
MLflow concern (traces/spans, not Runs; see `repair_trace_tracer.py` for
why these are kept as distinct abstractions). When provided, the whole
`graph.invoke()` call is wrapped in one best-effort MLflow trace via
`trace_tracer.trace_invocation`, which *guarantees* `graph.invoke` is
still called exactly once regardless of tracing outcome. Real per-node
child spans, the TOOL span, and the LLM span come entirely from MLflow's
own LangChain/OpenAI autologging (enabled once, at the composition root,
via `enable_tracing` — see `tracing_setup.py`), not from anything in
this module — this module only opens the outer trace and threads the
resulting `trace_id` into `payload["mlflow_trace_id"]` on every event,
mirroring exactly how `mlflow_run_id`/`mlflow_tracking_error` are
already threaded through. Omitting `trace_tracer` reproduces Ticket
010's behavior exactly.

Asymmetry with `run_tracker`, worth being explicit about: `run_tracker`
is only ever invoked *after* `failure_class` is known (so it is
genuinely skipped for healthy files), whereas `trace_tracer` must wrap
`graph.invoke()` itself — diagnosis happens *inside* the graph, so
there is no way to know in advance whether a given invocation will turn
out healthy. This is harmless: MLflow's own autologging would trace the
invocation regardless, with or without this wrapper, and the "zero LLM
calls on the healthy path" guarantee is unaffected (`propose`, the only
node that ever calls an LLM, is still never reached). What stays exactly
the same either way: a healthy file still creates no audit episode, no
event, and no trace tag.
"""

import time
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

from langgraph.graph.state import CompiledStateGraph

from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    CsvRepairWorkflowState,
)
from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.domain.interfaces.repositories.repair_audit_store import (
    RepairAuditStore,
)
from self_healing_pipeline.domain.interfaces.services.repair_run_tracker import RepairRunTracker
from self_healing_pipeline.domain.interfaces.services.repair_trace_tracer import RepairTraceTracer
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.pipeline_context import PipelineContext
from self_healing_pipeline.domain.value_objects.pipeline_status import RepairEpisodeStatus


def _error_type(failure_class: FailureClass | None) -> str:
    return failure_class.value if failure_class is not None else "unknown"


def _tracking_payload(tracking_error: str | None) -> dict[str, Any]:
    return {"mlflow_tracking_error": tracking_error} if tracking_error is not None else {}


def _trace_payload(trace_id: str | None) -> dict[str, Any]:
    return {"mlflow_trace_id": trace_id} if trace_id is not None else {}


def _multi_failure_payload(state: CsvRepairWorkflowState) -> dict[str, Any]:
    """`failure_classes` on every event when a genuine multi-failure
    episode is in progress; omitted entirely for the ordinary
    single-failure case, so existing payload shapes are unchanged."""
    failure_classes = state["failure_classes"]
    if len(failure_classes) <= 1:
        return {}
    return {"failure_classes": sorted(c.value for c in failure_classes)}


class _TrackingContext(TypedDict):
    """The per-episode tracking/tracing info threaded onto every recorded event."""

    mlflow_run_id: str | None
    latency_ms: int | None
    tracking_error: str | None
    trace_id: str | None


def _propose_event(
    state: CsvRepairWorkflowState,
    *,
    mlflow_run_id: str | None,
    latency_ms: int | None,
    tracking_error: str | None,
    trace_id: str | None,
) -> RepairEvent:
    return RepairEvent(
        episode_id=state["episode_id"],
        node="propose",
        status="proposed",
        error_type=_error_type(state["failure_class"]),
        mlflow_run_id=mlflow_run_id,
        latency_ms=latency_ms,
        payload={
            "raw_proposal": state["proposed_params"],
            **_tracking_payload(tracking_error),
            **_trace_payload(trace_id),
            **_multi_failure_payload(state),
        },
    )


def _validate_event(
    state: CsvRepairWorkflowState,
    *,
    mlflow_run_id: str | None,
    latency_ms: int | None,
    tracking_error: str | None,
    trace_id: str | None,
) -> RepairEvent:
    validated = state["validated_params"]
    return RepairEvent(
        episode_id=state["episode_id"],
        node="validate",
        status="valid" if validated is not None else "invalid",
        error_type=_error_type(state["failure_class"]),
        prescription=validated,
        mlflow_run_id=mlflow_run_id,
        latency_ms=latency_ms,
        payload={
            "validation_errors": state["validation_errors"],
            **_tracking_payload(tracking_error),
            **_trace_payload(trace_id),
            **_multi_failure_payload(state),
        },
    )


def _apply_event(
    state: CsvRepairWorkflowState,
    *,
    mlflow_run_id: str | None,
    latency_ms: int | None,
    tracking_error: str | None,
    trace_id: str | None,
) -> RepairEvent:
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
        mlflow_run_id=mlflow_run_id,
        latency_ms=latency_ms,
        payload={
            "message": result.message,
            "validation_errors": result.validation_errors,
            **_tracking_payload(tracking_error),
            **_trace_payload(trace_id),
            **_multi_failure_payload(state),
        },
    )


def _reverify_event(
    state: CsvRepairWorkflowState,
    *,
    mlflow_run_id: str | None,
    latency_ms: int | None,
    tracking_error: str | None,
    trace_id: str | None,
) -> RepairEvent:
    outcome = state["verification_result"]
    assert outcome is not None  # only called when verification_result was set
    return RepairEvent(
        episode_id=state["episode_id"],
        node="reverify",
        status="verified" if outcome.success else "verification_failed",
        error_type=_error_type(state["failure_class"]),
        confidence=outcome.confidence,
        mlflow_run_id=mlflow_run_id,
        latency_ms=latency_ms,
        payload={
            "message": outcome.message,
            "validation_errors": outcome.validation_errors,
            **_tracking_payload(tracking_error),
            **_trace_payload(trace_id),
            **_multi_failure_payload(state),
        },
    )


def _human_approval_event(
    state: CsvRepairWorkflowState,
    *,
    mlflow_run_id: str | None,
    latency_ms: int | None,
    tracking_error: str | None,
    trace_id: str | None,
) -> RepairEvent:
    approved = state["human_approved"]
    return RepairEvent(
        episode_id=state["episode_id"],
        node="human_approval",
        status="approved" if approved else "rejected",
        error_type=_error_type(state["failure_class"]),
        mlflow_run_id=mlflow_run_id,
        latency_ms=latency_ms,
        payload={
            "human_approved": approved,
            **_tracking_payload(tracking_error),
            **_trace_payload(trace_id),
            **_multi_failure_payload(state),
        },
    )


def run_audited_csv_repair(
    graph: CompiledStateGraph[CsvRepairWorkflowState, None, Any, Any],
    initial_state: CsvRepairWorkflowState,
    *,
    audit_store: RepairAuditStore,
    source: str = "local",
    run_tracker: RepairRunTracker | None = None,
    trace_tracer: RepairTraceTracer | None = None,
) -> CsvRepairWorkflowState:
    """Run `graph` to completion, then record an audit trail if a repair was attempted.

    Returns the workflow's final state unchanged. Persistence failures
    are not caught: if `audit_store` raises, this function raises too.
    `run_tracker` and `trace_tracer` are optional and strictly
    best-effort (see module docstring); omitting both reproduces Ticket
    009's audit-only behavior exactly.
    """
    started_at = time.perf_counter()
    if trace_tracer is not None:
        final_state, trace_id = trace_tracer.trace_invocation(
            lambda: graph.invoke(initial_state), episode_id=initial_state["episode_id"]
        )
        final_state = cast(CsvRepairWorkflowState, final_state)
    else:
        final_state = cast(CsvRepairWorkflowState, graph.invoke(initial_state))
        trace_id = None
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    failure_class = final_state["failure_class"]

    if failure_class is None:
        return final_state  # healthy path: no repair attempt, no audit episode, no tracking

    if trace_id is not None and trace_tracer is not None:
        trace_tracer.tag_trace(trace_id, {"failure_class": failure_class.value})

    mlflow_run_id: str | None = None
    tracking_error: str | None = None
    if run_tracker is not None:
        start_outcome = run_tracker.start_run(
            episode_id=final_state["episode_id"], failure_class=failure_class
        )
        if start_outcome.success:
            mlflow_run_id = start_outcome.run_id
            metrics_outcome = run_tracker.log_metrics(
                cast(str, mlflow_run_id), latency_ms=elapsed_ms, token_usage=None
            )
            if not metrics_outcome.success:
                tracking_error = metrics_outcome.error
            end_outcome = run_tracker.end_run(
                cast(str, mlflow_run_id), status=final_state["status"].value
            )
            if not end_outcome.success and tracking_error is None:
                tracking_error = end_outcome.error
        else:
            tracking_error = start_outcome.error

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

    event_kwargs: _TrackingContext = {
        "mlflow_run_id": mlflow_run_id,
        "latency_ms": elapsed_ms,
        "tracking_error": tracking_error,
        "trace_id": trace_id,
    }
    if final_state["proposed_params"] is not None:
        audit_store.record_event(_propose_event(final_state, **event_kwargs))
    audit_store.record_event(_validate_event(final_state, **event_kwargs))
    if final_state["human_approved"] is not None:
        audit_store.record_event(_human_approval_event(final_state, **event_kwargs))
    if final_state["repair_result"] is not None:
        audit_store.record_event(_apply_event(final_state, **event_kwargs))
    if final_state["verification_result"] is not None:
        audit_store.record_event(_reverify_event(final_state, **event_kwargs))

    now = datetime.now(timezone.utc)
    episode.status = final_state["status"]
    episode.is_active = False
    episode.last_active_at = now
    episode.completed_at = now
    audit_store.complete_episode(episode)

    return final_state
