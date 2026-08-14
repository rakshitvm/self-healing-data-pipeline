"""Focused unit tests for Ticket 013's trace-tracing integration into
`run_audited_csv_repair`.

Kept in a separate file from Ticket 009's `test_audited_csv_repair.py`
and Ticket 010's `test_audited_csv_repair_tracking.py` (both left
completely untouched) so this ticket's diff is unambiguous. Uses the
real (unmodified) Ticket 007 graph and real temporary CSV files, an
in-memory fake `RepairAuditStore`, and hand-rolled fake `RepairTraceTracer`
doubles — no real MLflow tracking server, no network access.
"""

from pathlib import Path
from typing import Any
from uuid import UUID

from self_healing_pipeline.application.orchestration.audited_csv_repair import (
    run_audited_csv_repair,
)
from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    build_csv_repair_workflow,
    build_initial_state,
)
from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.pipeline_status import RepairEpisodeStatus
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
)
from self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor import (
    PandasCsvRepairExecutor,
)

VALID_PROPOSAL = {"delimiter": ";", "encoding": "utf-8", "header_row": 0, "engine": "python"}
WRONG_DELIMITER_CSV = "id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n"


class _FakeProposalPort:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        return dict(self._payload)


class _InMemoryAuditStore:
    def __init__(self) -> None:
        self.episodes: list[RepairEpisode] = []
        self.events: list[RepairEvent] = []
        self.completed: list[RepairEpisode] = []

    def start_episode(self, episode: RepairEpisode) -> None:
        self.episodes.append(episode)

    def record_event(self, event: RepairEvent) -> None:
        self.events.append(event)

    def complete_episode(self, episode: RepairEpisode) -> None:
        self.completed.append(episode)


class _SucceedingTracer:
    """Fake `RepairTraceTracer`: `invoke` runs exactly once; records calls."""

    def __init__(self, trace_id: str = "tr-fake-abc") -> None:
        self.trace_id = trace_id
        self.invocations = 0
        self.episode_ids: list[UUID] = []
        self.tags: list[tuple[str, dict[str, str]]] = []

    def trace_invocation(self, invoke: Any, *, episode_id: UUID) -> tuple[Any, str | None]:
        self.invocations += 1
        self.episode_ids.append(episode_id)
        return invoke(), self.trace_id

    def tag_trace(self, trace_id: str, tags: dict[str, str]) -> None:
        self.tags.append((trace_id, tags))

    def get_llm_usage(self, trace_id: str) -> dict[str, Any] | None:
        return None


class _FailingTracer:
    """Fake `RepairTraceTracer`: tracing always fails; invoke still runs once."""

    def trace_invocation(self, invoke: Any, *, episode_id: UUID) -> tuple[Any, str | None]:
        return invoke(), None

    def tag_trace(self, trace_id: str, tags: dict[str, str]) -> None:
        raise AssertionError("should not be called: trace_id was never produced")

    def get_llm_usage(self, trace_id: str) -> dict[str, Any] | None:
        raise AssertionError("should not be called: trace_id was never produced")


class _ExplodingTracer:
    """Fake `RepairTraceTracer` whose methods raise — documents that the
    wrapper trusts the Protocol's best-effort contract, same as the
    equivalent test for RepairRunTracker in Ticket 010."""

    def trace_invocation(self, invoke: Any, *, episode_id: UUID) -> tuple[Any, str | None]:
        raise RuntimeError("a buggy tracer implementation raised instead of returning a tuple")

    def tag_trace(self, trace_id: str, tags: dict[str, str]) -> None:
        raise AssertionError("should not be called")

    def get_llm_usage(self, trace_id: str) -> dict[str, Any] | None:
        raise AssertionError("should not be called")


class _AlwaysApproveCsvHumanApprovalPort:
    """Fake `CsvHumanApprovalPort`: always approves — every non-healthy
    repair now requires explicit human approval before apply."""

    def request_approval(self, request: Any) -> bool:
        return True


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _graph() -> Any:
    return build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )


def test_omitting_trace_tracer_reproduces_prior_behavior_exactly(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()

    result = run_audited_csv_repair(_graph(), build_initial_state(file_path), audit_store=store)

    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert all("mlflow_trace_id" not in (e.payload or {}) for e in store.events)


def test_successful_tracing_calls_invoke_exactly_once_and_stamps_trace_id(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()
    tracer = _SucceedingTracer(trace_id="tr-xyz")

    result = run_audited_csv_repair(
        _graph(), build_initial_state(file_path), audit_store=store, trace_tracer=tracer
    )

    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert tracer.invocations == 1
    assert len(store.events) == 5  # propose, validate, human_approval, apply, reverify
    assert all(e.payload is not None and e.payload.get("mlflow_trace_id") == "tr-xyz" for e in store.events)


def test_trace_is_tagged_with_failure_class_after_result_is_known(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()
    tracer = _SucceedingTracer(trace_id="tr-xyz")

    run_audited_csv_repair(
        _graph(), build_initial_state(file_path), audit_store=store, trace_tracer=tracer
    )

    assert tracer.tags == [("tr-xyz", {"failure_class": "wrong_delimiter"})]


def test_tracing_failure_does_not_fail_or_alter_a_successful_repair(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()

    result = run_audited_csv_repair(
        _graph(), build_initial_state(file_path), audit_store=store, trace_tracer=_FailingTracer()
    )

    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].success is True
    assert all("mlflow_trace_id" not in (e.payload or {}) for e in store.events)


def test_healthy_path_still_wraps_the_invocation_but_creates_no_audit_episode(
    tmp_path: Path,
) -> None:
    """Unlike `run_tracker` (only ever called *after* `failure_class` is
    known, so genuinely skipped for healthy files), `trace_tracer` must
    wrap `graph.invoke()` itself — diagnosis happens *inside* the graph,
    so there is no way to know in advance whether a given invocation will
    turn out healthy. This is harmless: MLflow's autolog would trace the
    invocation regardless, with or without this wrapper, and the
    "zero LLM calls on the healthy path" guarantee is unaffected (the
    `propose` node, which is the only place an LLM is ever called, is
    still never reached). What *does* stay exactly as before: no audit
    episode, no event, no tag call, since there's nothing to correlate a
    trace with for a file that was never actually repaired.
    """
    file_path = _write(
        tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n"
    )
    store = _InMemoryAuditStore()
    tracer = _SucceedingTracer()

    result = run_audited_csv_repair(
        _graph(), build_initial_state(file_path), audit_store=store, trace_tracer=tracer
    )

    assert result["failure_class"] is None
    assert tracer.invocations == 1
    assert tracer.tags == []  # no failure_class to tag with, so tag_trace is never called
    assert store.episodes == []


def test_misbehaving_tracer_that_raises_still_propagates(tmp_path: Path) -> None:
    import pytest

    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()

    with pytest.raises(RuntimeError, match="a buggy tracer implementation raised"):
        run_audited_csv_repair(
            _graph(), build_initial_state(file_path), audit_store=store, trace_tracer=_ExplodingTracer()
        )


def test_run_tracker_and_trace_tracer_coexist_independently(tmp_path: Path) -> None:
    """Ticket 010's RepairRunTracker and Ticket 013's RepairTraceTracer are
    separate abstractions that must both work together without interfering."""
    from self_healing_pipeline.domain.interfaces.services.repair_run_tracker import TrackingOutcome

    class _FakeRunTracker:
        def __init__(self) -> None:
            self.started = False

        def start_run(self, *, episode_id: UUID, failure_class: FailureClass) -> TrackingOutcome:
            self.started = True
            return TrackingOutcome(success=True, run_id="run-123")

        def log_metrics(
            self,
            run_id: str,
            *,
            latency_ms: int | None = None,
            token_usage: int | None = None,
            trace_id: str | None = None,
        ) -> TrackingOutcome:
            return TrackingOutcome(success=True, run_id=run_id)

        def end_run(self, run_id: str, *, status: str) -> TrackingOutcome:
            return TrackingOutcome(success=True, run_id=run_id)

    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()
    run_tracker = _FakeRunTracker()
    trace_tracer = _SucceedingTracer(trace_id="tr-both")

    run_audited_csv_repair(
        _graph(),
        build_initial_state(file_path),
        audit_store=store,
        run_tracker=run_tracker,
        trace_tracer=trace_tracer,
    )

    assert run_tracker.started is True
    assert trace_tracer.invocations == 1
    assert all(e.mlflow_run_id == "run-123" for e in store.events)
    assert all(e.payload is not None and e.payload.get("mlflow_trace_id") == "tr-both" for e in store.events)
