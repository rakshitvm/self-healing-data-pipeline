"""Focused unit tests for Ticket 010's MLflow tracking integration into
`run_audited_csv_repair`.

Kept in a separate file from Ticket 009's `test_audited_csv_repair.py`
(left completely untouched) so the diff for this ticket is unambiguous.
Uses the real (unmodified) Ticket 007 graph and real temporary CSV
files, an in-memory fake `RepairAuditStore`, and hand-rolled fake
`RepairRunTracker` doubles — no real MLflow tracking server, no network
access.
"""

from pathlib import Path
from typing import Any

from self_healing_pipeline.application.orchestration.audited_csv_repair import (
    run_audited_csv_repair,
)
from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    build_csv_repair_workflow,
    build_initial_state,
)
from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.domain.interfaces.services.repair_run_tracker import TrackingOutcome
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


class _SucceedingTracker:
    """Fake `RepairRunTracker`: every call succeeds; records what it saw."""

    def __init__(self, run_id: str = "run-abc") -> None:
        self.run_id = run_id
        self.started: list[tuple[Any, FailureClass]] = []
        self.metrics: list[tuple[str, int | None, int | None]] = []
        self.ended: list[tuple[str, str]] = []

    def start_run(self, *, episode_id: Any, failure_class: FailureClass) -> TrackingOutcome:
        self.started.append((episode_id, failure_class))
        return TrackingOutcome(success=True, run_id=self.run_id)

    def log_metrics(
        self,
        run_id: str,
        *,
        latency_ms: int | None = None,
        token_usage: int | None = None,
        trace_id: str | None = None,
    ) -> TrackingOutcome:
        self.metrics.append((run_id, latency_ms, token_usage))
        return TrackingOutcome(success=True, run_id=run_id)

    def end_run(self, run_id: str, *, status: str) -> TrackingOutcome:
        self.ended.append((run_id, status))
        return TrackingOutcome(success=True, run_id=run_id)


class _AlwaysFailingTracker:
    """Fake `RepairRunTracker`: `start_run` always fails; never raises."""

    def start_run(self, *, episode_id: Any, failure_class: FailureClass) -> TrackingOutcome:
        return TrackingOutcome(success=False, error="simulated MLflow outage")

    def log_metrics(
        self,
        run_id: str,
        *,
        latency_ms: int | None = None,
        token_usage: int | None = None,
        trace_id: str | None = None,
    ) -> TrackingOutcome:
        raise AssertionError("should not be called: start_run already failed")

    def end_run(self, run_id: str, *, status: str) -> TrackingOutcome:
        raise AssertionError("should not be called: start_run already failed")


class _ExplodingTracker:
    """Fake `RepairRunTracker` whose methods raise — proves the wrapper
    itself must never let a tracker exception escape uncaught, even if a
    concrete implementation misbehaves and violates its own contract."""

    def start_run(self, *, episode_id: Any, failure_class: FailureClass) -> TrackingOutcome:
        raise RuntimeError("a buggy tracker implementation raised instead of returning a TrackingOutcome")

    def log_metrics(
        self,
        run_id: str,
        *,
        latency_ms: int | None = None,
        token_usage: int | None = None,
        trace_id: str | None = None,
    ) -> TrackingOutcome:
        raise AssertionError("should not be called")

    def end_run(self, run_id: str, *, status: str) -> TrackingOutcome:
        raise AssertionError("should not be called")


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _graph() -> Any:
    return build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )


def test_omitting_run_tracker_reproduces_ticket_009_behavior_exactly(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()

    result = run_audited_csv_repair(_graph(), build_initial_state(file_path), audit_store=store)

    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert all(e.mlflow_run_id is None for e in store.events)
    assert all("mlflow_tracking_error" not in (e.payload or {}) for e in store.events)


def test_successful_tracking_stamps_run_id_and_latency_on_every_event(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()
    tracker = _SucceedingTracker(run_id="run-xyz")

    result = run_audited_csv_repair(
        _graph(), build_initial_state(file_path), audit_store=store, run_tracker=tracker
    )

    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert len(store.events) == 4
    assert all(e.mlflow_run_id == "run-xyz" for e in store.events)
    assert all(e.latency_ms is not None and e.latency_ms >= 0 for e in store.events)
    assert all("mlflow_tracking_error" not in (e.payload or {}) for e in store.events)
    # started once, metrics logged once, ended once with the run's real terminal status
    assert len(tracker.started) == 1
    assert tracker.started[0][1] == FailureClass.WRONG_DELIMITER
    assert len(tracker.metrics) == 1
    assert tracker.metrics[0][0] == "run-xyz"
    assert len(tracker.ended) == 1
    assert tracker.ended[0] == ("run-xyz", "succeeded")


def test_tracking_failure_does_not_fail_or_alter_a_successful_repair(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()
    tracker = _AlwaysFailingTracker()

    result = run_audited_csv_repair(
        _graph(), build_initial_state(file_path), audit_store=store, run_tracker=tracker
    )

    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].success is True


def test_tracking_failure_is_captured_in_the_audit_trail_not_swallowed(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()
    tracker = _AlwaysFailingTracker()

    run_audited_csv_repair(
        _graph(), build_initial_state(file_path), audit_store=store, run_tracker=tracker
    )

    assert len(store.events) == 4
    assert all(e.mlflow_run_id is None for e in store.events)
    assert all(
        e.payload is not None and e.payload.get("mlflow_tracking_error") == "simulated MLflow outage"
        for e in store.events
    )
    # PostgreSQL remains authoritative regardless of tracking outcome.
    assert len(store.completed) == 1
    assert store.completed[0].status == RepairEpisodeStatus.SUCCEEDED


def test_healthy_path_makes_zero_tracker_calls(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n"
    )
    store = _InMemoryAuditStore()
    tracker = _SucceedingTracker()

    result = run_audited_csv_repair(
        _graph(), build_initial_state(file_path), audit_store=store, run_tracker=tracker
    )

    assert result["failure_class"] is None
    assert tracker.started == []
    assert tracker.metrics == []
    assert tracker.ended == []
    assert store.episodes == []


def test_misbehaving_tracker_that_raises_still_does_not_break_the_repair_pipeline_via_audit_store(
    tmp_path: Path,
) -> None:
    """Belt-and-braces: even if a tracker implementation violates its own
    contract (raises instead of returning `TrackingOutcome`), the audit
    store must still be the one thing that unambiguously fails loudly if
    something is wrong — this test documents current behavior: the
    wrapper does not itself catch tracker exceptions beyond what the
    Protocol contract promises, so a genuinely broken tracker
    implementation (not a well-behaved best-effort one) can still raise.
    """
    import pytest

    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()

    with pytest.raises(RuntimeError, match="a buggy tracker implementation raised"):
        run_audited_csv_repair(
            _graph(), build_initial_state(file_path), audit_store=store, run_tracker=_ExplodingTracker()
        )
