"""Focused unit tests for `run_audited_csv_repair`.

Uses the real (unmodified) Ticket 007 `csv_repair_workflow` graph, the
real `LocalCsvFailureDetector` and `PandasCsvRepairExecutor` against real
temporary CSV files, and an in-memory fake `RepairAuditStore` — no real
PostgreSQL, no network access.
"""

from pathlib import Path
from typing import Any

import pytest

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


class _AlwaysApproveCsvHumanApprovalPort:
    """Fake `CsvHumanApprovalPort`: always approves — every non-healthy
    repair now requires explicit human approval before apply."""

    def request_approval(self, request: Any) -> bool:
        return True


class _InMemoryAuditStore:
    """Simple in-memory `RepairAuditStore` fake, for tests."""

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


class _RaisingAuditStore:
    """`RepairAuditStore` fake whose `start_episode` always fails."""

    def start_episode(self, episode: RepairEpisode) -> None:
        raise RuntimeError("simulated audit persistence failure")

    def record_event(self, event: RepairEvent) -> None:
        raise AssertionError("should not be reached")

    def complete_episode(self, episode: RepairEpisode) -> None:
        raise AssertionError("should not be reached")


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_healthy_path_does_not_create_a_repair_episode(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n")
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )
    store = _InMemoryAuditStore()

    result = run_audited_csv_repair(graph, build_initial_state(file_path), audit_store=store)

    assert result["failure_class"] is None
    assert store.episodes == []
    assert store.events == []
    assert store.completed == []


def test_successful_repair_attempt_is_recorded(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )
    store = _InMemoryAuditStore()

    result = run_audited_csv_repair(graph, build_initial_state(file_path), audit_store=store)

    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert len(store.episodes) == 1
    assert store.episodes[0].episode_id == result["episode_id"]
    assert len(store.events) == 5  # propose, validate, human_approval, apply, reverify
    nodes = [e.node for e in store.events]
    assert nodes == ["propose", "validate", "human_approval", "apply", "reverify"]
    assert all(e.episode_id == result["episode_id"] for e in store.events)
    assert len(store.completed) == 1
    assert store.completed[0].status == RepairEpisodeStatus.SUCCEEDED
    assert store.completed[0].completed_at is not None
    assert store.completed[0].is_active is False


def test_failed_repair_attempt_is_recorded_with_terminal_failure(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    invalid_payload = {"delimiter": "too-long", "encoding": "utf-8"}
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(invalid_payload),
    )
    store = _InMemoryAuditStore()

    result = run_audited_csv_repair(
        graph, build_initial_state(file_path, max_retries=0), audit_store=store
    )

    assert result["status"] == RepairEpisodeStatus.FAILED
    assert len(store.episodes) == 1
    events_by_node = {e.node: e for e in store.events}
    assert events_by_node["validate"].status == "invalid"
    assert "apply" not in events_by_node  # invalid proposal never reached apply
    assert len(store.completed) == 1
    assert store.completed[0].status == RepairEpisodeStatus.FAILED


def test_event_payloads_carry_json_safe_structured_context(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )
    store = _InMemoryAuditStore()

    run_audited_csv_repair(graph, build_initial_state(file_path), audit_store=store)

    apply_event = next(e for e in store.events if e.node == "apply")
    assert apply_event.payload is not None
    assert "message" in apply_event.payload
    assert "validation_errors" in apply_event.payload
    assert apply_event.prescription is not None
    assert apply_event.prescription.delimiter == ";"


def test_persistence_failure_is_not_swallowed(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )

    with pytest.raises(RuntimeError, match="simulated audit persistence failure"):
        run_audited_csv_repair(
            graph, build_initial_state(file_path), audit_store=_RaisingAuditStore()
        )


def test_underlying_workflow_result_is_unchanged_by_auditing(tmp_path: Path) -> None:
    """The wrapper must not alter what the (unmodified) Ticket 007 graph produced."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )
    initial_state = build_initial_state(file_path, episode_id=None)

    direct_result = graph.invoke(dict(initial_state))
    audited_result = run_audited_csv_repair(
        graph,
        build_initial_state(file_path, episode_id=initial_state["episode_id"]),
        audit_store=_InMemoryAuditStore(),
    )

    assert direct_result["status"] == audited_result["status"]
    assert direct_result["repair_result"] == audited_result["repair_result"]
