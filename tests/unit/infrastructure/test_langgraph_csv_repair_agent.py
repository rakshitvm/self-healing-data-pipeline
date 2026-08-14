"""Focused unit tests for `LangGraphCsvRepairAgent`.

Uses the real `LocalCsvFailureDetector` and `PandasCsvRepairExecutor`
against real temporary CSV files, plus in-file fakes for the LLM
proposal port, audit store, and run tracker — no real Azure/Groq
network calls, no real PostgreSQL, no real MLflow.
"""

from pathlib import Path
from typing import Any

from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    build_csv_repair_workflow,
)
from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.domain.exceptions.csv_errors import WrongDelimiterError
from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.interfaces.agents.repair_agent import RepairAgent
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.repair_result import RepairResult
from self_healing_pipeline.infrastructure.agents.langgraph_csv_repair_agent import (
    LangGraphCsvRepairAgent,
)
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
        self.calls: list[FailureClass] = []

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        self.calls.append(failure_class)
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


class _AlwaysApproveCsvHumanApprovalPort:
    """Fake `CsvHumanApprovalPort`: always approves — every non-healthy
    repair now requires explicit human approval before apply."""

    def request_approval(self, request: Any) -> bool:
        return True


class _AlwaysReportsNoFailureDetector:
    """Fake `CsvFailureDetector` that always reports no failure."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, file_path: str) -> FailureClass | None:
        self.calls += 1
        return None


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_agent_satisfies_repair_agent_protocol() -> None:
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )
    agent = LangGraphCsvRepairAgent(graph, audit_store=_InMemoryAuditStore())

    assert isinstance(agent, RepairAgent)


def test_handler_invokes_the_langgraph_workflow(tmp_path: Path) -> None:
    """Proves `handle()` genuinely drives the canonical StateGraph, not a
    reimplementation: the real detector/executor are exercised on the
    real file."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    llm_port = _FakeProposalPort(VALID_PROPOSAL)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=llm_port,
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )
    agent = LangGraphCsvRepairAgent(graph, audit_store=_InMemoryAuditStore())
    error = WrongDelimiterError("bad delimiter", file_path=file_path)

    result = agent.handle(error)

    assert isinstance(result, RepairResult)
    assert result.success is True
    assert result.prescription is not None
    assert result.prescription.delimiter == ";"
    assert llm_port.calls == [FailureClass.WRONG_DELIMITER]


def test_audited_wrapper_is_invoked(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    audit_store = _InMemoryAuditStore()
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )
    agent = LangGraphCsvRepairAgent(graph, audit_store=audit_store)
    error = WrongDelimiterError("bad delimiter", file_path=file_path)

    agent.handle(error)

    assert len(audit_store.episodes) == 1
    assert len(audit_store.events) == 5  # propose, validate, human_approval, apply, reverify
    assert len(audit_store.completed) == 1


def test_broken_csv_reaches_the_proposal_stage(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_encoding.csv", WRONG_DELIMITER_CSV)
    llm_port = _FakeProposalPort(VALID_PROPOSAL)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(), executor=PandasCsvRepairExecutor(), llm_port=llm_port
    )
    agent = LangGraphCsvRepairAgent(graph, audit_store=_InMemoryAuditStore())
    error = WrongDelimiterError("bad delimiter", file_path=file_path)

    agent.handle(error)

    assert len(llm_port.calls) == 1


def test_no_failure_detected_returns_success_without_touching_audit_store(tmp_path: Path) -> None:
    """Edge case: the routed error claimed a failure, but the workflow's
    own (authoritative) re-diagnosis disagrees. The adapter must not
    fabricate a repair attempt or audit episode for a file that isn't
    actually broken."""
    file_path = _write(tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n")
    llm_port = _FakeProposalPort(VALID_PROPOSAL)
    audit_store = _InMemoryAuditStore()
    always_no_failure_detector = _AlwaysReportsNoFailureDetector()
    graph = build_csv_repair_workflow(
        detector=always_no_failure_detector, executor=PandasCsvRepairExecutor(), llm_port=llm_port
    )
    agent = LangGraphCsvRepairAgent(graph, audit_store=audit_store)
    error = WrongDelimiterError("stale failure signal", file_path=file_path)

    result = agent.handle(error)

    assert result.success is True
    assert result.applied is False
    assert always_no_failure_detector.calls == 1
    assert llm_port.calls == []
    assert audit_store.episodes == []


def test_missing_file_path_returns_unsuccessful_result_without_invoking_the_graph() -> None:
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )
    agent = LangGraphCsvRepairAgent(graph, audit_store=_InMemoryAuditStore())
    error = WrongDelimiterError("bad delimiter")  # file_path defaults to None

    result = agent.handle(error)

    assert result.success is False
    assert result.applied is False
    assert result.validation_errors == ["missing_file_path"]


def test_validation_exhaustion_returns_unsuccessful_result_reflecting_validation_errors(
    tmp_path: Path,
) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    invalid_llm_port = _FakeProposalPort({"delimiter": "too-long", "encoding": "utf-8"})
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=invalid_llm_port,
    )
    agent = LangGraphCsvRepairAgent(graph, audit_store=_InMemoryAuditStore(), max_retries=0)
    error = WrongDelimiterError("bad delimiter", file_path=file_path)

    result = agent.handle(error)

    assert result.success is False
    assert result.applied is False
    assert result.validation_errors  # carries the CsvRepairParams validation error(s)


def test_handle_returns_the_workflows_own_repair_result_object(tmp_path: Path) -> None:
    """The adapter must not rebuild a new RepairResult on the success path
    — it should return the one the workflow's apply node already built."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )
    agent = LangGraphCsvRepairAgent(graph, audit_store=_InMemoryAuditStore())
    error: PipelineError = WrongDelimiterError("bad delimiter", file_path=file_path)

    result = agent.handle(error)

    assert result.message is not None and "rows" in result.message and "columns" in result.message
