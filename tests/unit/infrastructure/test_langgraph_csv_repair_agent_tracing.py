"""Focused unit tests for Ticket 013's `trace_tracer` param on
`LangGraphCsvRepairAgent`.

Kept separate from Ticket 011's `test_langgraph_csv_repair_agent.py`
(left untouched). Real graph/detector/executor, fake LLM/audit/tracer.
"""

from pathlib import Path
from typing import Any
from uuid import UUID

from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    build_csv_repair_workflow,
)
from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.domain.exceptions.csv_errors import WrongDelimiterError
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
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

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        return dict(self._payload)


class _InMemoryAuditStore:
    def __init__(self) -> None:
        self.events: list[RepairEvent] = []

    def start_episode(self, episode: RepairEpisode) -> None:
        pass

    def record_event(self, event: RepairEvent) -> None:
        self.events.append(event)

    def complete_episode(self, episode: RepairEpisode) -> None:
        pass


class _FakeTracer:
    def __init__(self) -> None:
        self.invocations = 0

    def trace_invocation(self, invoke: Any, *, episode_id: UUID) -> tuple[Any, str | None]:
        self.invocations += 1
        return invoke(), "tr-fake"

    def tag_trace(self, trace_id: str, tags: dict[str, str]) -> None:
        pass


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_trace_tracer_is_invoked_when_provided(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )
    audit_store = _InMemoryAuditStore()
    tracer = _FakeTracer()
    agent = LangGraphCsvRepairAgent(graph, audit_store=audit_store, trace_tracer=tracer)

    result = agent.handle(WrongDelimiterError("bad delimiter", file_path=file_path))

    assert result.success is True
    assert tracer.invocations == 1
    assert all(e.payload is not None and e.payload.get("mlflow_trace_id") == "tr-fake" for e in audit_store.events)


def test_agent_works_without_a_trace_tracer(tmp_path: Path) -> None:
    """Omitting trace_tracer (the default) must reproduce prior behavior."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )
    audit_store = _InMemoryAuditStore()
    agent = LangGraphCsvRepairAgent(graph, audit_store=audit_store)

    result = agent.handle(WrongDelimiterError("bad delimiter", file_path=file_path))

    assert result.success is True
    assert all("mlflow_trace_id" not in (e.payload or {}) for e in audit_store.events)
