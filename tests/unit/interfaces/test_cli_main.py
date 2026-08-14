"""Focused unit tests for the Tier 1 CLI entry point (`interfaces/cli/main.py`).

`build_error_router` (pure dependency injection) is tested directly with
fakes. The `repair` command itself is tested via `click.testing.CliRunner`,
with `build_production_error_router` patched so no real Postgres, MLflow,
or LLM provider is ever touched — no real infrastructure or network
access is required by any test here.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

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
from self_healing_pipeline.interfaces.cli.main import build_error_router, cli

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


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_build_error_router_resolves_the_graph_backed_handler() -> None:
    """PART G item 1: ErrorRouter resolves the graph-backed CSV repair handler."""
    router = build_error_router(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        audit_store=_InMemoryAuditStore(),
    )

    resolved = router.resolve(WrongDelimiterError("x", file_path="/tmp/x.csv"))

    assert isinstance(resolved, LangGraphCsvRepairAgent)


def test_end_to_end_wiring_with_fakes_no_external_services(tmp_path: Path) -> None:
    """PART G item 13: full production wiring, provably reachable with
    fakes only — no real Postgres, MLflow, or LLM provider."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    audit_store = _InMemoryAuditStore()
    router = build_error_router(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        audit_store=audit_store,
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )

    result = router.route(WrongDelimiterError("bad delimiter", file_path=file_path))

    assert result.success is True
    assert len(audit_store.episodes) == 1
    assert len(audit_store.completed) == 1


def test_cli_healthy_csv_makes_zero_router_calls(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n")
    runner = CliRunner()

    with patch("self_healing_pipeline.interfaces.cli.main.build_production_error_router") as mock_build:
        result = runner.invoke(cli, ["repair", file_path])

    assert result.exit_code == 0
    assert "HEALTHY" in result.output
    mock_build.assert_not_called()


def test_cli_broken_csv_invokes_the_production_router_wiring(tmp_path: Path) -> None:
    """`build_production_error_router` is patched to return a fake-backed
    router instead of constructing real infrastructure — proves the CLI
    command itself correctly calls it and reports the result, without
    requiring any real Postgres/MLflow/LLM credentials."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    fake_router = build_error_router(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        audit_store=_InMemoryAuditStore(),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )
    runner = CliRunner()

    with patch(
        "self_healing_pipeline.interfaces.cli.main.build_production_error_router",
        return_value=fake_router,
    ) as mock_build:
        result = runner.invoke(cli, ["repair", file_path])

    assert result.exit_code == 0
    assert "success: True" in result.output
    assert "failure_class: wrong_delimiter" in result.output
    mock_build.assert_called_once()
