"""Focused unit tests for Ticket 013's tracing wiring in the CLI
composition root.

Kept separate from Ticket 011's `test_cli_main.py` (left untouched).
`build_production_error_router` is never called for real here (it would
construct live Postgres/MLflow/LLM infrastructure) — only patched to
verify wiring, and `build_error_router` (pure DI) is exercised with fakes.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.domain.exceptions.csv_errors import WrongDelimiterError
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
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
    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        return dict(VALID_PROPOSAL)


class _AlwaysApproveCsvHumanApprovalPort:
    """Fake `CsvHumanApprovalPort`: always approves — every non-healthy
    repair now requires explicit human approval before apply."""

    def request_approval(self, request: Any) -> bool:
        return True


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

    def trace_invocation(self, invoke: Any, *, episode_id: Any) -> tuple[Any, str | None]:
        self.invocations += 1
        return invoke(), "tr-fake"

    def tag_trace(self, trace_id: str, tags: dict[str, str]) -> None:
        pass

    def get_llm_usage(self, trace_id: str) -> dict[str, Any] | None:
        return None


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_build_error_router_threads_trace_tracer_through(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    store = _InMemoryAuditStore()
    tracer = _FakeTracer()
    router = build_error_router(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(),
        audit_store=store,
        trace_tracer=tracer,
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )

    result = router.route(WrongDelimiterError("bad delimiter", file_path=file_path))

    assert result.success is True
    assert tracer.invocations == 1


def test_build_production_error_router_calls_enable_tracing_and_builds_trace_tracer() -> None:
    with (
        patch("self_healing_pipeline.interfaces.cli.main.get_settings") as mock_get_settings,
        patch("self_healing_pipeline.interfaces.cli.main.enable_tracing") as mock_enable_tracing,
        patch("self_healing_pipeline.interfaces.cli.main.PostgresRepairAuditStore"),
        patch("self_healing_pipeline.interfaces.cli.main.MlflowRepairRunTracker"),
        patch("self_healing_pipeline.interfaces.cli.main.MlflowRepairTraceTracer") as mock_tracer_cls,
        patch("self_healing_pipeline.interfaces.cli.main.build_proposal_provider"),
    ):
        from self_healing_pipeline.interfaces.cli.main import build_production_error_router

        build_production_error_router()

    mock_enable_tracing.assert_called_once_with(mock_get_settings.return_value.mlflow)
    mock_tracer_cls.assert_called_once()


def test_cli_repair_command_calls_flush_traces_after_routing(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    fake_router = build_error_router(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(),
        audit_store=_InMemoryAuditStore(),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )
    runner = CliRunner()

    with (
        patch(
            "self_healing_pipeline.interfaces.cli.main.build_production_error_router",
            return_value=fake_router,
        ),
        patch("self_healing_pipeline.interfaces.cli.main.flush_traces") as mock_flush,
    ):
        result = runner.invoke(cli, ["repair", file_path])

    assert result.exit_code == 0
    mock_flush.assert_called_once()


def test_cli_healthy_path_never_calls_flush_traces(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n"
    )
    runner = CliRunner()

    with patch("self_healing_pipeline.interfaces.cli.main.flush_traces") as mock_flush:
        result = runner.invoke(cli, ["repair", file_path])

    assert result.exit_code == 0
    assert "HEALTHY" in result.output
    mock_flush.assert_not_called()
