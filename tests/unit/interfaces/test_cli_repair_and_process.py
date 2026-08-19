"""Tests for the new, additive `repair-and-process` CLI command.

Mirrors `test_cli_main.py`'s pattern exactly: `build_production_error_router`
is patched with a fake-backed router so no real Postgres/MLflow/LLM
infrastructure is touched. The most safety-critical property tested here
is that `repair-and-process` behaves identically to the pre-existing
`repair` command (same output shape for the shared fields, same exit
code) whenever no AZURE_STORAGE_*/DATABRICKS_* configuration is present —
this is what guarantees the existing `repair` command's users are wholly
unaffected by this addition.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.interfaces.cli.main import build_error_router, cli

VALID_PROPOSAL = {"delimiter": ";", "encoding": "utf-8", "header_row": 0, "engine": "python"}
WRONG_DELIMITER_CSV = "id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n"

_CLOUD_ENV_VARS = (
    "AZURE_STORAGE_CONTAINER_URL",
    "AZURE_STORAGE_SAS_TOKEN",
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_JOB_ID",
    "DATABRICKS_NOTEBOOK_PARAM_NAME",
)


@pytest.fixture(autouse=True)
def _no_cloud_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this file runs with zero cloud configuration present,
    regardless of the developer's local `.env` — this is what proves the
    "unconfigured" code path, not whatever happens to be in the shell."""
    for var in _CLOUD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class _FakeProposalPort:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        return dict(self._payload)


class _AlwaysApproveCsvHumanApprovalPort:
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


def _fake_router() -> Any:
    return build_error_router(
        detector=__import__(
            "self_healing_pipeline.infrastructure.csv.local_csv_failure_detector",
            fromlist=["LocalCsvFailureDetector"],
        ).LocalCsvFailureDetector(),
        executor=__import__(
            "self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor",
            fromlist=["PandasCsvRepairExecutor"],
        ).PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        audit_store=_InMemoryAuditStore(),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )


def test_repair_and_process_healthy_csv_makes_zero_router_calls(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n")
    runner = CliRunner()

    with patch("self_healing_pipeline.interfaces.cli.main.build_production_error_router") as mock_build:
        result = runner.invoke(cli, ["repair-and-process", file_path])

    assert result.exit_code == 0
    assert "HEALTHY" in result.output
    mock_build.assert_not_called()


def test_repair_and_process_without_cloud_config_matches_repair_exit_code_and_fields(
    tmp_path: Path,
) -> None:
    """The core safety property: with no cloud configuration present,
    `repair-and-process` reports the same repair outcome and exit code as
    the pre-existing `repair` command, plus an explicit note that the
    optional cloud step was skipped."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    runner = CliRunner()

    with patch(
        "self_healing_pipeline.interfaces.cli.main.build_production_error_router",
        side_effect=_fake_router,
    ):
        repair_result = runner.invoke(cli, ["repair", file_path])

    with patch(
        "self_healing_pipeline.interfaces.cli.main.build_production_error_router",
        side_effect=_fake_router,
    ):
        combined_result = runner.invoke(cli, ["repair-and-process", file_path])

    assert repair_result.exit_code == 0
    assert combined_result.exit_code == repair_result.exit_code
    assert "success: True" in combined_result.output
    assert "failure_class: wrong_delimiter" in combined_result.output
    assert "Azure Storage is not configured" in combined_result.output
    assert "Databricks is not configured" in combined_result.output
    assert "uploaded: False" in combined_result.output
    assert "databricks_triggered: False" in combined_result.output


def test_repair_and_process_reports_manual_action_when_repair_itself_fails(
    tmp_path: Path,
) -> None:
    """When the repair itself fails, `repair-and-process` must exit 1 —
    identical semantics to `repair` — and must not attempt the cloud step
    at all (there is no output_path to process)."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)

    class _RejectingApprovalPort:
        def request_approval(self, request: Any) -> bool:
            return False

    def _failing_router() -> Any:
        return build_error_router(
            detector=__import__(
                "self_healing_pipeline.infrastructure.csv.local_csv_failure_detector",
                fromlist=["LocalCsvFailureDetector"],
            ).LocalCsvFailureDetector(),
            executor=__import__(
                "self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor",
                fromlist=["PandasCsvRepairExecutor"],
            ).PandasCsvRepairExecutor(),
            llm_port=_FakeProposalPort(VALID_PROPOSAL),
            audit_store=_InMemoryAuditStore(),
            approval_port=_RejectingApprovalPort(),
        )

    runner = CliRunner()
    with patch(
        "self_healing_pipeline.interfaces.cli.main.build_production_error_router",
        side_effect=_failing_router,
    ):
        result = runner.invoke(cli, ["repair-and-process", file_path])

    assert result.exit_code == 1
    assert "cloud_integration: skipped" in result.output
