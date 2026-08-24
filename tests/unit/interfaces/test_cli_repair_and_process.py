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
from self_healing_pipeline.infrastructure.cloud.cloud_integration_service import (
    CloudIntegrationService,
)
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

_UNCONFIGURED_NOTES = [
    "Azure Storage is not configured "
    "(AZURE_STORAGE_CONTAINER_URL/AZURE_STORAGE_SAS_TOKEN) — cloud upload skipped.",
    "Databricks is not configured "
    "(DATABRICKS_HOST/DATABRICKS_TOKEN/DATABRICKS_JOB_ID) — job trigger skipped.",
]


@pytest.fixture(autouse=True)
def _no_cloud_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this file runs with zero cloud configuration present,
    deterministically — regardless of the developer's real `.env` file.
    Deleting the env vars alone is not sufficient: `pydantic-settings`
    reads the `.env` file directly (independent of `os.environ`), so if
    real Azure/Databricks credentials are present there (as they now are,
    once the integration is actually configured for a live run),
    `load_azure_storage_settings()`/`load_databricks_settings()` would
    still succeed. `_build_cloud_integration_service` is patched directly
    so this file's "unconfigured" tests prove the CLI's own behavior, not
    whatever happens to be in the shell or the real `.env` file."""
    for var in _CLOUD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "self_healing_pipeline.interfaces.cli.main._build_cloud_integration_service",
        lambda: (CloudIntegrationService(uploader=None, trigger=None), list(_UNCONFIGURED_NOTES)),
    )


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


class _RecordingCloudIntegrationService:
    """Stands in for `CloudIntegrationService`: records the
    `local_file_path` it was called with and returns a canned outcome —
    never touches real Azure/Databricks."""

    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome
        self.calls: list[str] = []

    def process(self, *, local_file_path: str) -> Any:
        self.calls.append(local_file_path)
        return self._outcome


def _success_cloud_outcome() -> Any:
    from self_healing_pipeline.domain.interfaces.services.cloud_integration_port import (
        CloudIntegrationOutcome,
    )

    return CloudIntegrationOutcome(
        uploaded=True,
        cloud_path="wasbs://c@acct.blob.core.windows.net/x.csv",
        triggered=True,
        databricks_run_id=42,
    )


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
    """No Tier 1/Tier 2 infrastructure (router, Postgres, LLM) is ever
    touched for a healthy file — only the (here, unconfigured per the
    autouse fixture) cloud step runs."""
    file_path = _write(tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n")
    runner = CliRunner()

    with patch("self_healing_pipeline.interfaces.cli.main.build_production_error_router") as mock_build:
        result = runner.invoke(cli, ["repair-and-process", file_path])

    assert result.exit_code == 0
    assert "HEALTHY" in result.output
    mock_build.assert_not_called()
    assert "Azure Storage is not configured" in result.output
    assert "uploaded: False" in result.output


def test_repair_and_process_healthy_csv_uploads_original_file_and_triggers_databricks(
    tmp_path: Path,
) -> None:
    """The focused behavior this task adds: a healthy file (no repair
    needed) still uploads the ORIGINAL `file_path` to Azure and triggers
    Databricks, using a mocked `CloudIntegrationService` — no real Azure
    or Databricks involved."""
    file_path = _write(tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n")
    cloud_service = _RecordingCloudIntegrationService(_success_cloud_outcome())
    runner = CliRunner()

    with (
        patch(
            "self_healing_pipeline.interfaces.cli.main.build_production_error_router"
        ) as mock_build,
        patch(
            "self_healing_pipeline.interfaces.cli.main._build_cloud_integration_service",
            return_value=(cloud_service, []),
        ),
    ):
        result = runner.invoke(cli, ["repair-and-process", file_path])

    assert result.exit_code == 0
    assert "HEALTHY" in result.output
    mock_build.assert_not_called()  # still zero Tier 1/2 infra touched
    assert cloud_service.calls == [file_path]  # the ORIGINAL file, not a repaired one
    assert "uploaded: True" in result.output
    assert "databricks_triggered: True" in result.output


def test_repair_and_process_healthy_csv_emits_structured_log_event(tmp_path: Path) -> None:
    """The observability-gap fix: a healthy `repair-and-process` execution
    (no Tier 1 failure, no `--table`) must still emit one structured
    `repair_completed` log event through the project's existing logger —
    previously this branch returned without logging anything at all.
    No LLM/router infrastructure is touched to produce it (proven here by
    `mock_build.assert_not_called()`, matching the sibling healthy tests
    above — this is the "no LLM call for the healthy path" property)."""
    file_path = _write(tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n")
    runner = CliRunner()

    with (
        patch("self_healing_pipeline.interfaces.cli.main.build_production_error_router") as mock_build,
        patch("self_healing_pipeline.interfaces.cli.main.get_logger") as mock_get_logger,
    ):
        result = runner.invoke(cli, ["repair-and-process", file_path])

    assert result.exit_code == 0
    mock_build.assert_not_called()  # no LLM/router infra touched for the healthy path

    mock_get_logger.assert_called_once_with(agent="CsvRepairAgent", node="healthy", table=None)
    bound_logger = mock_get_logger.return_value
    bound_logger.info.assert_called_once_with(
        "repair_completed",
        success=True,
        failure_class=None,
        source_path=file_path,
        message="No CSV failure detected; file already parses correctly. No repair needed.",
    )


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


def test_repair_and_process_tier1_repaired_uploads_the_repaired_output_path(tmp_path: Path) -> None:
    """Distinguishes the repaired case from the healthy case: the
    (mocked) cloud service must be called with `result.output_path` (the
    repaired file), not the original `file_path`."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    cloud_service = _RecordingCloudIntegrationService(_success_cloud_outcome())
    runner = CliRunner()

    with (
        patch(
            "self_healing_pipeline.interfaces.cli.main.build_production_error_router",
            side_effect=_fake_router,
        ),
        patch(
            "self_healing_pipeline.interfaces.cli.main._build_cloud_integration_service",
            return_value=(cloud_service, []),
        ),
    ):
        result = runner.invoke(cli, ["repair-and-process", file_path])

    assert result.exit_code == 0
    assert "success: True" in result.output
    assert "applied: True" in result.output
    assert len(cloud_service.calls) == 1
    assert cloud_service.calls[0] != file_path  # repaired output, not the original
    assert cloud_service.calls[0].endswith("wrong_delimiter.csv")  # Tier 1's repaired/<name>
    assert "uploaded: True" in result.output
    assert "databricks_triggered: True" in result.output


def test_repair_and_process_failed_repair_never_reaches_cloud_integration(tmp_path: Path) -> None:
    """Distinct from *rejected*: here the proposal repeatedly fails
    `CsvRepairParams` validation (an LLM-quality failure, not a human
    decision) and the workflow exhausts its retries — `human_approval`
    is approving throughout, so this proves the FAILED path specifically,
    not the REJECTED path already covered elsewhere in this file."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    invalid_proposal = {"delimiter": "too-long-invalid"}  # never a valid CsvRepairParams

    def _always_invalid_router() -> Any:
        return build_error_router(
            detector=__import__(
                "self_healing_pipeline.infrastructure.csv.local_csv_failure_detector",
                fromlist=["LocalCsvFailureDetector"],
            ).LocalCsvFailureDetector(),
            executor=__import__(
                "self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor",
                fromlist=["PandasCsvRepairExecutor"],
            ).PandasCsvRepairExecutor(),
            llm_port=_FakeProposalPort(invalid_proposal),
            audit_store=_InMemoryAuditStore(),
            approval_port=_AlwaysApproveCsvHumanApprovalPort(),
        )

    cloud_service = _RecordingCloudIntegrationService(_success_cloud_outcome())
    runner = CliRunner()

    with (
        patch(
            "self_healing_pipeline.interfaces.cli.main.build_production_error_router",
            side_effect=_always_invalid_router,
        ),
        patch(
            "self_healing_pipeline.interfaces.cli.main._build_cloud_integration_service",
            return_value=(cloud_service, []),
        ),
    ):
        result = runner.invoke(cli, ["repair-and-process", file_path])

    assert result.exit_code == 1
    assert "success: False" in result.output
    assert "cloud_integration: skipped" in result.output
    assert cloud_service.calls == []


def test_repair_and_process_wrong_encoding_still_works(tmp_path: Path) -> None:
    """Tier 1 regression B: wrong encoding, `--table` omitted — a
    genuinely correct proposal (matching the proven-correct fixture in
    `test_pandas_csv_repair_executor.py::test_wrong_encoding_is_repaired`)
    lets the real `PandasCsvRepairExecutor` actually succeed."""
    path = tmp_path / "wrong_encoding.csv"
    path.write_bytes("id,name,value\n1,café,10\n2,naïve,20\n3,façade,30\n".encode("latin-1"))
    encoding_proposal = {"delimiter": ",", "encoding": "latin-1", "header_row": 0, "engine": "python"}

    def _encoding_router() -> Any:
        return build_error_router(
            detector=__import__(
                "self_healing_pipeline.infrastructure.csv.local_csv_failure_detector",
                fromlist=["LocalCsvFailureDetector"],
            ).LocalCsvFailureDetector(),
            executor=__import__(
                "self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor",
                fromlist=["PandasCsvRepairExecutor"],
            ).PandasCsvRepairExecutor(),
            llm_port=_FakeProposalPort(encoding_proposal),
            audit_store=_InMemoryAuditStore(),
            approval_port=_AlwaysApproveCsvHumanApprovalPort(),
        )

    runner = CliRunner()
    with patch(
        "self_healing_pipeline.interfaces.cli.main.build_production_error_router",
        side_effect=_encoding_router,
    ):
        result = runner.invoke(cli, ["repair-and-process", str(path)])

    assert result.exit_code == 0
    assert "failure_class: wrong_encoding" in result.output
    assert "success: True" in result.output


def test_repair_and_process_single_column_malformation_still_works(tmp_path: Path) -> None:
    """Tier 1 regression C: single-column malformation, `--table` omitted.

    Variable-width whitespace genuinely cannot be repaired by a
    single-character delimiter proposal against the real
    `PandasCsvRepairExecutor` (see
    `test_csv_repair_workflow.py::test_single_column_malformation_flows_through_workflow`'s
    own docstring on this point — a pre-existing Tier 1 characteristic,
    not something this change affects). What matters here is proven the
    same way `test_repair_and_process_without_cloud_config_matches_repair_exit_code_and_fields`
    already proves it for `wrong_delimiter`: `repair-and-process`'s
    detection, routing, and exit code are identical to plain `repair`'s
    for the same file and router — untouched by the Tier 2 addition."""
    file_path = _write(
        tmp_path,
        "single_column.csv",
        "id  name    value\n1   alpha   10\n2  beta     20\n3    gamma  30\n",
    )
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

    assert "failure_class: single_column_malformation" in combined_result.output
    assert combined_result.exit_code == repair_result.exit_code


def test_repair_and_process_tier1_failure_takes_priority_over_table_option(tmp_path: Path) -> None:
    """A Tier 1 parse failure is handled first even when `--table` is also
    given — a file that doesn't even parse correctly is never diffed
    against a schema baseline. Proves the Tier 2 agent builder is never
    constructed in this case (no wasted Postgres/LLM construction)."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    runner = CliRunner()

    with (
        patch(
            "self_healing_pipeline.interfaces.cli.main.build_production_error_router",
            side_effect=_fake_router,
        ),
        patch(
            "self_healing_pipeline.interfaces.cli.main._build_production_schema_repair_agent"
        ) as mock_build_schema_agent,
    ):
        result = runner.invoke(
            cli, ["repair-and-process", file_path, "--table", "customers_nochange"]
        )

    assert result.exit_code == 0
    assert "failure_class: wrong_delimiter" in result.output
    mock_build_schema_agent.assert_not_called()


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
    with (
        patch(
            "self_healing_pipeline.interfaces.cli.main.build_production_error_router",
            side_effect=_failing_router,
        ),
        patch(
            "self_healing_pipeline.interfaces.cli.main._build_cloud_integration_service"
        ) as mock_cloud_service,
    ):
        result = runner.invoke(cli, ["repair-and-process", file_path])

    assert result.exit_code == 1
    assert "cloud_integration: skipped" in result.output
    mock_cloud_service.assert_not_called()
