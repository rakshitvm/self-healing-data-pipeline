"""Tests for Tier 2 (schema-drift) participation in the unified
`repair-and-process --table TABLE FILE_PATH` flow.

Uses the REAL `configs/schema_baselines/<table>.json` files already in
this repo (via the real `JsonSchemaBaselineStore`), the real
`PandasSchemaInspector`, and the real `PandasSchemaExecutor` against real
temporary CSV files — genuinely proving the existing baseline JSON is
what drives detection/repair, not a mock. Only the LLM rename-
confirmation port, the human-approval port, and the migration-history
store are faked (mirroring `tests/unit/application/test_schema_repair_workflow.py`'s
established convention) — no real Postgres/MLflow/LLM network calls.
`build_production_error_router` is patched exactly as in
`test_cli_repair_and_process.py` so Tier 1's own real infrastructure is
never touched either. Cloud integration is exercised via a fake
`CloudIntegrationService`-shaped stub (never real Azure/Databricks),
proving only that it *is* (or is not) invoked with the correct
`local_file_path` — the actual HTTP mechanics are already covered by
`tests/unit/infrastructure/test_cloud_integration.py`.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID

import pytest
from click.testing import CliRunner

from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.domain.entities.schema_migration_entry import SchemaMigrationEntry
from self_healing_pipeline.domain.interfaces.services.human_approval_port import ApprovalRequest
from self_healing_pipeline.domain.interfaces.services.rename_confirmation_port import (
    RenameConfirmation,
)
from self_healing_pipeline.domain.value_objects.column_diff import RenameHint
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.agents.langgraph_schema_repair_agent import (
    LangGraphSchemaRepairAgent,
)
from self_healing_pipeline.infrastructure.schema.json_schema_baseline_store import (
    JsonSchemaBaselineStore,
)
from self_healing_pipeline.infrastructure.schema.pandas_schema_executor import PandasSchemaExecutor
from self_healing_pipeline.infrastructure.schema.pandas_schema_inspector import (
    PandasSchemaInspector,
)
from self_healing_pipeline.application.orchestration.schema_repair_workflow import (
    build_schema_repair_workflow,
)
from self_healing_pipeline.interfaces.cli.main import build_error_router, cli

_REAL_BASELINES_DIR = Path(__file__).resolve().parents[3] / "configs" / "schema_baselines"

_CLOUD_ENV_VARS = (
    "AZURE_STORAGE_CONTAINER_URL",
    "AZURE_STORAGE_SAS_TOKEN",
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_JOB_ID",
    "DATABRICKS_NOTEBOOK_PARAM_NAME",
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


class _InMemoryHistoryStore:
    def __init__(self) -> None:
        self.entries: dict[UUID, list[SchemaMigrationEntry]] = {}

    def record(self, entry: SchemaMigrationEntry) -> None:
        self.entries.setdefault(entry.episode_id, []).append(entry)

    def load_episode_history(self, episode_id: UUID) -> tuple[SchemaMigrationEntry, ...]:
        return tuple(self.entries.get(episode_id, []))


class _AlwaysConfirmConfirmationPort:
    def __init__(self, confidence: float = 0.95) -> None:
        self._confidence = confidence
        self.calls = 0

    def confirm(self, *, table: str, hint: RenameHint) -> RenameConfirmation:
        self.calls += 1
        return RenameConfirmation(confirmed=True, llm_confidence=self._confidence)


class _RecordingApprovalPort:
    def __init__(self, approved: bool) -> None:
        self._approved = approved
        self.requests: list[ApprovalRequest] = []

    def request_approval(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self._approved


class _RecordingCloudIntegrationService:
    """Stands in for `CloudIntegrationService`: records the
    `local_file_path` it was called with and returns a canned outcome —
    never touches real Azure/Databricks, mirroring this file's module
    docstring."""

    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome
        self.calls: list[str] = []

    def process(self, *, local_file_path: str) -> Any:
        self.calls.append(local_file_path)
        return self._outcome


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _fake_tier1_router() -> Any:
    """A Tier 1 router that never resolves for these tests' healthy/schema-only
    CSVs (no Tier 1 failure exists in any fixture below), but is wired
    exactly like production so `build_production_error_router` can be
    patched consistently with `test_cli_repair_and_process.py`."""
    return build_error_router(
        detector=__import__(
            "self_healing_pipeline.infrastructure.csv.local_csv_failure_detector",
            fromlist=["LocalCsvFailureDetector"],
        ).LocalCsvFailureDetector(),
        executor=__import__(
            "self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor",
            fromlist=["PandasCsvRepairExecutor"],
        ).PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort({}),
        audit_store=_InMemoryAuditStore(),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )


def _fake_schema_agent(
    *, confirmation_port: Any = None, approval_port: Any = None
) -> LangGraphSchemaRepairAgent:
    """Real baseline store (pointed at the repo's actual
    `configs/schema_baselines/`), real inspector, real executor — only
    the LLM/human/history ports are faked."""
    history_store = _InMemoryHistoryStore()
    graph = build_schema_repair_workflow(
        baseline_store=JsonSchemaBaselineStore(root=_REAL_BASELINES_DIR),
        inspector=PandasSchemaInspector(),
        history_store=history_store,
        confirmation_port=confirmation_port or _AlwaysConfirmConfirmationPort(),
        approval_port=approval_port or _RecordingApprovalPort(approved=True),
        executor=PandasSchemaExecutor(),
    )
    return LangGraphSchemaRepairAgent(graph, history_store=history_store)


@pytest.fixture(autouse=True)
def _no_real_cloud_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _CLOUD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _invoke(
    runner: CliRunner,
    file_path: str,
    table: str,
    *,
    schema_agent: LangGraphSchemaRepairAgent,
    cloud_service: _RecordingCloudIntegrationService,
) -> Any:
    with (
        patch(
            "self_healing_pipeline.interfaces.cli.main.build_production_error_router",
            side_effect=_fake_tier1_router,
        ),
        patch(
            "self_healing_pipeline.interfaces.cli.main._build_production_schema_repair_agent",
            return_value=schema_agent,
        ),
        patch(
            "self_healing_pipeline.interfaces.cli.main._build_cloud_integration_service",
            return_value=(cloud_service, []),
        ),
    ):
        return runner.invoke(cli, ["repair-and-process", file_path, "--table", table])


def _success_cloud_outcome() -> Any:
    from self_healing_pipeline.domain.interfaces.services.cloud_integration_port import (
        CloudIntegrationOutcome,
    )

    return CloudIntegrationOutcome(
        uploaded=True, cloud_path="wasbs://c@acct.blob.core.windows.net/x.csv",
        triggered=True, databricks_run_id=42,
    )


# --- D. no drift (healthy) ---------------------------------------------


def test_no_change_baseline_is_healthy_no_llm_still_uploads_original_file(tmp_path: Path) -> None:
    """Healthy (no drift): zero LLM calls, but the ORIGINAL file is still
    uploaded to Azure and Databricks is still triggered — there is
    nothing to repair, so `output_path` stays `None` and the cloud step
    falls back to the original `file_path`."""
    file_path = _write(tmp_path, "customers.csv", "id,name,age\n1,Alice,30\n2,Bob,25\n")
    confirmation_port = _AlwaysConfirmConfirmationPort()
    schema_agent = _fake_schema_agent(confirmation_port=confirmation_port)
    cloud_service = _RecordingCloudIntegrationService(_success_cloud_outcome())
    runner = CliRunner()

    result = _invoke(
        runner, file_path, "customers_nochange", schema_agent=schema_agent, cloud_service=cloud_service
    )

    assert result.exit_code == 0
    assert "success: True" in result.output
    assert "applied: False" in result.output
    assert confirmation_port.calls == 0
    assert cloud_service.calls == [file_path]  # the original file, since nothing was repaired
    assert "uploaded: True" in result.output
    assert "databricks_triggered: True" in result.output


# --- E. added column -----------------------------------------------------


def test_added_column_is_repaired_and_reaches_cloud_integration(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "customers.csv", "id,name,age,city\n1,Alice,30,Rome\n")
    schema_agent = _fake_schema_agent()
    cloud_service = _RecordingCloudIntegrationService(_success_cloud_outcome())
    runner = CliRunner()

    result = _invoke(
        runner, file_path, "customers_added", schema_agent=schema_agent, cloud_service=cloud_service
    )

    assert result.exit_code == 0
    assert "success: True" in result.output
    assert "applied: True" in result.output
    assert "city" in Path(file_path).read_text(encoding="utf-8").splitlines()[0]  # source untouched
    assert len(cloud_service.calls) == 1
    output_path = cloud_service.calls[0]
    assert output_path != file_path
    assert f"output_path: {output_path}" in result.output
    assert "uploaded: True" in result.output
    assert "databricks_triggered: True" in result.output
    assert "city" not in Path(output_path).read_text(encoding="utf-8").splitlines()[0]


# --- F. removed column ----------------------------------------------------


def test_removed_column_is_repaired_and_reaches_cloud_integration(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "customers.csv", "id,name\n1,Alice\n2,Bob\n")
    schema_agent = _fake_schema_agent()
    cloud_service = _RecordingCloudIntegrationService(_success_cloud_outcome())
    runner = CliRunner()

    result = _invoke(
        runner, file_path, "customers_removed", schema_agent=schema_agent, cloud_service=cloud_service
    )

    assert result.exit_code == 0
    assert "applied: True" in result.output
    assert "age" not in Path(file_path).read_text(encoding="utf-8").splitlines()[0]  # source untouched
    assert len(cloud_service.calls) == 1
    output_path = cloud_service.calls[0]
    assert output_path != file_path
    assert "age" in Path(output_path).read_text(encoding="utf-8").splitlines()[0]


# --- G. rename -------------------------------------------------------------


def test_rename_drift_is_resolved_and_reaches_cloud_integration(tmp_path: Path) -> None:
    """`configs/schema_baselines/customers_rename.json` expects
    "customer_name"; this incoming file has "name" instead. The repair
    must rename the INCOMING column to the baseline's expected name
    (id,name,age -> id,customer_name,age)."""
    file_path = _write(tmp_path, "customers.csv", "id,name,age\n1,Alice,30\n")
    confirmation_port = _AlwaysConfirmConfirmationPort()
    schema_agent = _fake_schema_agent(confirmation_port=confirmation_port)
    cloud_service = _RecordingCloudIntegrationService(_success_cloud_outcome())
    runner = CliRunner()

    result = _invoke(
        runner, file_path, "customers_rename", schema_agent=schema_agent, cloud_service=cloud_service
    )

    assert result.exit_code == 0
    assert "applied: True" in result.output
    assert confirmation_port.calls >= 1  # the rename-resolution subgraph genuinely ran
    source_columns = Path(file_path).read_text(encoding="utf-8").splitlines()[0].split(",")
    assert source_columns == ["id", "name", "age"]  # source untouched
    assert len(cloud_service.calls) == 1
    output_path = cloud_service.calls[0]
    assert output_path != file_path
    columns = Path(output_path).read_text(encoding="utf-8").splitlines()[0].split(",")
    assert columns == ["id", "customer_name", "age"]  # renamed to the baseline's expected name
    assert "name" not in columns


# --- H. second rename --------------------------------------------------


def test_second_rename_baseline_is_resolved_and_reaches_cloud_integration(tmp_path: Path) -> None:
    """Same wiring proven against a second baseline
    (`customers_rename2.json`, which expects "full_name" instead of
    "name")."""
    file_path = _write(tmp_path, "customers.csv", "id,name,age\n1,Alice,30\n")
    confirmation_port = _AlwaysConfirmConfirmationPort()
    schema_agent = _fake_schema_agent(confirmation_port=confirmation_port)
    cloud_service = _RecordingCloudIntegrationService(_success_cloud_outcome())
    runner = CliRunner()

    result = _invoke(
        runner, file_path, "customers_rename2", schema_agent=schema_agent, cloud_service=cloud_service
    )

    assert result.exit_code == 0
    assert "applied: True" in result.output
    assert confirmation_port.calls >= 1
    source_columns = Path(file_path).read_text(encoding="utf-8").splitlines()[0].split(",")
    assert source_columns == ["id", "name", "age"]  # source untouched
    assert len(cloud_service.calls) == 1
    output_path = cloud_service.calls[0]
    assert output_path != file_path
    columns = Path(output_path).read_text(encoding="utf-8").splitlines()[0].split(",")
    assert columns == ["id", "full_name", "age"]
    assert "name" not in columns


# --- I. type change ---------------------------------------------------


def test_type_change_is_cast_verified_and_reaches_cloud_integration(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "customers.csv", "id,name,age\n1,Alice,30\n")
    schema_agent = _fake_schema_agent()
    cloud_service = _RecordingCloudIntegrationService(_success_cloud_outcome())
    runner = CliRunner()

    result = _invoke(
        runner, file_path, "customers_typechange", schema_agent=schema_agent, cloud_service=cloud_service
    )

    assert result.exit_code == 0
    assert "applied: True" in result.output
    assert len(cloud_service.calls) == 1
    assert cloud_service.calls[0] != file_path


# --- J. rejected -----------------------------------------------------------


def test_rejected_drift_never_reaches_cloud_integration(tmp_path: Path) -> None:
    """A human declining the Tier 2 prescription must never upload to
    Azure or trigger Databricks — proven by asserting the (fake) cloud
    service is never called at all."""
    file_path = _write(tmp_path, "customers.csv", "id,name,age,city\n1,Alice,30,Rome\n")
    schema_agent = _fake_schema_agent(approval_port=_RecordingApprovalPort(approved=False))
    cloud_service = _RecordingCloudIntegrationService(_success_cloud_outcome())
    runner = CliRunner()

    result = _invoke(
        runner, file_path, "customers_added", schema_agent=schema_agent, cloud_service=cloud_service
    )

    assert result.exit_code == 1
    assert "success: False" in result.output
    assert "applied: False" in result.output
    assert "cloud_integration: skipped" in result.output
    assert cloud_service.calls == []
