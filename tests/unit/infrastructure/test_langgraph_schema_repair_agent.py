"""Focused unit tests for `LangGraphSchemaRepairAgent`.

Mirrors `test_langgraph_csv_repair_agent.py`'s structure exactly, but for
the Tier 2 adapter. Uses the real `build_schema_repair_workflow`,
`compute_column_diff`, and (for the repair-succeeds test) the real
`PandasSchemaExecutor` against a real temporary CSV file — plus fakes
for the baseline store, inspector, history store, confirmation port, and
approval port, matching the conventions already established in
`tests/unit/application/test_schema_repair_workflow.py`. No real
Postgres/MLflow/LLM network calls.
"""

from pathlib import Path
from typing import Any
from uuid import UUID

from self_healing_pipeline.application.orchestration.schema_repair_workflow import (
    build_schema_repair_workflow,
)
from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition, SchemaBaseline
from self_healing_pipeline.domain.entities.schema_migration_entry import SchemaMigrationEntry
from self_healing_pipeline.domain.exceptions.schema_errors import SchemaDriftError
from self_healing_pipeline.domain.interfaces.agents.repair_agent import RepairAgent
from self_healing_pipeline.domain.interfaces.services.human_approval_port import ApprovalRequest
from self_healing_pipeline.domain.interfaces.services.rename_confirmation_port import (
    RenameConfirmation,
)
from self_healing_pipeline.domain.value_objects.column_diff import RenameHint
from self_healing_pipeline.domain.value_objects.repair_result import RepairResult
from self_healing_pipeline.infrastructure.agents.langgraph_schema_repair_agent import (
    LangGraphSchemaRepairAgent,
)
from self_healing_pipeline.infrastructure.schema.pandas_schema_executor import PandasSchemaExecutor

BASELINE = SchemaBaseline(
    table="customers",
    version=1,
    columns=(
        ColumnDefinition(name="id", type="int64", nullable=False),
        ColumnDefinition(name="name", type="string", nullable=False),
        ColumnDefinition(name="age", type="int64", nullable=True),
    ),
)


class _FakeBaselineStore:
    def __init__(self, baseline: SchemaBaseline | None) -> None:
        self._baseline = baseline

    def load_latest(self, table: str) -> SchemaBaseline | None:
        return self._baseline


class _FixedInspector:
    def __init__(self, columns: tuple[ColumnDefinition, ...]) -> None:
        self._columns = columns

    def inspect(self, file_path: str) -> tuple[ColumnDefinition, ...]:
        return self._columns


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


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _agent(
    *,
    baseline: SchemaBaseline | None,
    columns: tuple[ColumnDefinition, ...],
    confirmation_port: Any = None,
    approval_port: Any = None,
    history_store: Any = None,
) -> LangGraphSchemaRepairAgent:
    history_store = history_store if history_store is not None else _InMemoryHistoryStore()
    graph = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(baseline),
        inspector=_FixedInspector(columns),
        history_store=history_store,
        confirmation_port=confirmation_port or _AlwaysConfirmConfirmationPort(),
        approval_port=approval_port or _RecordingApprovalPort(approved=True),
        executor=PandasSchemaExecutor(),
    )
    return LangGraphSchemaRepairAgent(graph, history_store=history_store)


def test_agent_satisfies_repair_agent_protocol() -> None:
    agent = _agent(baseline=BASELINE, columns=BASELINE.columns)

    assert isinstance(agent, RepairAgent)


def test_no_drift_returns_success_true_applied_false_and_skips_llm(tmp_path: Path) -> None:
    """Healthy/no-change: mirrors Tier 1's own healthy-path shape exactly
    so the CLI's cloud-integration gating needs no Tier-2-specific case.
    Also proves zero LLM calls: `resolve_renames` (the only node that
    calls the confirmation port) is never reached when the diff has no
    changes."""
    file_path = _write(tmp_path, "customers.csv", "id,name,age\n1,Alice,30\n")
    confirmation_port = _AlwaysConfirmConfirmationPort()
    agent = _agent(baseline=BASELINE, columns=BASELINE.columns, confirmation_port=confirmation_port)
    error = SchemaDriftError("check", table_name="customers", file_path=file_path)

    result = agent.handle(error)

    assert isinstance(result, RepairResult)
    assert result.success is True
    assert result.applied is False
    assert result.output_path is None
    assert result.source_path == file_path
    assert confirmation_port.calls == 0


def test_added_column_drift_is_repaired_to_a_separate_output_file(tmp_path: Path) -> None:
    """Proves the adapter genuinely drives the canonical StateGraph (real
    `PandasSchemaExecutor` against a real file), and that the resulting
    `output_path` is a *separate* `repaired/<name>` file — mirroring
    Tier 1 exactly, the source is never touched."""
    file_path = _write(tmp_path, "customers.csv", "id,name,age,city\n1,Alice,30,Rome\n")
    agent = _agent(baseline=BASELINE, columns=(*BASELINE.columns, ColumnDefinition(name="city", type="string")))
    error = SchemaDriftError("check", table_name="customers", file_path=file_path)

    result = agent.handle(error)

    assert result.success is True
    assert result.applied is True
    assert result.output_path is not None
    assert result.output_path != file_path
    assert result.source_path == file_path
    # the source is never touched
    assert "city" in Path(file_path).read_text(encoding="utf-8").splitlines()[0]
    # "city" (unknown to the baseline) was dropped in the separate output file
    assert "city" not in Path(result.output_path).read_text(encoding="utf-8").splitlines()[0]


def test_rename_drift_renames_incoming_column_to_baseline_name(tmp_path: Path) -> None:
    """The baseline expects "name"; the incoming file has "customer_name"
    instead. The repair must rename the INCOMING column to the
    baseline's expected name (id,customer_name,age -> id,name,age), not
    the reverse — regression test for the rename-direction fix."""
    file_path = _write(tmp_path, "customers.csv", "id,customer_name,age\n1,Alice,30\n")
    confirmation_port = _AlwaysConfirmConfirmationPort()
    current_columns = (
        ColumnDefinition(name="id", type="int64", nullable=False),
        ColumnDefinition(name="customer_name", type="string", nullable=False),
        ColumnDefinition(name="age", type="int64", nullable=True),
    )
    agent = _agent(baseline=BASELINE, columns=current_columns, confirmation_port=confirmation_port)
    error = SchemaDriftError("check", table_name="customers", file_path=file_path)

    result = agent.handle(error)

    assert result.success is True
    assert result.applied is True
    assert confirmation_port.calls >= 1
    # the source is never touched
    source_columns = Path(file_path).read_text(encoding="utf-8").splitlines()[0].split(",")
    assert source_columns == ["id", "customer_name", "age"]
    assert result.output_path is not None
    columns = Path(result.output_path).read_text(encoding="utf-8").splitlines()[0].split(",")
    assert columns == ["id", "name", "age"]
    assert "customer_name" not in columns


def test_rejected_repair_returns_unsuccessful_result_with_applied_false(tmp_path: Path) -> None:
    """A human declining the prescription must map to `success=False,
    applied=False` — the same shape the CLI already treats as "skip the
    cloud step, exit non-zero" for Tier 1."""
    file_path = _write(tmp_path, "customers.csv", "id,name,age,city\n1,Alice,30,Rome\n")
    agent = _agent(
        baseline=BASELINE,
        columns=(*BASELINE.columns, ColumnDefinition(name="city", type="string")),
        approval_port=_RecordingApprovalPort(approved=False),
    )
    error = SchemaDriftError("check", table_name="customers", file_path=file_path)

    result = agent.handle(error)

    assert result.success is False
    assert result.applied is False
    assert result.output_path is None


def test_no_baseline_found_returns_unsuccessful_result(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "customers.csv", "id,name,age\n1,Alice,30\n")
    agent = _agent(baseline=None, columns=BASELINE.columns)
    error = SchemaDriftError("check", table_name="unknown_table", file_path=file_path)

    result = agent.handle(error)

    assert result.success is False
    assert result.applied is False
    assert result.output_path is None


def test_missing_table_name_returns_unsuccessful_result_without_invoking_the_graph() -> None:
    """`SchemaDriftError` itself requires `table_name`, but the adapter's
    `handle()` accepts the generic `PipelineError` Protocol type (any
    router-dispatched error could in principle lack a table) — this
    proves the defensive check, not just the type-level constraint."""
    from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
    from self_healing_pipeline.domain.value_objects.failure_class import FailureClass

    agent = _agent(baseline=BASELINE, columns=BASELINE.columns)
    error = PipelineError("no table", failure_class=FailureClass.SCHEMA_DRIFT, file_path="/tmp/x.csv")

    result = agent.handle(error)

    assert result.success is False
    assert result.applied is False
    assert result.validation_errors == ["missing_table_name_or_file_path"]


def test_missing_file_path_returns_unsuccessful_result_without_invoking_the_graph() -> None:
    agent = _agent(baseline=BASELINE, columns=BASELINE.columns)
    error = SchemaDriftError("check", table_name="customers")  # file_path defaults to None

    result = agent.handle(error)

    assert result.success is False
    assert result.applied is False
    assert result.validation_errors == ["missing_table_name_or_file_path"]
