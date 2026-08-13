"""Focused, comprehensive tests for the Tier 2 `schema_repair_workflow`.

Uses the real `compute_column_diff`, the real nested rename-resolution
subgraph, and (for end-to-end tests) the real `PandasSchemaExecutor`
against real temporary CSV files — combined with fake baseline/inspector/
history/confirmation/approval ports. No real Groq/Azure network access.
"""

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from self_healing_pipeline.application.orchestration.audited_schema_repair import (
    run_audited_schema_repair,
)
from self_healing_pipeline.application.orchestration.schema_diff import compute_column_diff
from self_healing_pipeline.application.orchestration.schema_repair_workflow import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    build_initial_schema_repair_state,
    build_schema_repair_workflow,
    result_from_final_state,
)
from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition, SchemaBaseline
from self_healing_pipeline.domain.entities.schema_migration_entry import SchemaMigrationEntry
from self_healing_pipeline.domain.interfaces.services.human_approval_port import ApprovalRequest
from self_healing_pipeline.domain.interfaces.services.rename_confirmation_port import (
    RenameConfirmation,
)
from self_healing_pipeline.domain.value_objects.column_diff import RenameHint
from self_healing_pipeline.domain.value_objects.schema_repair_operations import (
    OperationType,
    SchemaRepairOperation,
)
from self_healing_pipeline.domain.value_objects.schema_repair_prescription import (
    SchemaRepairPrescription,
)
from self_healing_pipeline.domain.value_objects.schema_repair_result import SchemaRepairStatus
from self_healing_pipeline.infrastructure.schema.pandas_schema_executor import PandasSchemaExecutor
from self_healing_pipeline.infrastructure.schema.pandas_schema_inspector import (
    PandasSchemaInspector,
)

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


class _AlwaysRejectConfirmationPort:
    def confirm(self, *, table: str, hint: RenameHint) -> RenameConfirmation:
        return RenameConfirmation(confirmed=False, llm_confidence=0.0)


class _AlwaysConfirmConfirmationPort:
    def __init__(self, confidence: float = 0.95) -> None:
        self._confidence = confidence

    def confirm(self, *, table: str, hint: RenameHint) -> RenameConfirmation:
        return RenameConfirmation(confirmed=True, llm_confidence=self._confidence)


class _RecordingApprovalPort:
    def __init__(self, approved: bool) -> None:
        self._approved = approved
        self.requests: list[ApprovalRequest] = []

    def request_approval(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self._approved


class _FakeExecutor:
    """Fake `SchemaExecutor`: records ops it was asked to execute/verify."""

    def __init__(self, success: bool = True) -> None:
        self._success = success
        self.executed_ops: list[Any] = []

    def execute(self, file_path: str, operations: Any) -> Any:
        from self_healing_pipeline.domain.interfaces.services.schema_executor import (
            SchemaExecutionOutcome,
        )

        self.executed_ops.append(list(operations))
        return SchemaExecutionOutcome(success=self._success, message="fake execute")

    def verify(self, file_path: str, operations: Any) -> Any:
        from self_healing_pipeline.domain.interfaces.services.schema_executor import (
            SchemaExecutionOutcome,
        )

        return SchemaExecutionOutcome(success=self._success, message="fake verify")


def _write_csv(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _graph(
    *,
    baseline: SchemaBaseline | None,
    columns: tuple[ColumnDefinition, ...],
    history_store: Any,
    confirmation_port: Any,
    approval_port: Any,
    executor: Any,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Any:
    return build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(baseline),
        inspector=_FixedInspector(columns),
        history_store=history_store,
        confirmation_port=confirmation_port,
        approval_port=approval_port,
        executor=executor,
        confidence_threshold=threshold,
    )


# --- healthy / no-change ----------------------------------------------


def test_healthy_path_bypasses_llm_and_human_approval(tmp_path: Path) -> None:
    file_path = _write_csv(tmp_path, "customers.csv", "id,name,age\n1,Alice,30\n")
    confirmation_port = _AlwaysConfirmConfirmationPort()
    approval_port = _RecordingApprovalPort(approved=True)
    graph = _graph(
        baseline=BASELINE,
        columns=BASELINE.columns,
        history_store=_InMemoryHistoryStore(),
        confirmation_port=confirmation_port,
        approval_port=approval_port,
        executor=_FakeExecutor(),
    )

    final_state = graph.invoke(
        build_initial_schema_repair_state(episode_id=uuid4(), table="customers", file_path=file_path)
    )

    assert final_state["status"] is SchemaRepairStatus.HEALTHY
    assert approval_port.requests == []  # never asked


def test_no_baseline_fails_cleanly() -> None:
    graph = _graph(
        baseline=None,
        columns=BASELINE.columns,
        history_store=_InMemoryHistoryStore(),
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=_RecordingApprovalPort(approved=True),
        executor=_FakeExecutor(),
    )

    final_state = graph.invoke(
        build_initial_schema_repair_state(episode_id=uuid4(), table="unknown_table", file_path="x.csv")
    )

    assert final_state["status"] is SchemaRepairStatus.FAILED


# --- malformed prescription --------------------------------------------


def test_malformed_prescription_never_reaches_apply() -> None:
    """An empty raw-operations assembly (e.g. every candidate op filtered
    out) is treated as invalid — never silently skipped straight to
    success, never sent to a human for approval of nothing."""
    columns = BASELINE.columns  # identical -> would be healthy...
    # Force a non-healthy diff with a type change, then pre-reject it so
    # propose ends up with zero operations after episode-continuity filtering.
    changed = tuple(
        c.model_copy(update={"type": "string"}) if c.name == "age" else c for c in columns
    )
    history = _InMemoryHistoryStore()
    episode_id = uuid4()
    rejected_prescription_entry = SchemaMigrationEntry(
        episode_id=episode_id,
        table="customers",
        baseline_version=1,
        current_schema=changed,
        diff=compute_column_diff(BASELINE, changed),
        prescription=SchemaRepairPrescription(
            table="customers",
            operations=(
                SchemaRepairOperation(op=OperationType.CAST, column="age", target_type="string"),
            ),
            confidence=1.0,
        ),
        confidence=1.0,
        status=SchemaRepairStatus.REJECTED,
        applied=False,
        human_approved=False,
        verification_message=None,
        trace_id=None,
        mlflow_run_id=None,
    )
    history.record(rejected_prescription_entry)

    approval_port = _RecordingApprovalPort(approved=True)
    graph = _graph(
        baseline=BASELINE,
        columns=changed,
        history_store=history,
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=approval_port,
        executor=_FakeExecutor(),
    )

    final_state = graph.invoke(
        build_initial_schema_repair_state(episode_id=episode_id, table="customers", file_path="x.csv")
    )

    assert final_state["status"] is SchemaRepairStatus.INVALID
    assert approval_port.requests == []  # never asked to approve nothing


# --- confidence gating + human approval ---------------------------------


def test_low_confidence_is_escalated_and_rejection_still_prevents_apply() -> None:
    columns = tuple(
        ColumnDefinition(name="fullname", type="string") if c.name == "name" else c
        for c in BASELINE.columns
    )
    approval_port = _RecordingApprovalPort(approved=False)
    executor = _FakeExecutor()
    graph = _graph(
        baseline=BASELINE,
        columns=columns,
        history_store=_InMemoryHistoryStore(),
        confirmation_port=_AlwaysConfirmConfirmationPort(confidence=0.1),  # low
        approval_port=approval_port,
        executor=executor,
        threshold=0.7,
    )

    final_state = graph.invoke(
        build_initial_schema_repair_state(episode_id=uuid4(), table="customers", file_path="x.csv")
    )

    assert len(approval_port.requests) == 1
    assert approval_port.requests[0].escalated is True
    assert final_state["status"] is SchemaRepairStatus.REJECTED
    assert executor.executed_ops == []  # never applied


def test_high_confidence_still_requires_human_approval() -> None:
    columns = tuple(
        ColumnDefinition(name="fullname", type="string") if c.name == "name" else c
        for c in BASELINE.columns
    )
    approval_port = _RecordingApprovalPort(approved=True)
    graph = _graph(
        baseline=BASELINE,
        columns=columns,
        history_store=_InMemoryHistoryStore(),
        confirmation_port=_AlwaysConfirmConfirmationPort(confidence=0.99),  # high
        approval_port=approval_port,
        executor=_FakeExecutor(),
        threshold=0.7,
    )

    graph.invoke(
        build_initial_schema_repair_state(episode_id=uuid4(), table="customers", file_path="x.csv")
    )

    assert len(approval_port.requests) == 1  # asked regardless of high confidence
    assert approval_port.requests[0].escalated is False


def test_human_rejection_does_not_modify_data(tmp_path: Path) -> None:
    original = "id,name,age\n1,Alice,30\n2,Bob,25\n"
    file_path = _write_csv(tmp_path, "customers.csv", original)
    graph = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(
            SchemaBaseline(table="customers", version=1, columns=BASELINE.columns[:2])
        ),
        inspector=PandasSchemaInspector(),
        history_store=_InMemoryHistoryStore(),
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=_RecordingApprovalPort(approved=False),
        executor=PandasSchemaExecutor(),
    )

    graph.invoke(
        build_initial_schema_repair_state(episode_id=uuid4(), table="customers", file_path=file_path)
    )

    assert Path(file_path).read_text(encoding="utf-8") == original


def test_human_approval_allows_apply_end_to_end_added_column(tmp_path: Path) -> None:
    """"Added column" mode (spec's own example: baseline lacks a column
    the current data has) is healed by dropping the unexpected column —
    matching mode 1's worked example (baseline: id,name / current:
    id,name,country) exactly."""
    file_path = _write_csv(
        tmp_path, "customers.csv", "id,name,age,country\n1,Alice,30,India\n2,Bob,25,USA\n"
    )
    graph = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(BASELINE),  # baseline: id, name, age (no country)
        inspector=PandasSchemaInspector(),
        history_store=_InMemoryHistoryStore(),
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=_RecordingApprovalPort(approved=True),
        executor=PandasSchemaExecutor(),
    )

    final_state = graph.invoke(
        build_initial_schema_repair_state(episode_id=uuid4(), table="customers", file_path=file_path)
    )

    assert final_state["status"] is SchemaRepairStatus.SUCCEEDED
    assert final_state["diff"] is not None
    assert [c.name for c in final_state["diff"].added] == ["country"]
    assert final_state["validated_operations"] is not None
    assert final_state["validated_operations"][0].op is OperationType.DROP

    import pandas as pd

    frame = pd.read_csv(file_path)
    assert "country" not in frame.columns


# --- end-to-end drift modes ----------------------------------------------


def test_end_to_end_removed_column_repair(tmp_path: Path) -> None:
    """"Removed column" mode (spec's own example: baseline has a column
    the current data is missing) is healed by adding it back with a
    default — matching mode 2's worked example (baseline: id,name,age /
    current: id,name) exactly."""
    file_path = _write_csv(tmp_path, "customers.csv", "id,name\n1,Alice\n2,Bob\n")
    graph = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(BASELINE),  # baseline has id,name,age
        inspector=PandasSchemaInspector(),
        history_store=_InMemoryHistoryStore(),
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=_RecordingApprovalPort(approved=True),
        executor=PandasSchemaExecutor(),
    )

    final_state = graph.invoke(
        build_initial_schema_repair_state(episode_id=uuid4(), table="customers", file_path=file_path)
    )

    assert final_state["status"] is SchemaRepairStatus.SUCCEEDED
    assert final_state["diff"] is not None
    assert [c.name for c in final_state["diff"].removed] == ["age"]  # age missing from current
    assert final_state["validated_operations"] is not None
    assert final_state["validated_operations"][0].op is OperationType.ADD_DEFAULT

    import pandas as pd

    frame = pd.read_csv(file_path)
    assert "age" in frame.columns


def test_end_to_end_type_change_repair(tmp_path: Path) -> None:
    file_path = _write_csv(
        tmp_path, "customers.csv", "id,name,age\n1,Alice,thirty\n2,Bob,twenty-five\n"
    )
    graph = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(BASELINE),  # baseline: age=int64
        inspector=PandasSchemaInspector(),  # current: age will infer as string
        history_store=_InMemoryHistoryStore(),
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=_RecordingApprovalPort(approved=True),
        executor=PandasSchemaExecutor(),
    )

    final_state = graph.invoke(
        build_initial_schema_repair_state(episode_id=uuid4(), table="customers", file_path=file_path)
    )

    assert final_state["diff"] is not None
    assert len(final_state["diff"].type_changed) == 1
    assert final_state["status"] is SchemaRepairStatus.FAILED  # can't cast "thirty" to int64
    # honest negative: this proves apply/verify genuinely check real data,
    # not merely that the workflow routes correctly


def test_end_to_end_type_change_repair_succeeds_with_castable_data(tmp_path: Path) -> None:
    file_path = _write_csv(
        tmp_path, "customers.csv", "id,name,age\n1,Alice,30\n2,Bob,25\n"
    )
    baseline_with_string_age = SchemaBaseline(
        table="customers",
        version=1,
        columns=(
            ColumnDefinition(name="id", type="int64", nullable=False),
            ColumnDefinition(name="name", type="string", nullable=False),
            ColumnDefinition(name="age", type="string", nullable=True),
        ),
    )
    graph = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(baseline_with_string_age),
        inspector=PandasSchemaInspector(),  # current: age infers as int64
        history_store=_InMemoryHistoryStore(),
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=_RecordingApprovalPort(approved=True),
        executor=PandasSchemaExecutor(),
    )

    final_state = graph.invoke(
        build_initial_schema_repair_state(episode_id=uuid4(), table="customers", file_path=file_path)
    )

    assert final_state["status"] is SchemaRepairStatus.SUCCEEDED


# --- migration history -----------------------------------------------


def test_migration_json_written_for_applied_and_rejected(tmp_path: Path) -> None:
    approved_file = _write_csv(
        tmp_path, "approved.csv", "id,name,age,country\n1,Alice,30,India\n"
    )
    history = _InMemoryHistoryStore()
    approved_graph = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(BASELINE),
        inspector=PandasSchemaInspector(),
        history_store=history,
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=_RecordingApprovalPort(approved=True),
        executor=PandasSchemaExecutor(),
    )
    approved_episode = uuid4()
    approved_final = run_audited_schema_repair(
        approved_graph,
        build_initial_schema_repair_state(
            episode_id=approved_episode, table="customers", file_path=approved_file
        ),
        history_store=history,
    )
    assert result_from_final_state(approved_final).status is SchemaRepairStatus.SUCCEEDED
    approved_entries = history.load_episode_history(approved_episode)
    assert len(approved_entries) == 1
    assert approved_entries[0].applied is True
    assert approved_entries[0].status is SchemaRepairStatus.SUCCEEDED

    rejected_file = _write_csv(
        tmp_path, "rejected.csv", "id,name,age,country\n1,Alice,30,India\n"
    )
    rejected_graph = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(BASELINE),
        inspector=PandasSchemaInspector(),
        history_store=history,
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=_RecordingApprovalPort(approved=False),
        executor=PandasSchemaExecutor(),
    )
    rejected_episode = uuid4()
    rejected_final = run_audited_schema_repair(
        rejected_graph,
        build_initial_schema_repair_state(
            episode_id=rejected_episode, table="customers", file_path=rejected_file
        ),
        history_store=history,
    )
    assert result_from_final_state(rejected_final).status is SchemaRepairStatus.REJECTED
    rejected_entries = history.load_episode_history(rejected_episode)
    assert len(rejected_entries) == 1
    assert rejected_entries[0].applied is False
    assert rejected_entries[0].status is SchemaRepairStatus.REJECTED
    # rejection must never modify data
    assert (
        Path(rejected_file).read_text(encoding="utf-8")
        == "id,name,age,country\n1,Alice,30,India\n"
    )


def test_healthy_episode_writes_no_migration_entry(tmp_path: Path) -> None:
    file_path = _write_csv(tmp_path, "customers.csv", "id,name,age\n1,Alice,30\n")
    history = _InMemoryHistoryStore()
    graph = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(BASELINE),
        inspector=PandasSchemaInspector(),
        history_store=history,
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=_RecordingApprovalPort(approved=True),
        executor=PandasSchemaExecutor(),
    )
    episode_id = uuid4()

    run_audited_schema_repair(
        graph,
        build_initial_schema_repair_state(episode_id=episode_id, table="customers", file_path=file_path),
        history_store=history,
    )

    assert history.load_episode_history(episode_id) == ()


# --- episode continuity -------------------------------------------------


def test_rejected_operation_is_not_re_proposed_in_same_episode(tmp_path: Path) -> None:
    file_path = _write_csv(
        tmp_path, "customers.csv", "id,name,age,country\n1,Alice,30,India\n"
    )
    history = _InMemoryHistoryStore()
    episode_id = uuid4()

    # First attempt: human rejects.
    graph = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(BASELINE),
        inspector=PandasSchemaInspector(),
        history_store=history,
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=_RecordingApprovalPort(approved=False),
        executor=PandasSchemaExecutor(),
    )
    first_final = run_audited_schema_repair(
        graph,
        build_initial_schema_repair_state(episode_id=episode_id, table="customers", file_path=file_path),
        history_store=history,
    )
    assert first_final["status"] is SchemaRepairStatus.REJECTED
    assert len(history.load_episode_history(episode_id)) == 1

    # Second attempt, same episode_id: the exact same operation (add
    # "country") must not be re-proposed -> nothing left to propose ->
    # INVALID, and the human is never asked again for the same thing.
    approval_port_2 = _RecordingApprovalPort(approved=True)
    graph_2 = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(BASELINE),
        inspector=PandasSchemaInspector(),
        history_store=history,
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=approval_port_2,
        executor=PandasSchemaExecutor(),
    )
    second_final = graph_2.invoke(
        build_initial_schema_repair_state(episode_id=episode_id, table="customers", file_path=file_path)
    )

    assert second_final["status"] is SchemaRepairStatus.INVALID
    assert approval_port_2.requests == []


def test_prior_history_is_loaded_on_retry(tmp_path: Path) -> None:
    file_path = _write_csv(
        tmp_path, "customers.csv", "id,name,age,country\n1,Alice,30,India\n"
    )
    history = _InMemoryHistoryStore()
    episode_id = uuid4()
    graph = build_schema_repair_workflow(
        baseline_store=_FakeBaselineStore(BASELINE),
        inspector=PandasSchemaInspector(),
        history_store=history,
        confirmation_port=_AlwaysRejectConfirmationPort(),
        approval_port=_RecordingApprovalPort(approved=False),
        executor=PandasSchemaExecutor(),
    )
    run_audited_schema_repair(
        graph,
        build_initial_schema_repair_state(episode_id=episode_id, table="customers", file_path=file_path),
        history_store=history,
    )

    final_state = graph.invoke(
        build_initial_schema_repair_state(episode_id=episode_id, table="customers", file_path=file_path)
    )

    assert len(final_state["rejected_operation_keys"]) == 1
