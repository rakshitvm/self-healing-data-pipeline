"""Tier 2 schema repair workflow.

    detect -> diff -> rename_resolution_subgraph -> propose -> validate
    -> confidence_gate -> human_approval -> apply -> verify

A LangGraph `StateGraph`, architecturally parallel to (and independent
of) Tier 1's `csv_repair_workflow` — a separate agent/workflow, not a
CSV repair variant. `rename_resolution_subgraph` is a genuinely nested,
independently-compiled `StateGraph` invoked as a single node (see that
module) — not an ordinary function call.

Human-in-the-loop is architectural, not optional: `human_approval` runs
for *every* prescription regardless of confidence (a lead requirement —
confidence only ever adds an `escalated` flag for extra visibility, it
never bypasses the human gate, and it never allows auto-apply). Nothing
downstream of `human_approval` runs unless a human explicitly approved.

No retry loop: unlike CSV repair, a human decision is terminal for this
invocation. Cross-invocation continuity (loading prior history, not
re-proposing an already-rejected operation within the same episode) is
handled by `detect` loading `episode_id`'s prior migration history.
"""

from typing import Any, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from self_healing_pipeline.application.orchestration.rename_resolution_subgraph import (
    build_initial_rename_resolution_state,
    build_rename_resolution_subgraph,
)
from self_healing_pipeline.application.orchestration.schema_diff import compute_column_diff
from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition, SchemaBaseline
from self_healing_pipeline.domain.interfaces.repositories.schema_migration_history_store import (
    SchemaMigrationHistoryStore,
)
from self_healing_pipeline.domain.interfaces.services.current_schema_inspector import (
    CurrentSchemaInspector,
)
from self_healing_pipeline.domain.interfaces.services.human_approval_port import (
    ApprovalRequest,
    HumanApprovalPort,
)
from self_healing_pipeline.domain.interfaces.services.rename_confirmation_port import (
    RenameConfirmationPort,
)
from self_healing_pipeline.domain.interfaces.services.schema_baseline_store import (
    SchemaBaselineStore,
)
from self_healing_pipeline.domain.interfaces.services.schema_executor import (
    SchemaExecutionOutcome,
    SchemaExecutor,
)
from self_healing_pipeline.domain.value_objects.column_diff import ColumnDiff, ConfirmedRename
from self_healing_pipeline.domain.value_objects.schema_repair_operations import (
    OperationType,
    SchemaRepairOperation,
    normalize_operation_order,
)
from self_healing_pipeline.domain.value_objects.schema_repair_prescription import (
    SchemaRepairPrescription,
)
from self_healing_pipeline.domain.value_objects.schema_repair_result import (
    SchemaRepairResult,
    SchemaRepairStatus,
)
from self_healing_pipeline.infrastructure.logging.logger import get_logger

DEFAULT_CONFIDENCE_THRESHOLD = 0.7
_STRING_LIKE_TYPES = {"string", "str", "object", "text"}


def _default_value_for_type(column_type: str) -> str | int | float | bool | None:
    """Deterministic default-value policy for a newly added column.

    Not LLM-derived: which default to propose for a brand-new column is a
    mechanical policy decision, not a fact requiring reasoning, so the
    LLM is never consulted for it (Tier 1/2 shared principle: LLM calls
    only where genuine reasoning is needed).
    """
    normalized = column_type.strip().lower()
    if normalized in _STRING_LIKE_TYPES:
        return "UNKNOWN"
    if normalized in {"int64", "int", "integer"}:
        return 0
    if normalized in {"float64", "float", "double"}:
        return 0.0
    if normalized in {"bool", "boolean"}:
        return False
    return None


def operation_key(operation: SchemaRepairOperation) -> tuple[str, str, str | None, str | None]:
    """A stable identity for an operation, used to detect prior rejection."""
    return (operation.op.value, operation.column, operation.target_column, operation.target_type)


class SchemaRepairWorkflowState(TypedDict):
    """State threaded through the Tier 2 schema repair workflow."""

    episode_id: UUID
    table: str
    file_path: str
    baseline: SchemaBaseline | None
    current_schema: tuple[ColumnDefinition, ...]
    diff: ColumnDiff | None
    confirmed_renames: list[ConfirmedRename]
    rejected_operation_keys: set[tuple[str, str, str | None, str | None]]
    raw_operations: list[SchemaRepairOperation]
    validated_operations: list[SchemaRepairOperation] | None
    validation_errors: list[str]
    confidence: float | None
    escalated: bool
    human_approved: bool | None
    verification_outcome: SchemaExecutionOutcome | None
    status: SchemaRepairStatus | None
    message: str | None


def build_initial_schema_repair_state(
    *, episode_id: UUID, table: str, file_path: str
) -> SchemaRepairWorkflowState:
    """Build the initial state for a fresh schema repair episode."""
    return SchemaRepairWorkflowState(
        episode_id=episode_id,
        table=table,
        file_path=file_path,
        baseline=None,
        current_schema=(),
        diff=None,
        confirmed_renames=[],
        rejected_operation_keys=set(),
        raw_operations=[],
        validated_operations=None,
        validation_errors=[],
        confidence=None,
        escalated=False,
        human_approved=None,
        verification_outcome=None,
        status=None,
        message=None,
    )


def _make_detect_node(
    baseline_store: SchemaBaselineStore,
    inspector: CurrentSchemaInspector,
    history_store: SchemaMigrationHistoryStore,
) -> Any:
    def detect(state: SchemaRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="SchemaRepairAgent", node="detect", table=state["table"])

        baseline = baseline_store.load_latest(state["table"])
        current_schema = inspector.inspect(state["file_path"])

        prior_entries = history_store.load_episode_history(state["episode_id"])
        rejected_keys: set[tuple[str, str, str | None, str | None]] = set()
        for entry in prior_entries:
            if entry.status is SchemaRepairStatus.REJECTED and entry.prescription is not None:
                for op in entry.prescription.operations:
                    rejected_keys.add(operation_key(op))

        logger.info(
            "node_completed",
            baseline_found=baseline is not None,
            current_column_count=len(current_schema),
            prior_rejected_operations=len(rejected_keys),
        )

        return {
            "baseline": baseline,
            "current_schema": current_schema,
            "rejected_operation_keys": rejected_keys,
        }

    return detect


def _route_after_detect(state: SchemaRepairWorkflowState) -> str:
    return "diff" if state["baseline"] is not None else "no_baseline"


def _diff(state: SchemaRepairWorkflowState) -> dict[str, Any]:
    logger = get_logger(agent="SchemaRepairAgent", node="diff", table=state["table"])
    assert state["baseline"] is not None  # guaranteed by _route_after_detect

    diff = compute_column_diff(state["baseline"], state["current_schema"])

    logger.info(
        "node_completed",
        added=[c.name for c in diff.added],
        removed=[c.name for c in diff.removed],
        type_changed=[t.column for t in diff.type_changed],
        rename_hint_count=len(diff.rename_hints),
    )

    return {"diff": diff}


def _route_after_diff(state: SchemaRepairWorkflowState) -> str:
    assert state["diff"] is not None
    return "resolve_renames" if state["diff"].has_changes else "healthy"


def _make_propose_node() -> Any:
    def propose(state: SchemaRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="SchemaRepairAgent", node="propose", table=state["table"])
        diff = state["diff"]
        assert diff is not None

        renamed_removed = {r.removed_column for r in state["confirmed_renames"]}
        renamed_added = {r.added_column for r in state["confirmed_renames"]}

        operations: list[SchemaRepairOperation] = []
        confidences: list[float] = []

        for rename in state["confirmed_renames"]:
            # `rename.added_column` is the column *currently present* in the
            # data (unknown to the baseline); `rename.removed_column` is the
            # column the baseline *expects* but the data doesn't have. The
            # repair must rename the incoming column to the baseline's
            # expected name — i.e. `column` (rename FROM) is the current
            # name, `target_column` (rename TO) is the baseline name.
            # `PandasSchemaExecutor` renames `column` -> `target_column`
            # verbatim, so this ordering is what makes that rename real
            # rather than a no-op against a column name the file never had.
            operations.append(
                SchemaRepairOperation(
                    op=OperationType.RENAME,
                    column=rename.added_column,
                    target_column=rename.removed_column,
                )
            )
            confidences.append(rename.confidence)

        for change in diff.type_changed:
            operations.append(
                SchemaRepairOperation(
                    op=OperationType.CAST, column=change.column, target_type=change.to_type
                )
            )
            confidences.append(1.0)

        # A column the baseline expects but that is missing from the
        # current data is healed by adding it back with a default (there
        # is nothing to drop — it is already absent). A column present
        # in the current data but unknown to the baseline is healed by
        # dropping it. This matches both worked examples in the spec:
        # "removed column" (baseline has it, current doesn't) -> add
        # back with a default; "added column" (current has it, baseline
        # doesn't) -> drop.
        for removed in diff.removed:
            if removed.name in renamed_removed:
                continue
            operations.append(
                SchemaRepairOperation(
                    op=OperationType.ADD_DEFAULT,
                    column=removed.name,
                    target_type=removed.type,
                    default=_default_value_for_type(removed.type),
                )
            )
            confidences.append(1.0)

        for added in diff.added:
            if added.name in renamed_added:
                continue
            operations.append(SchemaRepairOperation(op=OperationType.DROP, column=added.name))
            confidences.append(1.0)

        # Episode continuity: never re-propose an operation already
        # rejected earlier in this same episode.
        rejected_keys = state["rejected_operation_keys"]
        kept_operations: list[SchemaRepairOperation] = []
        kept_confidences: list[float] = []
        skipped = 0
        for operation, confidence in zip(operations, confidences, strict=True):
            if operation_key(operation) in rejected_keys:
                skipped += 1
                continue
            kept_operations.append(operation)
            kept_confidences.append(confidence)

        overall_confidence = min(kept_confidences) if kept_confidences else 1.0

        logger.info(
            "node_completed",
            operation_count=len(kept_operations),
            skipped_previously_rejected=skipped,
            confidence=overall_confidence,
        )

        return {"raw_operations": kept_operations, "confidence": overall_confidence}

    return propose


def _validate(state: SchemaRepairWorkflowState) -> dict[str, Any]:
    logger = get_logger(agent="SchemaRepairAgent", node="validate", table=state["table"])

    if not state["raw_operations"]:
        logger.info("node_completed", validation_status="invalid", validation_errors=["no_operations"])
        return {"validated_operations": None, "validation_errors": ["no_operations_to_apply"]}

    try:
        ordered = normalize_operation_order(state["raw_operations"])
        validated_operations: list[SchemaRepairOperation] | None = ordered
        validation_errors: list[str] = []
    except Exception as exc:  # noqa: BLE001 - any assembly/order failure is a validation failure
        validated_operations = None
        validation_errors = [str(exc)]

    logger.info(
        "node_completed",
        validation_status="valid" if validated_operations is not None else "invalid",
        validation_errors=validation_errors,
    )

    return {"validated_operations": validated_operations, "validation_errors": validation_errors}


def _route_after_validation(state: SchemaRepairWorkflowState) -> str:
    return "confidence_gate" if state["validated_operations"] is not None else "invalid"


def _make_confidence_gate_node(threshold: float) -> Any:
    def confidence_gate(state: SchemaRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(
            agent="SchemaRepairAgent", node="confidence_gate", table=state["table"]
        )
        confidence = state["confidence"] or 0.0
        escalated = confidence < threshold

        if escalated:
            logger.warning(
                "escalation",
                confidence=confidence,
                threshold=threshold,
                reason="confidence below threshold; human approval required (never auto-applied)",
            )
        logger.info("node_completed", confidence=confidence, escalated=escalated)

        return {"escalated": escalated}

    return confidence_gate


def _make_human_approval_node(port: HumanApprovalPort) -> Any:
    def human_approval(state: SchemaRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(
            agent="SchemaRepairAgent", node="human_approval", table=state["table"]
        )
        assert state["diff"] is not None
        assert state["validated_operations"] is not None

        prescription = SchemaRepairPrescription(
            table=state["table"],
            operations=tuple(state["validated_operations"]),
            confidence=state["confidence"] or 0.0,
        )
        request = ApprovalRequest(
            table=state["table"],
            diff=state["diff"],
            prescription=prescription,
            confidence=state["confidence"] or 0.0,
            escalated=state["escalated"],
        )

        # The LLM never authorizes application: this is the sole gate.
        approved = port.request_approval(request)

        logger.info("node_completed", human_approved=approved)

        return {"human_approved": approved}

    return human_approval


def _route_after_human_approval(state: SchemaRepairWorkflowState) -> str:
    return "apply" if state["human_approved"] else "rejected"


def _make_apply_node(executor: SchemaExecutor) -> Any:
    def apply(state: SchemaRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="SchemaRepairAgent", node="apply", table=state["table"])
        assert state["validated_operations"] is not None

        outcome = executor.execute(state["file_path"], tuple(state["validated_operations"]))

        logger.info("node_completed", success=outcome.success, message=outcome.message)

        return {"verification_outcome": outcome}

    return apply


def _make_verify_node(executor: SchemaExecutor) -> Any:
    def verify(state: SchemaRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="SchemaRepairAgent", node="verify", table=state["table"])
        assert state["validated_operations"] is not None

        outcome = executor.verify(state["file_path"], tuple(state["validated_operations"]))

        logger.info("node_completed", success=outcome.success, message=outcome.message)

        return {"verification_outcome": outcome}

    return verify


def _route_after_verify(state: SchemaRepairWorkflowState) -> str:
    outcome = state["verification_outcome"]
    return "succeeded" if outcome is not None and outcome.success else "failed"


def _terminal(status: SchemaRepairStatus, message: str) -> Any:
    def node(state: SchemaRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="SchemaRepairAgent", node=status.value, table=state["table"])
        logger.info("repair_completed", status=status.value, success=status is SchemaRepairStatus.SUCCEEDED)
        return {"status": status, "message": message}

    return node


def build_schema_repair_workflow(
    *,
    baseline_store: SchemaBaselineStore,
    inspector: CurrentSchemaInspector,
    history_store: SchemaMigrationHistoryStore,
    confirmation_port: RenameConfirmationPort,
    approval_port: HumanApprovalPort,
    executor: SchemaExecutor,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> CompiledStateGraph[SchemaRepairWorkflowState, None, Any, Any]:
    """Build and compile the Tier 2 schema repair `StateGraph`."""
    graph = StateGraph(SchemaRepairWorkflowState)

    graph.add_node("detect", _make_detect_node(baseline_store, inspector, history_store))
    graph.add_node("diff", _diff)
    graph.add_node("resolve_renames", _make_rename_resolution_node(confirmation_port))
    graph.add_node("propose", _make_propose_node())
    graph.add_node("validate", _validate)
    graph.add_node("confidence_gate", _make_confidence_gate_node(confidence_threshold))
    graph.add_node("human_approval", _make_human_approval_node(approval_port))
    graph.add_node("apply", _make_apply_node(executor))
    graph.add_node("verify", _make_verify_node(executor))
    graph.add_node("no_baseline", _terminal(SchemaRepairStatus.FAILED, "no baseline found for table"))
    graph.add_node("healthy", _terminal(SchemaRepairStatus.HEALTHY, "no schema drift detected"))
    graph.add_node("invalid", _terminal(SchemaRepairStatus.INVALID, "prescription failed validation"))
    graph.add_node("rejected", _terminal(SchemaRepairStatus.REJECTED, "human rejected the proposed repair"))
    graph.add_node("succeeded", _terminal(SchemaRepairStatus.SUCCEEDED, "schema repair applied and verified"))
    graph.add_node("failed", _terminal(SchemaRepairStatus.FAILED, "apply/verify did not succeed"))

    graph.add_edge(START, "detect")
    graph.add_conditional_edges(
        "detect", _route_after_detect, {"diff": "diff", "no_baseline": "no_baseline"}
    )
    graph.add_conditional_edges(
        "diff", _route_after_diff, {"resolve_renames": "resolve_renames", "healthy": "healthy"}
    )
    graph.add_edge("resolve_renames", "propose")
    graph.add_edge("propose", "validate")
    graph.add_conditional_edges(
        "validate", _route_after_validation, {"confidence_gate": "confidence_gate", "invalid": "invalid"}
    )
    graph.add_edge("confidence_gate", "human_approval")
    graph.add_conditional_edges(
        "human_approval", _route_after_human_approval, {"apply": "apply", "rejected": "rejected"}
    )
    graph.add_edge("apply", "verify")
    graph.add_conditional_edges(
        "verify", _route_after_verify, {"succeeded": "succeeded", "failed": "failed"}
    )
    graph.add_edge("no_baseline", END)
    graph.add_edge("healthy", END)
    graph.add_edge("invalid", END)
    graph.add_edge("rejected", END)
    graph.add_edge("succeeded", END)
    graph.add_edge("failed", END)

    return graph.compile()


def _make_rename_resolution_node(confirmation_port: RenameConfirmationPort) -> Any:
    """Wraps the compiled nested subgraph: translates parent state into the
    subgraph's own state, invokes it (a genuine nested `StateGraph`, not a
    plain function call), and translates its result back."""
    subgraph = build_rename_resolution_subgraph(confirmation_port=confirmation_port)

    def resolve_renames(state: SchemaRepairWorkflowState) -> dict[str, Any]:
        assert state["diff"] is not None
        sub_initial = build_initial_rename_resolution_state(
            table=state["table"],
            removed_columns=tuple(c.name for c in state["diff"].removed),
            added_columns=tuple(c.name for c in state["diff"].added),
        )
        sub_result = subgraph.invoke(sub_initial)
        return {"confirmed_renames": sub_result["confirmed_renames"]}

    return resolve_renames


def result_from_final_state(final_state: SchemaRepairWorkflowState) -> SchemaRepairResult:
    """Translate the workflow's final state into the public `SchemaRepairResult`."""
    prescription: SchemaRepairPrescription | None = None
    if final_state["validated_operations"] is not None:
        prescription = SchemaRepairPrescription(
            table=final_state["table"],
            operations=tuple(final_state["validated_operations"]),
            confidence=final_state["confidence"] or 0.0,
        )

    status = final_state["status"] or SchemaRepairStatus.FAILED
    return SchemaRepairResult(
        status=status,
        diff=final_state["diff"],
        prescription=prescription,
        confidence=final_state["confidence"],
        human_approved=final_state["human_approved"],
        validation_errors=final_state["validation_errors"],
        message=final_state["message"],
    )
