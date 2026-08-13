"""Tier 1 CSV repair workflow: sample -> diagnose -> propose -> validate -> [human_approval] -> apply -> re-verify.

A LangGraph `StateGraph` orchestrating the existing Tier 1 building blocks
(`CsvFailureDetector`, `CsvRepairExecutor`, and a provider-agnostic
`CsvRepairProposalPort`) into a single self-healing workflow. This module
is orchestration only: it contains no pandas, no filesystem repair logic,
and no LLM-provider-specific code — every infrastructure concern is
injected as one of the three ports above, and the only filesystem access
here is the deterministic sampling tool, which reads (never writes) a few
lines of the target file.

Node responsibilities:

- `prepare_sample_call` / the `sample_tool` `ToolNode` / `extract_sample`:
  deterministically obtain a small sample of the CSV. No LLM call.
- `diagnose`: delegate to `CsvFailureDetector`. No LLM call. If no Tier 1
  failure is detected, the graph short-circuits straight to success
  without ever invoking the proposal port. A single file can genuinely
  exhibit more than one Tier 1 failure dimension at once (see
  `LocalCsvFailureDetector.detect_all`) — `state["failure_classes"]`
  holds the complete set, while `state["failure_class"]` keeps holding
  the single backward-compatible primary value every other node already
  relied on before multi-failure detection existed.
- `propose`: the reasoning boundary. Delegates to `CsvRepairProposalPort`
  and stores its *raw, unvalidated* output. For a multi-failure episode,
  the sample is prefixed with a deterministic evidence line listing every
  detected failure, so the LLM proposes one combined prescription instead
  of fixing only the primary dimension. For `WRONG_ENCODING` (whether
  alone or alongside other failures) specifically, a deterministic (no
  LLM) `chardet` pass over the file's raw bytes additionally adds a
  best-effort encoding hint to the *local* sample text — `state["sample"]`
  itself is never altered. Because `encoding` is directly determined by
  the file's actual bytes rather than a matter of judgment, when chardet
  names one, that single field of the LLM's raw proposal is
  deterministically corrected to match before the proposal is stored —
  every other field (including whatever the LLM proposed for delimiter/
  header_row/engine to address the *other* detected failures) remains
  exactly what the LLM returned, so one failure's correction can never
  overwrite another's required parameter. The (possibly-corrected) result
  still flows through `validate` unchanged, so an invalid proposal still
  never reaches `apply`.
- `validate`: the only thing allowed to turn a raw proposal into a
  `CsvRepairParams` that `apply` may use. An invalid proposal never
  reaches `apply`.
- `human_approval`: only reached for a genuine multi-failure episode
  (`len(failure_classes) > 1`) — a single-failure repair auto-applies
  exactly as it always has. Delegates to the injected
  `CsvHumanApprovalPort`; a rejection routes straight to `set_rejected`
  and `apply` is never called. Omitting `approval_port` (the default)
  reproduces prior (single-failure-only) behavior exactly, and — as a
  fail-safe — is treated as an automatic rejection if a multi-failure
  episode is ever reached without one wired in, so approval can never be
  silently bypassed.
- `apply`: delegates to `CsvRepairExecutor` with the validated
  prescription — no pandas logic is duplicated here.
- `reverify`: an independent, deterministic post-apply check, again via
  `CsvRepairExecutor`. No LLM call.

Retries are bounded by `max_retries` (checked before every retry) and are
driven by two conditional edges (after `validate` and after `reverify`),
both funnelling into a single `increment_retry` node before looping back
to `propose` — there is no unconditional cycle in this graph.
"""

from typing import Annotated, Any, TypedDict
from uuid import UUID, uuid4

import chardet
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from self_healing_pipeline.domain.interfaces.services.csv_approval_port import (
    CsvApprovalRequest,
    CsvHumanApprovalPort,
)
from self_healing_pipeline.domain.interfaces.services.csv_failure_detector import (
    CsvFailureDetector,
)
from self_healing_pipeline.domain.interfaces.services.csv_repair_executor import (
    CsvExecutionOutcome,
    CsvRepairExecutor,
)
from self_healing_pipeline.domain.interfaces.services.csv_repair_proposal_port import (
    CsvRepairProposalPort,
)
from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.pipeline_status import RepairEpisodeStatus
from self_healing_pipeline.domain.value_objects.repair_result import RepairResult
from self_healing_pipeline.infrastructure.logging.logger import get_logger

SAMPLE_TOOL_NODE_NAME = "sample_tool"
DEFAULT_MAX_SAMPLE_LINES = 5
DEFAULT_MAX_RETRIES = 2


@tool
def sample_csv_file(file_path: str, max_lines: int) -> str:
    """Read up to `max_lines` lines from the start of a local CSV file.

    Encoding-aware via the existing `_detect_encoding` (chardet) helper,
    so e.g. UTF-16 files are decoded correctly instead of corrupted —
    falling back to UTF-8 with `errors="replace"`, exactly as before,
    whenever detection finds nothing or the detected encoding itself
    can't be used to open the file.
    """
    detected = _detect_encoding(file_path)
    encoding = detected["encoding"] if detected is not None else "utf-8"

    lines: list[str] = []
    try:
        with open(file_path, encoding=encoding, errors="replace") as fh:
            for _, line in zip(range(max_lines), fh, strict=False):
                lines.append(line)
    except (LookupError, OSError):
        lines = []
        with open(file_path, encoding="utf-8", errors="replace") as fh:
            for _, line in zip(range(max_lines), fh, strict=False):
                lines.append(line)

    return "".join(lines)


class CsvRepairWorkflowState(TypedDict):
    """State threaded through the Tier 1 CSV repair workflow."""

    episode_id: UUID
    file_path: str
    failure_class: FailureClass | None
    failure_classes: frozenset[FailureClass]
    sample: str | None
    proposed_params: dict[str, Any] | None
    validated_params: CsvRepairParams | None
    validation_errors: list[str]
    repair_result: RepairResult | None
    verification_result: CsvExecutionOutcome | None
    retry_count: int
    max_retries: int
    status: RepairEpisodeStatus
    error_message: str | None
    human_approved: bool | None
    messages: Annotated[list[BaseMessage], add_messages]


def build_initial_state(
    file_path: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    episode_id: UUID | None = None,
) -> CsvRepairWorkflowState:
    """Build the initial state for a fresh workflow run over `file_path`."""
    return CsvRepairWorkflowState(
        episode_id=episode_id or uuid4(),
        file_path=file_path,
        failure_class=None,
        failure_classes=frozenset(),
        sample=None,
        proposed_params=None,
        validated_params=None,
        validation_errors=[],
        repair_result=None,
        verification_result=None,
        retry_count=0,
        max_retries=max_retries,
        status=RepairEpisodeStatus.PENDING,
        error_message=None,
        human_approved=None,
        messages=[],
    )


def _make_prepare_sample_call_node(max_sample_lines: int) -> Any:
    def prepare_sample_call(state: CsvRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="CsvRepairAgent", node="prepare_sample_call", table=None)

        call = ToolCall(
            name=sample_csv_file.name,
            args={"file_path": state["file_path"], "max_lines": max_sample_lines},
            id="sample-call",
            type="tool_call",
        )

        logger.info("node_completed", file_path=state["file_path"])

        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    return prepare_sample_call


def _extract_sample(state: CsvRepairWorkflowState) -> dict[str, Any]:
    content = state["messages"][-1].content
    return {"sample": content if isinstance(content, str) else str(content)}


def _make_diagnose_node(detector: CsvFailureDetector) -> Any:
    def diagnose(state: CsvRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(
            agent="CsvRepairAgent",
            node="diagnose",
            table=None,
        )

        # `detect` (the original, single-value Protocol method) always
        # gives the backward-compatible primary result. `detect_all` is
        # an additive capability some detectors expose (the real
        # `LocalCsvFailureDetector` does) — used when available, with a
        # graceful single-value fallback for any test double that only
        # implements `detect`, so no existing caller can ever break.
        failure_class = detector.detect(state["file_path"])
        detect_all = getattr(detector, "detect_all", None)
        if callable(detect_all):
            failure_classes: frozenset[FailureClass] = detect_all(state["file_path"])
        else:
            failure_classes = frozenset({failure_class}) if failure_class is not None else frozenset()

        logger.info(
            "node_completed",
            file_path=state["file_path"],
            failure_class=(failure_class.value if failure_class is not None else None),
            failure_classes=sorted(c.value for c in failure_classes),
        )

        return {"failure_class": failure_class, "failure_classes": failure_classes}

    return diagnose


def _route_after_diagnose(state: CsvRepairWorkflowState) -> str:
    return "propose" if state["failure_class"] is not None else "healthy"


def _detect_encoding(file_path: str) -> chardet.DetectionDict | None:
    """Best-effort, deterministic byte-level encoding detection.

    Reads `file_path`'s raw bytes and runs `chardet.detect` (no LLM call,
    no randomness). Returns `None` on any I/O failure or when chardet
    can't name an encoding, so callers can always fall back gracefully
    without special-casing failure.
    """
    try:
        with open(file_path, "rb") as fh:
            raw = fh.read()
        detected = chardet.detect(raw)
    except Exception:  # noqa: BLE001 - best-effort: must never block the LLM proposal
        return None

    if detected["encoding"] is None:
        return None
    return detected


def _make_propose_node(llm_port: CsvRepairProposalPort) -> Any:
    def propose(state: CsvRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="CsvRepairAgent", node="propose", table=None)

        failure_class = state["failure_class"]
        assert failure_class is not None  # guaranteed by _route_after_diagnose
        failure_classes = state["failure_classes"] or frozenset({failure_class})
        sample = state["sample"] or ""

        if len(failure_classes) > 1:
            # Multi-failure evidence: tell the LLM about every detected
            # dimension so it proposes one combined prescription, not
            # just a fix for the (arbitrary) primary `failure_class`.
            other_names = ", ".join(
                sorted(c.value for c in failure_classes if c is not failure_class)
            )
            sample = (
                f"Multiple simultaneous failures detected: {failure_class.value}, "
                f"{other_names}. Propose a single combined set of repair "
                f"parameters that addresses ALL of them at once.\n{sample}"
            )

        detected_encoding: str | None = None
        if FailureClass.WRONG_ENCODING in failure_classes:
            detected = _detect_encoding(state["file_path"])
            if detected is not None:
                detected_encoding = detected["encoding"]
                sample = (
                    f"Deterministic encoding detection (chardet): {detected_encoding} "
                    f"(confidence={detected['confidence']:.2f})\n{sample}"
                )
        raw = llm_port.propose(
            failure_class=failure_class,
            sample=sample,
            file_path=state["file_path"],
        )
        if detected_encoding is not None and isinstance(raw, dict):
            # The LLM still proposes every other field; only `encoding` is
            # deterministically known from the actual bytes, so it — and
            # only it — is corrected here. The result still flows through
            # the unchanged `validate` node before ever reaching `apply`.
            raw = {**raw, "encoding": detected_encoding}

        logger.info(
            "node_completed",
            failure_class=failure_class.value,
            failure_classes=sorted(c.value for c in failure_classes),
            proposal=raw,
        )

        return {"proposed_params": raw}

    return propose


def _validate(state: CsvRepairWorkflowState) -> dict[str, Any]:
    logger = get_logger(agent="CsvRepairAgent", node="validate", table=None)

    raw = state["proposed_params"] or {}
    validated_params: CsvRepairParams | None
    validation_errors: list[str]
    try:
        validated_params = CsvRepairParams.model_validate(raw)
        validation_errors = []
    except ValidationError as exc:
        validated_params = None
        validation_errors = [
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()
        ]

    logger.info(
        "node_completed",
        validation_status="valid" if validated_params is not None else "invalid",
        validation_errors=validation_errors,
    )

    return {"validated_params": validated_params, "validation_errors": validation_errors}


def _route_after_validation(state: CsvRepairWorkflowState) -> str:
    if state["validated_params"] is not None:
        # Single-failure repairs auto-apply exactly as before (backward
        # compatible). Only a genuine multi-failure episode requires
        # human approval before apply.
        return "human_approval" if len(state["failure_classes"]) > 1 else "apply"
    return "retry" if state["retry_count"] < state["max_retries"] else "fail"


def _make_human_approval_node(approval_port: CsvHumanApprovalPort | None) -> Any:
    def human_approval(state: CsvRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="CsvRepairAgent", node="human_approval", table=None)
        params = state["validated_params"]
        assert params is not None  # guaranteed by _route_after_validation

        if approval_port is None:
            # Fail-safe: a multi-failure repair must never silently
            # bypass approval just because no approval mechanism was
            # wired in — treat the absence of a port as a rejection.
            approved = False
        else:
            request = CsvApprovalRequest(
                file_path=state["file_path"],
                failure_classes=state["failure_classes"],
                prescription=params,
            )
            approved = approval_port.request_approval(request)

        logger.info("node_completed", human_approved=approved)

        return {"human_approved": approved}

    return human_approval


def _route_after_human_approval(state: CsvRepairWorkflowState) -> str:
    return "apply" if state["human_approved"] else "rejected"


def _make_apply_node(executor: CsvRepairExecutor) -> Any:
    def apply(state: CsvRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="CsvRepairAgent", node="apply", table=None)

        params = state["validated_params"]
        assert params is not None  # guaranteed by _route_after_validation
        outcome = executor.execute(state["file_path"], params)
        result = RepairResult(
            success=outcome.success,
            applied=outcome.success,
            confidence=outcome.confidence,
            prescription=params,
            validation_errors=outcome.validation_errors,
            message=outcome.message,
        )

        logger.info(
            "node_completed",
            success=result.success,
            applied=result.applied,
            message=result.message,
        )

        return {"repair_result": result}

    return apply


def _make_reverify_node(executor: CsvRepairExecutor) -> Any:
    def reverify(state: CsvRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="CsvRepairAgent", node="reverify", table=None)

        params = state["validated_params"]
        assert params is not None  # guaranteed by _route_after_validation
        outcome = executor.execute(state["file_path"], params)

        logger.info(
            "node_completed",
            success=outcome.success,
            message=outcome.message,
        )

        return {"verification_result": outcome}

    return reverify


def _route_after_verification(state: CsvRepairWorkflowState) -> str:
    outcome = state["verification_result"]
    if outcome is not None and outcome.success:
        return "verified"
    return "retry" if state["retry_count"] < state["max_retries"] else "fail"


def _increment_retry(state: CsvRepairWorkflowState) -> dict[str, Any]:
    return {"retry_count": state["retry_count"] + 1}


def _set_success(state: CsvRepairWorkflowState) -> dict[str, Any]:
    logger = get_logger(agent="CsvRepairAgent", node="set_success", table=None)
    logger.info("repair_completed", success=True)
    return {"status": RepairEpisodeStatus.SUCCEEDED, "error_message": None}


def _set_failure(state: CsvRepairWorkflowState) -> dict[str, Any]:
    logger = get_logger(agent="CsvRepairAgent", node="set_failure", table=None)
    reason = "; ".join(state["validation_errors"]) or "verification did not succeed"

    failure_class = state["failure_class"]
    logger.info(
        "repair_completed",
        success=False,
        failure_class=failure_class.value if failure_class is not None else None,
        validation_errors=state["validation_errors"],
    )

    return {"status": RepairEpisodeStatus.FAILED, "error_message": reason}


def _set_rejected(state: CsvRepairWorkflowState) -> dict[str, Any]:
    logger = get_logger(agent="CsvRepairAgent", node="set_rejected", table=None)
    logger.info(
        "repair_completed",
        success=False,
        failure_classes=sorted(c.value for c in state["failure_classes"]),
    )
    return {
        "status": RepairEpisodeStatus.REJECTED,
        "error_message": "human rejected the proposed combined repair",
    }


def build_csv_repair_workflow(
    *,
    detector: CsvFailureDetector,
    executor: CsvRepairExecutor,
    llm_port: CsvRepairProposalPort,
    approval_port: CsvHumanApprovalPort | None = None,
    max_sample_lines: int = DEFAULT_MAX_SAMPLE_LINES,
) -> CompiledStateGraph[CsvRepairWorkflowState, None, Any, Any]:
    """Build and compile the Tier 1 CSV repair `StateGraph`.

    `approval_port` is only ever consulted for a genuine multi-failure
    episode (`len(failure_classes) > 1`) — a single-failure repair
    auto-applies exactly as it always has, so omitting `approval_port`
    reproduces prior behavior exactly for every existing caller.

    `detector`, `executor`, and `llm_port` are injected ports (Dependency
    Inversion) — this function contains no pandas, filesystem repair, or
    LLM-provider code of its own.
    """
    graph = StateGraph(CsvRepairWorkflowState)

    graph.add_node("prepare_sample_call", _make_prepare_sample_call_node(max_sample_lines))
    graph.add_node(SAMPLE_TOOL_NODE_NAME, ToolNode([sample_csv_file], name=SAMPLE_TOOL_NODE_NAME))
    graph.add_node("extract_sample", _extract_sample)
    graph.add_node("diagnose", _make_diagnose_node(detector))
    graph.add_node("propose", _make_propose_node(llm_port))
    graph.add_node("validate", _validate)
    graph.add_node("human_approval", _make_human_approval_node(approval_port))
    graph.add_node("apply", _make_apply_node(executor))
    graph.add_node("reverify", _make_reverify_node(executor))
    graph.add_node("increment_retry", _increment_retry)
    graph.add_node("set_success", _set_success)
    graph.add_node("set_failure", _set_failure)
    graph.add_node("set_rejected", _set_rejected)

    graph.add_edge(START, "prepare_sample_call")
    graph.add_edge("prepare_sample_call", SAMPLE_TOOL_NODE_NAME)
    graph.add_edge(SAMPLE_TOOL_NODE_NAME, "extract_sample")
    graph.add_edge("extract_sample", "diagnose")
    graph.add_conditional_edges(
        "diagnose", _route_after_diagnose, {"propose": "propose", "healthy": "set_success"}
    )
    graph.add_edge("propose", "validate")
    graph.add_conditional_edges(
        "validate",
        _route_after_validation,
        {
            "apply": "apply",
            "human_approval": "human_approval",
            "retry": "increment_retry",
            "fail": "set_failure",
        },
    )
    graph.add_conditional_edges(
        "human_approval",
        _route_after_human_approval,
        {"apply": "apply", "rejected": "set_rejected"},
    )
    graph.add_edge("apply", "reverify")
    graph.add_conditional_edges(
        "reverify",
        _route_after_verification,
        {"verified": "set_success", "retry": "increment_retry", "fail": "set_failure"},
    )
    graph.add_edge("increment_retry", "propose")
    graph.add_edge("set_success", END)
    graph.add_edge("set_failure", END)
    graph.add_edge("set_rejected", END)

    return graph.compile()
