"""Tier 1 CSV repair workflow: sample -> diagnose -> propose -> validate -> apply -> re-verify.

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
  without ever invoking the proposal port.
- `propose`: the reasoning boundary. Delegates to `CsvRepairProposalPort`
  and stores its *raw, unvalidated* output.
- `validate`: the only thing allowed to turn a raw proposal into a
  `CsvRepairParams` that `apply` may use. An invalid proposal never
  reaches `apply`.
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

from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

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

SAMPLE_TOOL_NODE_NAME = "sample_tool"
DEFAULT_MAX_SAMPLE_LINES = 5
DEFAULT_MAX_RETRIES = 2


@tool
def sample_csv_file(file_path: str, max_lines: int) -> str:
    """Read up to `max_lines` lines from the start of a local CSV file."""
    lines: list[str] = []
    with open(file_path, encoding="utf-8", errors="replace") as fh:
        for _, line in zip(range(max_lines), fh, strict=False):
            lines.append(line)
    return "".join(lines)


class CsvRepairWorkflowState(TypedDict):
    """State threaded through the Tier 1 CSV repair workflow."""

    episode_id: UUID
    file_path: str
    failure_class: FailureClass | None
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
        messages=[],
    )


def _make_prepare_sample_call_node(max_sample_lines: int) -> Any:
    def prepare_sample_call(state: CsvRepairWorkflowState) -> dict[str, Any]:
        call = ToolCall(
            name=sample_csv_file.name,
            args={"file_path": state["file_path"], "max_lines": max_sample_lines},
            id="sample-call",
            type="tool_call",
        )
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    return prepare_sample_call


def _extract_sample(state: CsvRepairWorkflowState) -> dict[str, Any]:
    content = state["messages"][-1].content
    return {"sample": content if isinstance(content, str) else str(content)}


def _make_diagnose_node(detector: CsvFailureDetector) -> Any:
    def diagnose(state: CsvRepairWorkflowState) -> dict[str, Any]:
        return {"failure_class": detector.detect(state["file_path"])}

    return diagnose


def _route_after_diagnose(state: CsvRepairWorkflowState) -> str:
    return "propose" if state["failure_class"] is not None else "healthy"


def _make_propose_node(llm_port: CsvRepairProposalPort) -> Any:
    def propose(state: CsvRepairWorkflowState) -> dict[str, Any]:
        failure_class = state["failure_class"]
        assert failure_class is not None  # guaranteed by _route_after_diagnose
        raw = llm_port.propose(
            failure_class=failure_class,
            sample=state["sample"] or "",
            file_path=state["file_path"],
        )
        return {"proposed_params": raw}

    return propose


def _validate(state: CsvRepairWorkflowState) -> dict[str, Any]:
    raw = state["proposed_params"] or {}
    try:
        params = CsvRepairParams.model_validate(raw)
    except ValidationError as exc:
        errors = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return {"validated_params": None, "validation_errors": errors}
    return {"validated_params": params, "validation_errors": []}


def _route_after_validation(state: CsvRepairWorkflowState) -> str:
    if state["validated_params"] is not None:
        return "apply"
    return "retry" if state["retry_count"] < state["max_retries"] else "fail"


def _make_apply_node(executor: CsvRepairExecutor) -> Any:
    def apply(state: CsvRepairWorkflowState) -> dict[str, Any]:
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
        return {"repair_result": result}

    return apply


def _make_reverify_node(executor: CsvRepairExecutor) -> Any:
    def reverify(state: CsvRepairWorkflowState) -> dict[str, Any]:
        params = state["validated_params"]
        assert params is not None  # guaranteed by _route_after_validation
        outcome = executor.execute(state["file_path"], params)
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
    return {"status": RepairEpisodeStatus.SUCCEEDED, "error_message": None}


def _set_failure(state: CsvRepairWorkflowState) -> dict[str, Any]:
    reason = "; ".join(state["validation_errors"]) or "verification did not succeed"
    return {"status": RepairEpisodeStatus.FAILED, "error_message": reason}


def build_csv_repair_workflow(
    *,
    detector: CsvFailureDetector,
    executor: CsvRepairExecutor,
    llm_port: CsvRepairProposalPort,
    max_sample_lines: int = DEFAULT_MAX_SAMPLE_LINES,
) -> CompiledStateGraph[CsvRepairWorkflowState, None, Any, Any]:
    """Build and compile the Tier 1 CSV repair `StateGraph`.

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
    graph.add_node("apply", _make_apply_node(executor))
    graph.add_node("reverify", _make_reverify_node(executor))
    graph.add_node("increment_retry", _increment_retry)
    graph.add_node("set_success", _set_success)
    graph.add_node("set_failure", _set_failure)

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
        {"apply": "apply", "retry": "increment_retry", "fail": "set_failure"},
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

    return graph.compile()
