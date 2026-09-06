"""Tier 1 CSV repair workflow: sample -> diagnose -> propose -> validate -> [human_approval] -> apply -> re-verify.

A LangGraph `StateGraph` orchestrating the Tier 1 building blocks
(`CsvFailureDetector`, `CsvRepairExecutor`, and a provider-agnostic
`CsvRepairProposalPort`) into one self-healing workflow. Orchestration
only — no pandas, no filesystem repair logic, no LLM-provider code; every
infrastructure concern is injected via the three ports above. The only
filesystem access here is the sampling tool, which reads (never writes)
a few lines of the target file.

Node responsibilities:

- `prepare_sample_call` / `sample_tool` / `extract_sample`: get a small
  sample of the CSV, deterministically. No LLM call.
- `diagnose`: delegate to `CsvFailureDetector`. No LLM call. A healthy
  file short-circuits straight to success without invoking the proposal
  port. A file can exhibit more than one failure dimension at once
  (`LocalCsvFailureDetector.detect_all`) — `state["failure_classes"]`
  holds the full set, `state["failure_class"]` keeps the single primary
  value other nodes rely on.
- `propose`: the reasoning boundary. Delegates to `CsvRepairProposalPort`,
  stores its raw, unvalidated output. For a multi-failure episode, the
  sample is prefixed with a line listing every detected failure so the
  LLM proposes one combined prescription. Several fields are mechanical
  once a failure is detected and are deterministically corrected after
  the LLM call rather than left to it: `WRONG_ENCODING`'s `encoding`
  (from a `chardet` pass over the raw bytes), `HEADER_DETECTION`'s
  `header_row`, `NO_HEADER`'s `header_row=None`, and `MIXED_DELIMITER`'s
  `delimiter`/`mixed_delimiter_rows`. Each correction only touches its
  own field — the rest of the LLM's proposal passes through unchanged —
  and the result still flows through `validate` before `apply`.
  `INVISIBLE_CHARACTERS` needs no proposal-level correction: the fix is
  unconditional column-name cleanup in the executor itself.
- `validate`: the only place a raw proposal becomes a `CsvRepairParams`
  `apply` may use.
- `human_approval`: reached for every non-healthy repair, single- or
  multi-failure. Delegates to the injected `CsvHumanApprovalPort`; a
  rejection routes to `set_rejected`, `apply` is never called. No
  `approval_port` configured means automatic rejection (fail-safe).
- `apply`: delegates to `CsvRepairExecutor.execute` with the validated
  prescription. `execute` never modifies the source file; on success it
  writes a separate repaired output and reports its path.
  `RepairResult.applied` is only `True` once that file actually exists.
  `MIXED_DELIMITER` prescriptions route to the separately-injected
  `mixed_delimiter_executor` instead of the whole-file executor,
  selected purely by data on `params` — no extra graph nodes/edges.
- `reverify`: independent, read-only check of the repaired output via
  `CsvRepairExecutor.verify` — never the source, never writes. If
  `apply` produced no output, this reports failure without touching any
  file. No LLM call.

Retries are bounded by `max_retries`, driven by conditional edges after
`validate` and `reverify`, both funnelling into `increment_retry` before
looping back to `propose`.
"""

import codecs
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
from self_healing_pipeline.domain.value_objects.mixed_delimiter_row_repair import (
    MixedDelimiterRowRepair,
)
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
    output_path: str | None
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
        output_path=None,
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


# Single-byte Western/Central-European/Baltic codecs chardet tends to
# confuse with one another at low confidence on short, mostly-ASCII
# samples with only a handful of accented characters. A real Latin-1
# file was misidentified as ISO-8859-4 (Baltic) at ~3.6% confidence,
# corrupting `ñ` into `ņ` despite `chardet.detect` reporting success.
# Multi-byte/CJK/Cyrillic encodings aren't part of this and are never
# second-guessed here — they have far more distinctive byte patterns.
_CONFUSABLE_WESTERN_SINGLE_BYTE_ENCODINGS = frozenset(
    {
        "iso88591",
        "iso88592",
        "iso88594",
        "iso88599",
        "iso885913",
        "iso885915",
        "iso885916",
        "windows1250",
        "windows1252",
        "windows1254",
        "windows1257",
        "cp1252",
        "macroman",
    }
)
# Below this, chardet's specific single-byte-family guess isn't trusted
# (see above), but a low-confidence guess still means the file isn't
# plain UTF-8/ASCII — fall back to a safe, superset-compatible default
# (Windows-1252) instead.
_CHARDET_MIN_CONFIDENCE = 0.5
_SAFE_WESTERN_FALLBACK_ENCODING = "cp1252"


def _detect_encoding(file_path: str) -> chardet.DetectionDict | None:
    """Best-effort, deterministic byte-level encoding detection.

    Reads `file_path`'s raw bytes and runs `chardet.detect` (no LLM call,
    no randomness). Returns `None` on any I/O failure or when chardet
    can't name an encoding, so callers can always fall back gracefully
    without special-casing failure.

    If chardet names a low-confidence,
    `_CONFUSABLE_WESTERN_SINGLE_BYTE_ENCODINGS` guess, the returned
    `"encoding"` is corrected to `_SAFE_WESTERN_FALLBACK_ENCODING`
    instead (see the constants above for why) — `"confidence"` is left
    exactly as chardet reported it, so callers displaying it as evidence
    still show the real (low) number.
    """
    try:
        with open(file_path, "rb") as fh:
            raw = fh.read()
        detected = chardet.detect(raw)
    except Exception:  # noqa: BLE001 - best-effort: must never block the LLM proposal
        return None

    if detected["encoding"] is None:
        return None

    # Canonicalize by stripping every non-alphanumeric character before
    # comparing — chardet, Python's codec registry, and this module's
    # own constants don't agree on hyphens/underscores (chardet itself
    # returns `"iso8859-4"`, not `"ISO-8859-4"`), so an exact string
    # match would silently never fire.
    normalized = "".join(ch for ch in detected["encoding"].lower() if ch.isalnum())
    if (
        detected["confidence"] < _CHARDET_MIN_CONFIDENCE
        and normalized in _CONFUSABLE_WESTERN_SINGLE_BYTE_ENCODINGS
        and not _same_codec(detected["encoding"], _SAFE_WESTERN_FALLBACK_ENCODING)
    ):
        detected = {**detected, "encoding": _SAFE_WESTERN_FALLBACK_ENCODING}

    return detected


def _same_codec(left: str, right: str) -> bool:
    """`True` if `left` and `right` name the same underlying codec (e.g.
    `"Windows-1252"` and `"cp1252"` are aliases for one identical codec)
    — used so a low-confidence guess that's already functionally
    equivalent to the safe fallback is left exactly as chardet reported
    it, rather than being needlessly renamed."""
    try:
        return codecs.lookup(left).name == codecs.lookup(right).name
    except LookupError:
        return False


def _make_propose_node(llm_port: CsvRepairProposalPort, detector: CsvFailureDetector) -> Any:
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

        # MIXED_DELIMITER evidence: computed deterministically over the
        # WHOLE file by the detector (never limited by the sample cap
        # above), and only the flagged rows' evidence — not the whole
        # file — is added to what the LLM sees. Neither the established
        # delimiter nor which rows are affected is ever left to the LLM
        # to discover or decide; both are force-corrected below,
        # regardless of what (if anything) the LLM proposes for them.
        established_delimiter: str | None = None
        mixed_delimiter_rows: tuple[MixedDelimiterRowRepair, ...] = ()
        if FailureClass.MIXED_DELIMITER in failure_classes:
            detect_evidence = getattr(detector, "detect_mixed_delimiter_evidence", None)
            if callable(detect_evidence):
                established_delimiter, mixed_delimiter_rows = detect_evidence(
                    state["file_path"]
                )
                if mixed_delimiter_rows:
                    evidence_lines = "\n".join(
                        f"  Row {r.row_number}: {r.original_text!r} — established "
                        f"delimiter {established_delimiter!r} expects "
                        f"{r.expected_field_count} fields, found {r.actual_field_count}; "
                        f"observed delimiter {r.observed_delimiter!r} produces "
                        f"{r.expected_field_count} fields."
                        for r in mixed_delimiter_rows
                    )
                    sample = (
                        "Deterministic mixed-delimiter detection: "
                        f"{len(mixed_delimiter_rows)} row(s) do not match the "
                        f"established delimiter {established_delimiter!r}:\n"
                        f"{evidence_lines}\n{sample}"
                    )

        # HEADER_DETECTION evidence: the number of leading preamble lines
        # to skip is computed deterministically over the WHOLE file (see
        # `detect_header_offset_evidence`), never limited by the sample
        # cap above — a multi-line preamble can push the real header
        # beyond what a 5-line sample even shows the LLM.
        header_offset: int | None = None
        garbled_header_delimiter: str | None = None
        garbled_header_repair: MixedDelimiterRowRepair | None = None
        if FailureClass.HEADER_DETECTION in failure_classes:
            detect_offset = getattr(detector, "detect_header_offset_evidence", None)
            if callable(detect_offset):
                header_offset = detect_offset(state["file_path"])
            # The header line itself might be malformed by several
            # different delimiter characters used interchangeably (e.g.
            # `id;name|age,state,country`) rather than genuinely being
            # junk to skip — see `detect_garbled_header_repair`. Also
            # deterministic, never left to the LLM to guess.
            detect_garbled = getattr(detector, "detect_garbled_header_repair", None)
            if callable(detect_garbled):
                garbled_result = detect_garbled(state["file_path"])
                if garbled_result is not None:
                    garbled_header_delimiter, garbled_header_repair = garbled_result

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
        if mixed_delimiter_rows and isinstance(raw, dict):
            # Same override pattern as WRONG_ENCODING above: both fields
            # are deterministically known, so both are corrected here
            # unconditionally — the LLM's own delimiter guess (and its
            # complete lack of per-row information) is discarded.
            raw = {
                **raw,
                "delimiter": established_delimiter,
                "mixed_delimiter_rows": [r.model_dump() for r in mixed_delimiter_rows],
            }
        if header_offset is not None and isinstance(raw, dict):
            # Same override pattern: the correct header_row is
            # deterministically known once HEADER_DETECTION fires, so the
            # LLM's own guess (which may not even have seen the real
            # header line, given the sample cap) is discarded.
            raw = {**raw, "header_row": header_offset}
        if garbled_header_repair is not None and isinstance(raw, dict):
            # Same override pattern as MIXED_DELIMITER above: a header
            # line using several delimiter characters interchangeably is
            # exactly the kind of thing this project never lets the LLM
            # guess. Independent of (and can fire alongside) the
            # header_offset override above — one corrects *where* the
            # header is, this corrects *what the header line says*.
            raw = {
                **raw,
                "delimiter": garbled_header_delimiter,
                "mixed_delimiter_rows": [garbled_header_repair.model_dump()],
            }
        if FailureClass.NO_HEADER in failure_classes and isinstance(raw, dict):
            # Mechanical, not a judgment call: NO_HEADER means the file
            # has no header row at all, so every row is data — never left
            # to the LLM to guess.
            raw = {**raw, "header_row": None}

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
        # Every non-healthy repair — single-failure or multi-failure —
        # requires explicit human approval before apply. A healthy file
        # never reaches this node at all (it short-circuits at
        # `_route_after_diagnose`), so this does not affect the
        # "no approval for a healthy file" guarantee.
        return "human_approval"
    return "retry" if state["retry_count"] < state["max_retries"] else "fail"


def _make_human_approval_node(approval_port: CsvHumanApprovalPort | None) -> Any:
    def human_approval(state: CsvRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="CsvRepairAgent", node="human_approval", table=None)
        params = state["validated_params"]
        assert params is not None  # guaranteed by _route_after_validation

        if approval_port is None:
            # Fail-safe: a repair must never silently bypass approval
            # just because no approval mechanism was wired in — treat
            # the absence of a port as a rejection.
            approved = False
        else:
            request = CsvApprovalRequest(
                file_path=state["file_path"],
                failure_classes=state["failure_classes"],
                prescription=params,
                mixed_delimiter_rows=params.mixed_delimiter_rows,
            )
            approved = approval_port.request_approval(request)

        logger.info("node_completed", human_approved=approved)

        return {"human_approved": approved}

    return human_approval


def _route_after_human_approval(state: CsvRepairWorkflowState) -> str:
    return "apply" if state["human_approved"] else "rejected"


def _select_executor(
    params: CsvRepairParams,
    executor: CsvRepairExecutor,
    mixed_delimiter_executor: CsvRepairExecutor | None,
) -> tuple[CsvRepairExecutor, CsvExecutionOutcome | None]:
    """Pick which `CsvRepairExecutor` applies/verifies `params`.

    A `mixed_delimiter_rows` prescription routes to the separately
    injected `mixed_delimiter_executor` instead of the whole-file
    `executor`, selected purely by data on `params`. If one reaches here
    without an executor wired in, fail safe rather than falling back to
    the whole-file executor, which can't apply per-row exceptions.
    """
    if not params.mixed_delimiter_rows:
        return executor, None
    if mixed_delimiter_executor is None:
        return executor, CsvExecutionOutcome(
            success=False,
            validation_errors=["no_mixed_delimiter_executor_configured"],
            message=(
                "This prescription carries mixed_delimiter_rows but no "
                "mixed_delimiter_executor was injected into the workflow."
            ),
        )
    return mixed_delimiter_executor, None


def _make_apply_node(
    executor: CsvRepairExecutor, mixed_delimiter_executor: CsvRepairExecutor | None = None
) -> Any:
    def apply(state: CsvRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="CsvRepairAgent", node="apply", table=None)

        params = state["validated_params"]
        assert params is not None  # guaranteed by _route_after_validation
        chosen_executor, failure = _select_executor(params, executor, mixed_delimiter_executor)
        outcome = failure if failure is not None else chosen_executor.execute(
            state["file_path"], params
        )
        result = RepairResult(
            success=outcome.success,
            applied=outcome.success,
            confidence=outcome.confidence,
            prescription=params,
            validation_errors=outcome.validation_errors,
            message=outcome.message,
            source_path=state["file_path"],
            output_path=outcome.output_path,
        )

        logger.info(
            "node_completed",
            success=result.success,
            applied=result.applied,
            source_path=result.source_path,
            output_path=result.output_path,
            message=result.message,
        )

        return {"repair_result": result, "output_path": outcome.output_path}

    return apply


def _make_reverify_node(
    executor: CsvRepairExecutor, mixed_delimiter_executor: CsvRepairExecutor | None = None
) -> Any:
    def reverify(state: CsvRepairWorkflowState) -> dict[str, Any]:
        logger = get_logger(agent="CsvRepairAgent", node="reverify", table=None)

        params = state["validated_params"]
        assert params is not None  # guaranteed by _route_after_validation
        output_path = state["output_path"]
        if output_path is None:
            # apply ran (this node is always reached after apply, win or
            # lose) but produced no repaired output — nothing exists to
            # independently re-check. Report failure without touching
            # any file; never falls back to re-checking the source.
            outcome = CsvExecutionOutcome(
                success=False,
                validation_errors=["no_repaired_output_to_verify"],
                message="No repaired output file was created by apply; nothing to reverify.",
            )
        else:
            chosen_executor, failure = _select_executor(params, executor, mixed_delimiter_executor)
            outcome = failure if failure is not None else chosen_executor.verify(
                output_path, params
            )

        logger.info(
            "node_completed",
            success=outcome.success,
            output_path=output_path,
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
    mixed_delimiter_executor: CsvRepairExecutor | None = None,
) -> CompiledStateGraph[CsvRepairWorkflowState, None, Any, Any]:
    """Build and compile the Tier 1 CSV repair `StateGraph`.

    `approval_port` is consulted for every non-healthy repair; `apply` is
    never reached without an explicit approval. Omitting it fails safe:
    a repair reaching `human_approval` without one is treated as
    rejected.

    `detector`, `executor`, and `llm_port` are injected ports (Dependency
    Inversion). `mixed_delimiter_executor` is a second, optional
    `CsvRepairExecutor` used only for `MIXED_DELIMITER` prescriptions
    (see `_select_executor`); omitting it is a no-op for every other
    failure class and fails safe if a `MIXED_DELIMITER` repair needs it.
    """
    graph = StateGraph(CsvRepairWorkflowState)

    graph.add_node("prepare_sample_call", _make_prepare_sample_call_node(max_sample_lines))
    graph.add_node(SAMPLE_TOOL_NODE_NAME, ToolNode([sample_csv_file], name=SAMPLE_TOOL_NODE_NAME))
    graph.add_node("extract_sample", _extract_sample)
    graph.add_node("diagnose", _make_diagnose_node(detector))
    graph.add_node("propose", _make_propose_node(llm_port, detector))
    graph.add_node("validate", _validate)
    graph.add_node("human_approval", _make_human_approval_node(approval_port))
    graph.add_node("apply", _make_apply_node(executor, mixed_delimiter_executor))
    graph.add_node("reverify", _make_reverify_node(executor, mixed_delimiter_executor))
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
