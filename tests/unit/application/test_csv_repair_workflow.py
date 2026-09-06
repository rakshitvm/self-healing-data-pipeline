"""Focused unit tests for the Tier 1 CSV repair LangGraph workflow.

Uses the real `LocalCsvFailureDetector` (Ticket 006, deterministic,
stdlib-only) against real temporary CSV files, combined with in-file
fakes for `CsvRepairExecutor` and `CsvRepairProposalPort` — no Azure
OpenAI, no network access, no Databricks/Spark/PostgreSQL/MLflow/
Elasticsearch.
"""

from pathlib import Path
from typing import Any

import chardet
import pandas as pd

from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    SAMPLE_TOOL_NODE_NAME,
    build_csv_repair_workflow,
    build_initial_state,
)
from self_healing_pipeline.domain.interfaces.services.csv_repair_executor import (
    CsvExecutionOutcome,
)
from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.pipeline_status import RepairEpisodeStatus
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
)
from self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor import (
    PandasCsvRepairExecutor,
)
from langgraph.prebuilt import ToolNode


class _FakeProposalPort:
    """Fake `CsvRepairProposalPort`: returns a canned raw payload, records calls."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[FailureClass] = []

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        self.calls.append(failure_class)
        return dict(self._payload)


class _FakeCsvRepairExecutor:
    """Fake `CsvRepairExecutor`: returns a canned outcome for both `execute`
    (apply) and `verify` (reverify); records each separately."""

    def __init__(self, outcome: CsvExecutionOutcome) -> None:
        self._outcome = outcome
        self.execute_calls: list[tuple[str, CsvRepairParams]] = []
        self.verify_calls: list[tuple[str, CsvRepairParams]] = []

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        self.execute_calls.append((file_path, params))
        return self._outcome

    def verify(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        self.verify_calls.append((file_path, params))
        return self._outcome


class _ExplodingCsvRepairExecutor:
    """Fake `CsvRepairExecutor` that fails the test if ever called."""

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        raise AssertionError("executor.execute() should not have been called")

    def verify(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        raise AssertionError("executor.verify() should not have been called")


class _AlwaysApproveCsvHumanApprovalPort:
    """Fake `CsvHumanApprovalPort`: always approves.

    Every non-healthy repair now requires explicit human approval before
    apply — this stands in for a human saying "yes" so tests that assert
    a successful end-to-end repair can still reach `apply`/`reverify`.
    """

    def request_approval(self, request: Any) -> bool:
        return True


class _RecordingCsvHumanApprovalPort:
    """Fake `CsvHumanApprovalPort`: records every request; configurable
    approve/reject decision."""

    def __init__(self, approved: bool) -> None:
        self._approved = approved
        self.requests: list[Any] = []

    def request_approval(self, request: Any) -> bool:
        self.requests.append(request)
        return self._approved


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


VALID_PROPOSAL = {"delimiter": ";", "encoding": "utf-8", "header_row": 0, "engine": "python"}
WRONG_DELIMITER_CSV = "id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n"


def test_graph_construction_succeeds() -> None:
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )

    assert graph is not None
    assert "diagnose" in graph.get_graph().nodes


def test_healthy_csv_follows_deterministic_path_without_llm(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n")
    llm_port = _FakeProposalPort(VALID_PROPOSAL)
    executor = _ExplodingCsvRepairExecutor()
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(), executor=executor, llm_port=llm_port
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["failure_class"] is None
    assert llm_port.calls == []  # LLM never invoked for a healthy file
    assert result["repair_result"] is None


def test_wrong_delimiter_flows_through_full_workflow(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_class"] == FailureClass.WRONG_DELIMITER
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].success is True
    assert result["verification_result"] is not None
    assert result["verification_result"].success is True


def test_wrong_encoding_flows_through_workflow(tmp_path: Path) -> None:
    path = tmp_path / "wrong_encoding.csv"
    original_bytes = "id,name,value\n1,café,10\n2,naïve,20\n3,façade,30\n".encode("latin-1")
    path.write_bytes(original_bytes)
    # Captured from the file's *original* bytes, before `apply` physically
    # rewrites it into canonical UTF-8 — chardet run after the repair
    # would (correctly) detect the now-rewritten file's real encoding
    # instead of the original one this test is about.
    expected_encoding = chardet.detect(original_bytes)["encoding"]
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(
            {"delimiter": ",", "encoding": "latin-1", "header_row": 0, "engine": "python"}
        ),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )

    result = graph.invoke(build_initial_state(str(path)))

    assert result["failure_class"] == FailureClass.WRONG_ENCODING
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].prescription is not None
    # `propose` deterministically corrects `encoding` from the file's real
    # (original) bytes via chardet, overriding whatever the (fake) LLM
    # proposed — so the applied encoding is chardet's answer, not the fake
    # port's "latin-1", even though both decode this fixture identically.
    assert result["repair_result"].prescription.encoding == expected_encoding


def test_single_column_malformation_flows_through_workflow(tmp_path: Path) -> None:
    """Irregular whitespace collapses to a single column under
    `LocalCsvFailureDetector` (real, deterministic detection — matching
    Ticket 006's own fixture). No single-character delimiter can repair
    variable-width whitespace, so a fake executor stands in for the apply
    step here, the same honest split used in Ticket 006: this test proves
    the *workflow* correctly routes SINGLE_COLUMN_MALFORMATION end to end,
    not that pandas can fix whitespace-separated data via `sep`.
    """
    file_path = _write(
        tmp_path,
        "single_column.csv",
        "id  name    value\n1   alpha   10\n2  beta     20\n3    gamma  30\n",
    )
    executor = _FakeCsvRepairExecutor(
        CsvExecutionOutcome(success=True, confidence=1.0, output_path=f"{file_path}.repaired")
    )
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=executor,
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_class"] == FailureClass.SINGLE_COLUMN_MALFORMATION
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None and result["repair_result"].success is True


def test_header_detection_flows_through_full_workflow(tmp_path: Path) -> None:
    """Real detector + real executor: a leading junk title line is
    genuinely resolved by skipping it via `header_row`. Matches the
    proposal a real Groq run produced for this exact fixture."""
    file_path = _write(
        tmp_path,
        "header_detection.csv",
        "Sales Report - Q3 2026\nid,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(
            {"delimiter": ",", "encoding": "utf-8", "header_row": 1, "engine": "python"}
        ),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_class"] == FailureClass.HEADER_DETECTION
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].success is True
    assert result["verification_result"] is not None
    assert result["verification_result"].success is True


def test_engine_selection_flows_through_full_workflow(tmp_path: Path) -> None:
    """Real detector + real executor: a row with a missing trailing field
    (fewer fields than the header, not more) is NOT silently treated as
    "genuinely repairable" — `PandasCsvRepairExecutor` independently
    cross-checks every row's raw field count (bypassing pandas' own
    NaN-padding for a short row), so this correctly fails rather than
    reporting a false success. This was a real, previously-undetected
    bug (this test used to assert the opposite): pandas' `c`/`python`
    engines silently pad a short row with NaN rather than raising, which
    made this fixture look "successfully repaired" even though row 2's
    data was actually corrupted, not genuinely fixed.
    """
    file_path = _write(
        tmp_path,
        "engine_selection.csv",
        "id,name,value\n1,alpha,10\n2,beta\n3,gamma,30\n",
    )
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(
            {"delimiter": ",", "encoding": "utf-8", "header_row": 0, "engine": "python"}
        ),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )

    result = graph.invoke(build_initial_state(file_path, max_retries=0))

    assert result["failure_class"] == FailureClass.ENGINE_SELECTION
    assert result["status"] == RepairEpisodeStatus.FAILED
    assert result["repair_result"] is not None
    assert result["repair_result"].success is False
    assert any("row 3" in e for e in result["repair_result"].validation_errors)  # "2,beta"


def test_valid_csv_repair_params_reaches_apply(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    executor = _FakeCsvRepairExecutor(
        CsvExecutionOutcome(success=True, confidence=1.0, output_path=f"{file_path}.repaired")
    )
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=executor,
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )

    graph.invoke(build_initial_state(file_path))

    assert len(executor.execute_calls) == 1  # apply
    assert len(executor.verify_calls) == 1  # reverify
    _, params_passed_to_apply = executor.execute_calls[0]
    assert isinstance(params_passed_to_apply, CsvRepairParams)
    assert params_passed_to_apply.delimiter == ";"


def test_invalid_repair_parameters_never_reach_apply(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    llm_port = _FakeProposalPort({"delimiter": "too-long", "encoding": "utf-8"})
    executor = _ExplodingCsvRepairExecutor()
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(), executor=executor, llm_port=llm_port
    )

    result = graph.invoke(build_initial_state(file_path, max_retries=0))

    assert result["validated_params"] is None
    assert result["validation_errors"]
    assert result["repair_result"] is None
    assert result["status"] == RepairEpisodeStatus.FAILED


def test_verification_failure_retries_when_budget_remains(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    executor = _FakeCsvRepairExecutor(
        CsvExecutionOutcome(success=False, validation_errors=["still broken"])
    )
    llm_port = _FakeProposalPort(VALID_PROPOSAL)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=executor,
        llm_port=llm_port,
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )

    result = graph.invoke(build_initial_state(file_path, max_retries=2))

    assert result["retry_count"] == 2  # exhausted, but only after retrying
    assert len(llm_port.calls) == 3  # initial attempt + 2 retries
    assert result["status"] == RepairEpisodeStatus.FAILED


def test_verification_failure_becomes_terminal_failure_when_budget_exhausted(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    executor = _FakeCsvRepairExecutor(
        CsvExecutionOutcome(success=False, validation_errors=["still broken"])
    )
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=executor,
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )

    result = graph.invoke(build_initial_state(file_path, max_retries=0))

    assert result["retry_count"] == 0
    assert result["status"] == RepairEpisodeStatus.FAILED
    assert result["error_message"] == "verification did not succeed"


# --- Human approval now gates every non-healthy repair, single-failure --
# --- included, not only multi-failure episodes. -------------------------


def test_single_failure_repair_calls_human_approval(tmp_path: Path) -> None:
    """A single-failure repair must now invoke the approval port — this
    was NOT true before this behavioral correction (single-failure used
    to auto-apply)."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    approval_port = _RecordingCsvHumanApprovalPort(approved=True)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=approval_port,
    )

    graph.invoke(build_initial_state(file_path))

    assert len(approval_port.requests) == 1
    request = approval_port.requests[0]
    assert request.file_path == file_path
    assert request.failure_classes == frozenset({FailureClass.WRONG_DELIMITER})
    assert request.prescription.delimiter == ";"


def test_single_failure_approval_yes_reaches_apply_and_succeeds(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_RecordingCsvHumanApprovalPort(approved=True),
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["human_approved"] is True
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].success is True
    assert result["verification_result"] is not None
    assert result["verification_result"].success is True


def test_single_failure_approval_no_does_not_apply(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    executor = _ExplodingCsvRepairExecutor()
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=executor,
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_RecordingCsvHumanApprovalPort(approved=False),
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["human_approved"] is False
    assert result["status"] == RepairEpisodeStatus.REJECTED
    assert result["repair_result"] is None  # apply never ran
    assert result["verification_result"] is None


def test_single_failure_rejection_leaves_file_unchanged(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_RecordingCsvHumanApprovalPort(approved=False),
    )

    graph.invoke(build_initial_state(file_path))

    assert Path(file_path).read_text(encoding="utf-8") == WRONG_DELIMITER_CSV


def test_single_failure_without_approval_port_fails_safe_not_auto_apply(tmp_path: Path) -> None:
    """Regression guard: a future change must not be able to accidentally
    restore auto-apply for single-failure repairs by, e.g., special-casing
    `approval_port is None`. Omitting the port must still fail safe
    (rejected), exactly like the existing multi-failure fail-safe."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=_ExplodingCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["human_approved"] is False
    assert result["status"] == RepairEpisodeStatus.REJECTED
    assert result["repair_result"] is None


# --- The source file is never modified; repairs land in a separate ------
# --- output file (live-demo bug: apply used to overwrite the source). ---


def test_approved_single_failure_repair_creates_output_and_leaves_source_untouched(
    tmp_path: Path,
) -> None:
    """The exact bug found during the live demo: `apply` used to
    physically overwrite the source file in place. An approved repair
    must instead create a *separate* repaired output file, with the
    source left byte-for-byte identical, and the corrected data
    genuinely preserved in the new output file."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].applied is True
    assert result["repair_result"].source_path == file_path

    # The source is completely untouched.
    assert Path(file_path).read_text(encoding="utf-8") == WRONG_DELIMITER_CSV

    # A separate repaired output file exists, preserving the filename.
    output_path = result["repair_result"].output_path
    assert output_path is not None
    assert output_path != file_path
    assert Path(output_path).name == "wrong_delimiter.csv"
    assert result["output_path"] == output_path
    assert result["verification_result"] is not None
    assert result["verification_result"].success is True

    on_disk_output = Path(output_path).read_text(encoding="utf-8")
    assert ";" not in on_disk_output  # the malformed delimiter is gone from the output

    repaired = pd.read_csv(output_path)  # plain defaults: the output is canonical
    assert list(repaired.columns) == ["id", "name", "value"]
    assert repaired.shape == (3, 3)
    assert repaired.iloc[0].tolist() == [1, "alpha", 10]
    assert repaired.iloc[1].tolist() == [2, "beta", 20]
    assert repaired.iloc[2].tolist() == [3, "gamma", 30]


def test_rejected_repair_leaves_file_byte_for_byte_unchanged(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    original_bytes = Path(file_path).read_bytes()
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
        approval_port=_RecordingCsvHumanApprovalPort(approved=False),
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["status"] == RepairEpisodeStatus.REJECTED
    assert Path(file_path).read_bytes() == original_bytes
    # No repaired output is created for a rejected repair.
    assert result["output_path"] is None
    assert not (tmp_path / "repaired").exists()


def test_tool_node_is_present_in_compiled_graph() -> None:
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )

    node = graph.get_graph().nodes[SAMPLE_TOOL_NODE_NAME]
    assert isinstance(node.data, ToolNode)
