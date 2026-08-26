"""Focused unit tests for MIXED_DELIMITER support in the Tier 1
`csv_repair_workflow`.

Mirrors `test_csv_repair_workflow_multi_error.py`'s conventions exactly:
real `LocalCsvFailureDetector`, real `MixedDelimiterCsvRepairExecutor`
(and, for the "LLM override" tests, the real `PandasCsvRepairExecutor`
too, to prove which executor is actually selected), fake LLM proposal
port, fake approval port. No real Groq/Azure network access. Includes
the exact panel fixture as an explicit end-to-end regression test.
"""

from pathlib import Path
from typing import Any

from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    build_csv_repair_workflow,
    build_initial_state,
)
from self_healing_pipeline.domain.interfaces.services.csv_approval_port import CsvApprovalRequest
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.pipeline_status import RepairEpisodeStatus
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
)
from self_healing_pipeline.infrastructure.csv.mixed_delimiter_row_repair_executor import (
    MixedDelimiterCsvRepairExecutor,
)
from self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor import (
    PandasCsvRepairExecutor,
)

# Deliberately wrong on both counts the deterministic evidence must
# override: a comma-only file with a semicolon row should never end up
# with delimiter=";" (that would break every other row), and the LLM is
# never even asked for row-level information in the first place.
LLM_PROPOSAL_THAT_MUST_BE_OVERRIDDEN = {
    "delimiter": ";",
    "encoding": "utf-8",
    "header_row": 0,
    "engine": "python",
}


class _FakeProposalPort:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.samples: list[str] = []

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        self.samples.append(sample)
        return dict(self._payload)


class _RecordingApprovalPort:
    def __init__(self, approved: bool) -> None:
        self._approved = approved
        self.requests: list[CsvApprovalRequest] = []

    def request_approval(self, request: CsvApprovalRequest) -> bool:
        self.requests.append(request)
        return self._approved


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _graph(llm_port: Any, approval_port: Any) -> Any:
    return build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=llm_port,
        approval_port=approval_port,
        mixed_delimiter_executor=MixedDelimiterCsvRepairExecutor(),
    )


PANEL_CSV = (
    "id,name,age\n"
    "1,Alice,30\n2,Bob,25\n3,Charlie,35\n4,Diana,28\n5,Ethan,40\n"
    "6;Fiona;33\n"
    "7,George,29\n8,Hannah,31\n9,Ian,26\n10,Julia,37\n"
)


# --- deterministic evidence reaches the LLM sample, never the whole file ---


def test_evidence_reaches_llm_sample_without_sending_the_whole_file(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "panel.csv", PANEL_CSV)
    llm_port = _FakeProposalPort(LLM_PROPOSAL_THAT_MUST_BE_OVERRIDDEN)
    graph = _graph(llm_port, _RecordingApprovalPort(approved=True))

    graph.invoke(build_initial_state(file_path))

    assert len(llm_port.samples) == 1
    sample = llm_port.samples[0]
    assert "Deterministic mixed-delimiter detection" in sample
    assert "6;Fiona;33" in sample
    # only the flagged row's evidence is added — not the surrounding
    # healthy rows 8/9/10, which are far outside the 5-line sample cap
    assert "9,Ian,26" not in sample
    assert "10,Julia,37" not in sample


# --- LLM's own delimiter/row opinions are discarded -------------------------


def test_llm_delimiter_and_row_opinions_are_overridden_by_detector_evidence(
    tmp_path: Path,
) -> None:
    file_path = _write(tmp_path, "panel.csv", PANEL_CSV)
    llm_port = _FakeProposalPort(LLM_PROPOSAL_THAT_MUST_BE_OVERRIDDEN)
    graph = _graph(llm_port, _RecordingApprovalPort(approved=True))

    result = graph.invoke(build_initial_state(file_path))

    params = result["validated_params"]
    assert params is not None
    assert params.delimiter == ","  # NOT ";" — the LLM's own guess is discarded
    assert len(params.mixed_delimiter_rows) == 1
    assert params.mixed_delimiter_rows[0].row_number == 7
    assert params.mixed_delimiter_rows[0].observed_delimiter == ";"


# --- human approval receives the row-level evidence -------------------------


def test_human_approval_receives_row_level_evidence(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "panel.csv", PANEL_CSV)
    approval_port = _RecordingApprovalPort(approved=True)
    graph = _graph(_FakeProposalPort(LLM_PROPOSAL_THAT_MUST_BE_OVERRIDDEN), approval_port)

    graph.invoke(build_initial_state(file_path))

    assert len(approval_port.requests) == 1
    request = approval_port.requests[0]
    assert len(request.mixed_delimiter_rows) == 1
    row = request.mixed_delimiter_rows[0]
    assert row.row_number == 7
    assert row.original_text == "6;Fiona;33"
    assert row.repaired_text == "6,Fiona,33"


def test_rejection_prevents_apply_and_leaves_source_untouched(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "panel.csv", PANEL_CSV)
    original_bytes = Path(file_path).read_bytes()
    graph = _graph(
        _FakeProposalPort(LLM_PROPOSAL_THAT_MUST_BE_OVERRIDDEN),
        _RecordingApprovalPort(approved=False),
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["status"] == RepairEpisodeStatus.REJECTED
    assert result["repair_result"] is None
    assert Path(file_path).read_bytes() == original_bytes


# --- only the flagged row changes; reverify catches an incomplete repair ---


def test_reverify_fails_if_the_mixed_delimiter_executor_is_not_wired_in(tmp_path: Path) -> None:
    """Fail-safe proof: omitting `mixed_delimiter_executor` must never
    silently fall back to the whole-file `PandasCsvRepairExecutor`
    (which cannot apply per-row exceptions) — it must fail explicitly."""
    file_path = _write(tmp_path, "panel.csv", PANEL_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(LLM_PROPOSAL_THAT_MUST_BE_OVERRIDDEN),
        approval_port=_RecordingApprovalPort(approved=True),
        # mixed_delimiter_executor intentionally omitted
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["status"] == RepairEpisodeStatus.FAILED
    assert result["repair_result"] is not None
    assert result["repair_result"].success is False


# --- the exact panel fixture, end to end ------------------------------------


def test_panel_scenario_end_to_end_only_fionas_row_is_repaired(tmp_path: Path) -> None:
    """The exact fixture from the panel test case. Proves the full chain:
    detect -> MIXED_DELIMITER -> propose (evidence overrides the LLM) ->
    validate -> human_approval -> apply (row-level executor) -> reverify
    (every row checked) -> succeeded, with only row 7 changed and every
    other row byte-identical to the source."""
    file_path = _write(tmp_path, "panel.csv", PANEL_CSV)
    graph = _graph(
        _FakeProposalPort(LLM_PROPOSAL_THAT_MUST_BE_OVERRIDDEN),
        _RecordingApprovalPort(approved=True),
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_class"] == FailureClass.MIXED_DELIMITER
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["verification_result"] is not None
    assert result["verification_result"].success is True

    repair_result = result["repair_result"]
    assert repair_result is not None
    assert repair_result.success is True
    assert repair_result.applied is True
    assert repair_result.output_path is not None

    repaired_lines = Path(repair_result.output_path).read_text(encoding="utf-8").splitlines()
    source_lines = [
        line for line in PANEL_CSV.splitlines() if line.strip()
    ]
    assert len(repaired_lines) == len(source_lines)
    for i, (repaired, source) in enumerate(zip(repaired_lines, source_lines, strict=True)):
        if source == "6;Fiona;33":
            assert repaired == "6,Fiona,33"
        else:
            assert repaired == source, f"row {i + 1} was unexpectedly rewritten"

    # source file itself was never touched
    assert Path(file_path).read_text(encoding="utf-8") == PANEL_CSV
