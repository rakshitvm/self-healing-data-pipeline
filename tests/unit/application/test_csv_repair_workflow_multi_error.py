"""Focused unit tests for multi-failure CSV repair.

Kept separate from `test_csv_repair_workflow.py` (left untouched). Uses
the real `LocalCsvFailureDetector`/`PandasCsvRepairExecutor` against the
real, shared multi-error fixture (`tests/fixtures/multi_error_csv.py`) —
no mocking of the detector, no fake sample content. No real Groq/Azure
network access: a fake `CsvRepairProposalPort` stands in for the LLM.

Note on "repair ordering" (requirement 7 of the multi-error ticket): CSV
repair, unlike Tier 2 schema repair, has no sequence of discrete
operations to order — every Tier 1 dimension (encoding, delimiter,
header_row, engine) is just a field of one `CsvRepairParams` object
applied via a single `pd.read_csv(...)` call. "Ordering" therefore means
the deterministic *evidence/correction* priority already established for
single-failure WRONG_ENCODING (chardet corrects only the `encoding`
field, unconditionally, after the LLM responds) — extended here to fire
whenever WRONG_ENCODING is one of several simultaneous failures, so it
can never be lost regardless of what else is being repaired at the same
time. `test_encoding_correction_does_not_overwrite_other_fields` proves
this directly (requirement 9).
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
from self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor import (
    PandasCsvRepairExecutor,
)
from tests.fixtures.multi_error_csv import two_failure_bytes

VALID_PROPOSAL = {"delimiter": ";", "encoding": "utf-8", "header_row": 0, "engine": "python"}
COMBINED_PROPOSAL = {"delimiter": ";", "encoding": "latin-1", "header_row": 0, "engine": "python"}


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


def _write(tmp_path: Path, name: str, content: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


def _graph(llm_port: Any, approval_port: Any = None, executor: Any = None) -> Any:
    return build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=executor or PandasCsvRepairExecutor(),
        llm_port=llm_port,
        approval_port=approval_port,
    )


# --- zero / one failure (backward compatibility) --------------------------


def test_healthy_file_short_circuits_without_llm_or_approval(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "healthy.csv", b"id,name,value\n1,alpha,10\n2,beta,20\n")
    llm_port = _FakeProposalPort(VALID_PROPOSAL)
    approval_port = _RecordingApprovalPort(approved=True)
    graph = _graph(llm_port, approval_port)

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_class"] is None
    assert result["failure_classes"] == frozenset()
    assert llm_port.samples == []
    assert approval_port.requests == []


def test_single_failure_never_invokes_approval_port(tmp_path: Path) -> None:
    """Backward compatibility: an ordinary single-failure file must
    auto-apply exactly as before Tier 1's multi-error support existed —
    the approval port must never even be consulted."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", b"id;name;value\n1;alpha;10\n2;beta;20\n")
    approval_port = _RecordingApprovalPort(approved=False)  # would reject if ever asked
    graph = _graph(_FakeProposalPort(VALID_PROPOSAL), approval_port)

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_classes"] == frozenset({FailureClass.WRONG_DELIMITER})
    assert approval_port.requests == []
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED


# --- two simultaneous failures --------------------------------------------


def test_two_simultaneous_failures_are_detected_and_evidenced_to_the_llm(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "multi_error.csv", two_failure_bytes())
    llm_port = _FakeProposalPort(COMBINED_PROPOSAL)
    graph = _graph(llm_port, _RecordingApprovalPort(approved=True))

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_classes"] == frozenset(
        {FailureClass.WRONG_ENCODING, FailureClass.WRONG_DELIMITER}
    )
    assert len(llm_port.samples) == 1
    assert "Multiple simultaneous failures detected" in llm_port.samples[0]
    assert "wrong_delimiter" in llm_port.samples[0]
    assert "Deterministic encoding detection (chardet)" in llm_port.samples[0]


def test_two_simultaneous_failures_require_human_approval_before_apply(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "multi_error.csv", two_failure_bytes())
    approval_port = _RecordingApprovalPort(approved=True)
    graph = _graph(_FakeProposalPort(COMBINED_PROPOSAL), approval_port)

    result = graph.invoke(build_initial_state(file_path))

    assert len(approval_port.requests) == 1
    request = approval_port.requests[0]
    assert request.failure_classes == frozenset(
        {FailureClass.WRONG_ENCODING, FailureClass.WRONG_DELIMITER}
    )
    assert request.prescription.delimiter == ";"
    assert result["human_approved"] is True
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].success is True


def test_human_rejection_prevents_apply_for_combined_repair(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "multi_error.csv", two_failure_bytes())
    original_bytes = two_failure_bytes()
    graph = _graph(_FakeProposalPort(COMBINED_PROPOSAL), _RecordingApprovalPort(approved=False))

    result = graph.invoke(build_initial_state(file_path))

    assert result["human_approved"] is False
    assert result["status"] == RepairEpisodeStatus.REJECTED
    assert result["repair_result"] is None  # apply never ran
    assert result["verification_result"] is None
    assert Path(file_path).read_bytes() == original_bytes  # untouched


def test_no_approval_port_wired_fails_safe_for_multi_error(tmp_path: Path) -> None:
    """Omitting `approval_port` must never silently auto-apply a
    multi-failure repair — it must fail safe (treated as rejected)."""
    file_path = _write(tmp_path, "multi_error.csv", two_failure_bytes())
    graph = _graph(_FakeProposalPort(COMBINED_PROPOSAL), approval_port=None)

    result = graph.invoke(build_initial_state(file_path))

    assert result["human_approved"] is False
    assert result["status"] == RepairEpisodeStatus.REJECTED
    assert result["repair_result"] is None


def test_combined_prescription_is_one_atomic_csv_repair_params(tmp_path: Path) -> None:
    """CSV repair has no sequence of operations to order (unlike Tier 2's
    rename->cast->drop->add_default): every dimension is a field of one
    `CsvRepairParams`, validated and applied as a single atomic object."""
    file_path = _write(tmp_path, "multi_error.csv", two_failure_bytes())
    graph = _graph(_FakeProposalPort(COMBINED_PROPOSAL), _RecordingApprovalPort(approved=True))

    result = graph.invoke(build_initial_state(file_path))

    assert result["validated_params"] is not None
    assert result["validated_params"].delimiter == ";"
    # `encoding` is deterministically corrected by chardet (matching the
    # established single-failure WRONG_ENCODING behavior) regardless of
    # what the fake LLM proposed — proven separately and precisely by
    # test_encoding_correction_does_not_overwrite_other_fields below.
    assert result["validated_params"].encoding != "utf-8"


def test_encoding_correction_does_not_overwrite_other_fields(tmp_path: Path) -> None:
    """Requirement 9: encoding detection must not be lost when delimiter
    repair is proposed, and vice versa — the LLM's delimiter/header_row/
    engine choices survive the chardet-based encoding correction
    untouched, even though the LLM's own encoding guess is wrong."""
    file_path = _write(tmp_path, "multi_error.csv", two_failure_bytes())
    wrong_encoding_guess = {
        "delimiter": ";",
        "encoding": "utf-8",  # deliberately wrong; chardet must correct only this
        "header_row": 0,
        "engine": "python",
    }
    graph = _graph(_FakeProposalPort(wrong_encoding_guess), _RecordingApprovalPort(approved=True))

    result = graph.invoke(build_initial_state(file_path))

    assert result["proposed_params"] is not None
    assert result["proposed_params"]["delimiter"] == ";"  # untouched, LLM's own value
    assert result["proposed_params"]["header_row"] == 0  # untouched
    assert result["proposed_params"]["engine"] == "python"  # untouched
    assert result["proposed_params"]["encoding"] != "utf-8"  # corrected
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED


def test_malformed_combined_prescription_never_reaches_apply(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "multi_error.csv", two_failure_bytes())
    malformed_proposal = {"encoding": "latin-1"}  # missing required delimiter
    approval_port = _RecordingApprovalPort(approved=True)
    graph = _graph(_FakeProposalPort(malformed_proposal), approval_port)

    result = graph.invoke(build_initial_state(file_path, max_retries=0))

    assert result["validated_params"] is None
    assert result["status"] == RepairEpisodeStatus.FAILED
    assert approval_port.requests == []  # never reached human_approval
    assert result["repair_result"] is None
