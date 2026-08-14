"""Focused unit tests for the WRONG_ENCODING chardet fix.

Covers two layers, both introduced for WRONG_ENCODING only and gated on
`FailureClass.WRONG_ENCODING`:

1. Deterministic chardet evidence added to the *local* sample text handed
   to `CsvRepairProposalPort.propose()` (does not alter `state["sample"]`).
2. A deterministic post-proposal correction: when chardet names an
   encoding, that single field of the LLM's raw proposal is overridden to
   match before `proposed_params` is stored — every other field stays
   exactly what the LLM returned, and the result still flows through the
   unchanged `validate` node before `apply`.

Kept in a separate file from `test_csv_repair_workflow.py` (left
untouched) so this fix's diff is unambiguous. Uses the real
`LocalCsvFailureDetector`/`PandasCsvRepairExecutor` against real temporary
CSV files, combined with spy/fake `CsvRepairProposalPort` doubles — no
real Groq/Azure network access, no API key.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import chardet

from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    _detect_encoding,
    build_csv_repair_workflow,
    build_initial_state,
)
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.pipeline_status import RepairEpisodeStatus
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
)
from self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor import (
    PandasCsvRepairExecutor,
)

VALID_PROPOSAL = {"delimiter": ";", "encoding": "utf-8", "header_row": 0, "engine": "python"}
WRONG_DELIMITER_CSV = "id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n"
LATIN1_CONTENT = "id,name,value\n1,café,10\n2,naïve,20\n3,façade,30\n"
_EVIDENCE_MARKER = "Deterministic encoding detection (chardet):"


class _SpyProposalPort:
    """Fake `CsvRepairProposalPort`: records every `sample` it is called
    with and always returns the same canned (possibly wrong) payload —
    mirrors the real Groq behavior observed in the live E2E run, which
    kept proposing `encoding="utf-8"` regardless of the sample content."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.samples: list[str] = []

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        self.samples.append(sample)
        return dict(self._payload)


class _EvidenceAwareProposalPort:
    """Fake `CsvRepairProposalPort` that reads the chardet evidence line out
    of `sample` and proposes the encoding it names — mimicking what a
    competent real LLM should do, without any real network call."""

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        assert _EVIDENCE_MARKER in sample
        evidence_line = next(line for line in sample.splitlines() if _EVIDENCE_MARKER in line)
        encoding = evidence_line.split(_EVIDENCE_MARKER, 1)[1].strip().split(" ", 1)[0]
        return {"delimiter": ",", "encoding": encoding, "header_row": 0, "engine": "python"}


class _AlwaysApproveCsvHumanApprovalPort:
    """Fake `CsvHumanApprovalPort`: always approves — every non-healthy
    repair now requires explicit human approval before apply."""

    def request_approval(self, request: Any) -> bool:
        return True


def _write_latin1(tmp_path: Path, name: str = "wrong_encoding.csv") -> str:
    path = tmp_path / name
    path.write_bytes(LATIN1_CONTENT.encode("latin-1"))
    return str(path)


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _graph(executor: Any, llm_port: Any) -> Any:
    return build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=executor,
        llm_port=llm_port,
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )


def _expected_encoding(file_path: str) -> str:
    detected = chardet.detect(Path(file_path).read_bytes())["encoding"]
    assert detected is not None
    return detected


# --- Evidence reaches the sample -----------------------------------------


def test_wrong_encoding_sample_includes_chardet_evidence(tmp_path: Path) -> None:
    file_path = _write_latin1(tmp_path)
    # Captured before `graph.invoke` — a successful `apply` would
    # physically rewrite the file into canonical UTF-8, which would make
    # a post-invoke chardet read report the *repaired* file's encoding
    # instead of the original one this test is about.
    expected_encoding = _expected_encoding(file_path)
    spy = _SpyProposalPort(VALID_PROPOSAL)
    graph = _graph(PandasCsvRepairExecutor(), spy)

    graph.invoke(build_initial_state(file_path, max_retries=0))

    assert len(spy.samples) == 1
    sample = spy.samples[0]
    assert _EVIDENCE_MARKER in sample
    assert expected_encoding in sample


def test_wrong_delimiter_sample_is_unchanged(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    spy = _SpyProposalPort(VALID_PROPOSAL)
    graph = _graph(PandasCsvRepairExecutor(), spy)

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_class"] == FailureClass.WRONG_DELIMITER
    assert len(spy.samples) == 1
    assert _EVIDENCE_MARKER not in spy.samples[0]
    assert spy.samples[0] == WRONG_DELIMITER_CSV


# --- 2. LLM's wrong encoding guess is deterministically corrected --------


def test_wrong_encoding_proposal_is_corrected_to_the_detected_encoding(tmp_path: Path) -> None:
    """Reproduces the real live-Groq failure mode: the LLM ignores the
    chardet evidence and keeps proposing `encoding="utf-8"`. The raw
    proposal entering `validate` must nonetheless carry the detected
    encoding, not the LLM's wrong guess."""
    file_path = _write_latin1(tmp_path)
    # Captured before `graph.invoke` — this repair succeeds and
    # physically rewrites the file into canonical UTF-8, so a post-invoke
    # chardet read would report the *repaired* file's encoding instead of
    # the original one this assertion is about.
    expected_encoding = _expected_encoding(file_path)
    wrong_llm_guess = {"delimiter": ",", "encoding": "utf-8", "header_row": 0, "engine": "python"}
    spy = _SpyProposalPort(wrong_llm_guess)
    graph = _graph(PandasCsvRepairExecutor(), spy)

    result = graph.invoke(build_initial_state(file_path, max_retries=0))

    assert result["proposed_params"] is not None
    assert result["proposed_params"]["encoding"] == expected_encoding
    assert result["proposed_params"]["encoding"] != "utf-8"
    # every other field is untouched, still exactly what the LLM proposed
    assert result["proposed_params"]["delimiter"] == ","
    assert result["proposed_params"]["header_row"] == 0
    assert result["proposed_params"]["engine"] == "python"


def test_correction_still_passes_through_existing_validation(tmp_path: Path) -> None:
    file_path = _write_latin1(tmp_path)
    # Captured before `graph.invoke` — see the comment in
    # `test_wrong_encoding_proposal_is_corrected_to_the_detected_encoding`.
    expected_encoding = _expected_encoding(file_path)
    wrong_llm_guess = {"delimiter": ",", "encoding": "utf-8", "header_row": 0, "engine": "python"}
    graph = _graph(PandasCsvRepairExecutor(), _SpyProposalPort(wrong_llm_guess))

    result = graph.invoke(build_initial_state(file_path))

    assert result["validated_params"] is not None
    assert result["validated_params"].encoding == expected_encoding
    assert result["validation_errors"] == []


# --- 3. WRONG_DELIMITER's proposal is never touched -----------------------


def test_wrong_delimiter_encoding_is_not_overridden(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    llm_guess = {"delimiter": ";", "encoding": "iso-8859-1", "header_row": 0, "engine": "python"}
    graph = _graph(PandasCsvRepairExecutor(), _SpyProposalPort(llm_guess))

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_class"] == FailureClass.WRONG_DELIMITER
    assert result["proposed_params"] is not None
    assert result["proposed_params"]["encoding"] == "iso-8859-1"


# --- 4. Full fake-port Latin-1 workflow: validate -> apply -> reverify ---


def test_wrong_encoding_is_genuinely_repaired_end_to_end(tmp_path: Path) -> None:
    file_path = _write_latin1(tmp_path)
    # Even a fake port that (like real Groq) always proposes utf-8 must
    # now succeed, since the encoding field is corrected deterministically.
    graph = _graph(PandasCsvRepairExecutor(), _SpyProposalPort(dict(VALID_PROPOSAL) | {"delimiter": ","}))

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_class"] == FailureClass.WRONG_ENCODING
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].success is True
    assert result["verification_result"] is not None
    assert result["verification_result"].success is True


def test_wrong_encoding_is_genuinely_repaired_when_evidence_is_used(tmp_path: Path) -> None:
    """Retained from the prior ticket: an evidence-aware fake port (one
    that reads the prompt evidence itself, rather than relying on the new
    post-proposal correction) also still succeeds end to end."""
    file_path = _write_latin1(tmp_path)
    graph = _graph(PandasCsvRepairExecutor(), _EvidenceAwareProposalPort())

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_class"] == FailureClass.WRONG_ENCODING
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].success is True
    assert result["verification_result"] is not None
    assert result["verification_result"].success is True


# --- 5. Best-effort edge cases ---------------------------------------------


def test_detect_encoding_returns_none_on_missing_file() -> None:
    assert _detect_encoding("/nonexistent/path/does_not_exist.csv") is None


def test_detect_encoding_returns_none_when_chardet_finds_no_encoding(tmp_path: Path) -> None:
    file_path = _write_latin1(tmp_path)

    with patch(
        "self_healing_pipeline.application.orchestration.csv_repair_workflow.chardet.detect",
        return_value={"encoding": None, "confidence": 0.0, "language": None, "mime_type": None},
    ):
        assert _detect_encoding(file_path) is None


def test_original_proposal_is_kept_when_chardet_finds_no_encoding(tmp_path: Path) -> None:
    file_path = _write_latin1(tmp_path)
    spy = _SpyProposalPort(VALID_PROPOSAL)
    graph = _graph(PandasCsvRepairExecutor(), spy)

    with patch(
        "self_healing_pipeline.application.orchestration.csv_repair_workflow.chardet.detect",
        return_value={"encoding": None, "confidence": 0.0, "language": None, "mime_type": None},
    ):
        result = graph.invoke(build_initial_state(file_path, max_retries=0))

    assert len(spy.samples) == 1
    assert _EVIDENCE_MARKER not in spy.samples[0]
    assert result["proposed_params"] == VALID_PROPOSAL  # unmodified, incl. encoding="utf-8"


def test_proposal_still_runs_when_chardet_itself_raises(tmp_path: Path) -> None:
    file_path = _write_latin1(tmp_path)
    spy = _SpyProposalPort(VALID_PROPOSAL)
    graph = _graph(PandasCsvRepairExecutor(), spy)

    with patch(
        "self_healing_pipeline.application.orchestration.csv_repair_workflow.chardet.detect",
        side_effect=RuntimeError("simulated chardet failure"),
    ):
        result = graph.invoke(build_initial_state(file_path, max_retries=0))

    assert len(spy.samples) == 1
    assert _EVIDENCE_MARKER not in spy.samples[0]
    assert spy.samples[0] == result["sample"]
    assert result["failure_class"] == FailureClass.WRONG_ENCODING
    assert result["proposed_params"] == VALID_PROPOSAL  # unmodified fallback
