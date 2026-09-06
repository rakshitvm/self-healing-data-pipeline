"""Focused unit tests for two deterministic `propose`-node overrides:

1. HEADER_DETECTION's `header_row` — the number of leading preamble
   lines to skip, computed deterministically over the whole file
   (`detect_header_offset_evidence`) and force-corrected onto the LLM's
   raw proposal, exactly like WRONG_ENCODING's chardet correction.
2. NO_HEADER's `header_row=None` — mechanical, never left to the LLM to
   guess.

Kept in a separate file (mirrors
`test_csv_repair_workflow_encoding_evidence.py`'s own convention) so each
fix's diff stays unambiguous. Uses the real `LocalCsvFailureDetector`/
`PandasCsvRepairExecutor` against real temporary CSV files, combined with
fake `CsvRepairProposalPort`/`CsvHumanApprovalPort` doubles — no real
network access, no API key.
"""

from pathlib import Path
from typing import Any

from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
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


class _SpyProposalPort:
    """Fake `CsvRepairProposalPort`: records every `sample` it is called
    with and always returns the same canned (possibly wrong) payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.samples: list[str] = []

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        self.samples.append(sample)
        return dict(self._payload)


class _AlwaysApproveCsvHumanApprovalPort:
    def request_approval(self, request: Any) -> bool:
        return True


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _graph(llm_port: Any) -> Any:
    return build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=llm_port,
        approval_port=_AlwaysApproveCsvHumanApprovalPort(),
    )


def test_header_detection_offset_is_corrected_for_a_multi_line_preamble(
    tmp_path: Path,
) -> None:
    """The LLM's wrong guess (`header_row=1`, correct only for a
    single-line preamble) must be force-corrected to the real offset
    (`3`, for this 3-line preamble) — a 5-line sample cap wouldn't even
    show the LLM the real header line for a longer preamble, so this
    must never be left to the LLM to guess."""
    file_path = _write(
        tmp_path,
        "multi_line_preamble.csv",
        "Weekly Sales Report\n"
        "Reporting Period: 2026-01-01 to 2026-01-14\n"
        "Prepared by: Data Team\n"
        "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )
    wrong_llm_guess = {"delimiter": ",", "encoding": "utf-8", "header_row": 1, "engine": "python"}
    spy = _SpyProposalPort(wrong_llm_guess)

    result = _graph(spy).invoke(build_initial_state(file_path, max_retries=0))

    assert result["failure_class"] == FailureClass.HEADER_DETECTION
    assert result["proposed_params"]["header_row"] == 3
    assert result["proposed_params"]["header_row"] != 1
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"].output_path is not None
    repaired = Path(result["repair_result"].output_path).read_text(encoding="utf-8")
    assert repaired.splitlines()[0] == "id,name,value"
    assert len(repaired.splitlines()) == 4  # header + 3 data rows, nothing lost


def test_header_detection_single_line_preamble_still_works(tmp_path: Path) -> None:
    """Regression guard: the override must reproduce the pre-existing
    single-line-preamble behavior exactly (offset=1)."""
    file_path = _write(
        tmp_path,
        "single_line_preamble.csv",
        "Sales Report - Q3 2026\nid,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )
    wrong_llm_guess = {"delimiter": ",", "encoding": "utf-8", "header_row": 0, "engine": "python"}
    spy = _SpyProposalPort(wrong_llm_guess)

    result = _graph(spy).invoke(build_initial_state(file_path, max_retries=0))

    assert result["proposed_params"]["header_row"] == 1
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED


def test_no_header_row_is_forced_to_none(tmp_path: Path) -> None:
    """The LLM's wrong guess (`header_row=0`, which would silently
    swallow row 1 as a fake header) must be force-corrected to `None` —
    mechanical, never left to the LLM."""
    file_path = _write(tmp_path, "no_header.csv", "1,Alice,30\n2,Bob,25\n3,Carol,40\n4,Dave,22\n")
    wrong_llm_guess = {"delimiter": ",", "encoding": "utf-8", "header_row": 0, "engine": "python"}
    spy = _SpyProposalPort(wrong_llm_guess)

    result = _graph(spy).invoke(build_initial_state(file_path, max_retries=0))

    assert result["failure_class"] == FailureClass.NO_HEADER
    assert result["proposed_params"]["header_row"] is None
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    repaired = Path(result["repair_result"].output_path).read_text(encoding="utf-8")
    # 4 data rows survive, plus the synthesized "0,1,2" positional header
    # pandas writes for an int-columned frame — nothing was swallowed.
    assert len(repaired.splitlines()) == 5
