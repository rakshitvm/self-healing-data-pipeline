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
    """Fake `CsvRepairExecutor`: returns a canned outcome, records calls."""

    def __init__(self, outcome: CsvExecutionOutcome) -> None:
        self._outcome = outcome
        self.calls: list[tuple[str, CsvRepairParams]] = []

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        self.calls.append((file_path, params))
        return self._outcome


class _ExplodingCsvRepairExecutor:
    """Fake `CsvRepairExecutor` that fails the test if ever called."""

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        raise AssertionError("executor.execute() should not have been called")


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
    path.write_bytes("id,name,value\n1,café,10\n2,naïve,20\n3,façade,30\n".encode("latin-1"))
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(
            {"delimiter": ",", "encoding": "latin-1", "header_row": 0, "engine": "python"}
        ),
    )

    result = graph.invoke(build_initial_state(str(path)))

    assert result["failure_class"] == FailureClass.WRONG_ENCODING
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].prescription is not None
    # `propose` deterministically corrects `encoding` from the file's real
    # bytes via chardet, overriding whatever the (fake) LLM proposed — so
    # the applied encoding is chardet's answer, not the fake port's
    # "latin-1", even though both decode this fixture identically.
    expected_encoding = chardet.detect(path.read_bytes())["encoding"]
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
    executor = _FakeCsvRepairExecutor(CsvExecutionOutcome(success=True, confidence=1.0))
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(), executor=executor, llm_port=_FakeProposalPort(VALID_PROPOSAL)
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
    (fewer fields than the header, not more) is genuinely repairable —
    pandas pads the gap with NaN — but only with `engine="python"`/`"c"`,
    not `"pyarrow"` (verified separately: pyarrow raises `ParserError` on
    this exact fixture). Matches the proposal a real Groq run produced.
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
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["failure_class"] == FailureClass.ENGINE_SELECTION
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert result["repair_result"] is not None
    assert result["repair_result"].success is True
    assert result["verification_result"] is not None
    assert result["verification_result"].success is True


def test_valid_csv_repair_params_reaches_apply(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    executor = _FakeCsvRepairExecutor(CsvExecutionOutcome(success=True, confidence=1.0))
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(), executor=executor, llm_port=_FakeProposalPort(VALID_PROPOSAL)
    )

    graph.invoke(build_initial_state(file_path))

    assert len(executor.calls) == 2  # apply + reverify
    _, params_passed_to_apply = executor.calls[0]
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
    graph = build_csv_repair_workflow(detector=LocalCsvFailureDetector(), executor=executor, llm_port=llm_port)

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
        detector=LocalCsvFailureDetector(), executor=executor, llm_port=_FakeProposalPort(VALID_PROPOSAL)
    )

    result = graph.invoke(build_initial_state(file_path, max_retries=0))

    assert result["retry_count"] == 0
    assert result["status"] == RepairEpisodeStatus.FAILED
    assert result["error_message"] == "verification did not succeed"


def test_tool_node_is_present_in_compiled_graph() -> None:
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )

    node = graph.get_graph().nodes[SAMPLE_TOOL_NODE_NAME]
    assert isinstance(node.data, ToolNode)
