"""Focused unit tests for the Tier 1 `CsvRepairAgent`.

Uses a fake `CsvRepairExecutor` throughout — no pandas, Spark, Databricks,
Azure, or any external service is required to exercise this agent.
"""

import pytest

from self_healing_pipeline.domain.exceptions.csv_errors import (
    EngineSelectionError,
    HeaderDetectionError,
    SingleColumnMalformationError,
    WrongDelimiterError,
    WrongEncodingError,
)
from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.interfaces.agents.repair_agent import RepairAgent
from self_healing_pipeline.domain.interfaces.services.csv_repair_executor import (
    CsvExecutionOutcome,
    CsvRepairExecutor,
)
from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.repair_result import RepairResult
from self_healing_pipeline.infrastructure.agents.csv_repair_agent import CsvRepairAgent


class _FakeCsvRepairExecutor:
    """Fake `CsvRepairExecutor`: returns a canned outcome, records calls."""

    def __init__(self, outcome: CsvExecutionOutcome) -> None:
        self._outcome = outcome
        self.calls: list[tuple[str, CsvRepairParams]] = []

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        self.calls.append((file_path, params))
        return self._outcome


class _ExplodingExecutor:
    """Fake `CsvRepairExecutor` that fails the test if ever called."""

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        raise AssertionError("executor.execute() should not have been called")


def test_successful_delimiter_repair() -> None:
    executor = _FakeCsvRepairExecutor(
        CsvExecutionOutcome(success=True, confidence=0.95, message="parsed cleanly")
    )
    agent = CsvRepairAgent(executor)
    error = WrongDelimiterError("bad delimiter", file_path="/mnt/landing/sales.csv")

    result = agent.handle(error)

    assert result.success is True
    assert result.applied is True
    assert result.confidence == 0.95
    assert result.prescription is not None
    assert result.prescription.delimiter == ";"
    assert executor.calls == [("/mnt/landing/sales.csv", result.prescription)]


def test_successful_encoding_repair() -> None:
    executor = _FakeCsvRepairExecutor(CsvExecutionOutcome(success=True, confidence=0.9))
    agent = CsvRepairAgent(executor)
    error = WrongEncodingError("bad codec", file_path="/mnt/landing/sales.csv")

    result = agent.handle(error)

    assert result.success is True
    assert result.prescription is not None
    assert result.prescription.encoding == "latin-1"


def test_header_detection_repair_prescription() -> None:
    executor = _FakeCsvRepairExecutor(CsvExecutionOutcome(success=True))
    agent = CsvRepairAgent(executor)
    error = HeaderDetectionError("header not found", file_path="/mnt/landing/sales.csv")

    result = agent.handle(error)

    assert result.prescription is not None
    assert result.prescription.header_row == 1
    assert result.prescription.delimiter == ","


def test_engine_selection_repair_prescription() -> None:
    from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvEngine

    executor = _FakeCsvRepairExecutor(CsvExecutionOutcome(success=True))
    agent = CsvRepairAgent(executor)
    error = EngineSelectionError("no engine could parse the file", file_path="/mnt/landing/x.csv")

    result = agent.handle(error)

    assert result.prescription is not None
    assert result.prescription.engine == CsvEngine.PYARROW


def test_single_column_malformation_repair_prescription() -> None:
    executor = _FakeCsvRepairExecutor(CsvExecutionOutcome(success=True))
    agent = CsvRepairAgent(executor)
    error = SingleColumnMalformationError(
        "silently collapsed to one column", file_path="/mnt/landing/x.csv"
    )

    result = agent.handle(error)

    assert result.prescription is not None
    assert result.prescription.delimiter == ";"


def test_repair_failure_returns_unsuccessful_result() -> None:
    executor = _FakeCsvRepairExecutor(
        CsvExecutionOutcome(
            success=False,
            validation_errors=["still 1 column after retry"],
            message="prescription did not resolve the malformation",
        )
    )
    agent = CsvRepairAgent(executor)
    error = SingleColumnMalformationError("still broken", file_path="/mnt/landing/x.csv")

    result = agent.handle(error)

    assert result.success is False
    assert result.applied is False
    assert result.validation_errors == ["still 1 column after retry"]
    assert result.message == "prescription did not resolve the malformation"
    # The attempted prescription is still reported for observability.
    assert result.prescription is not None


def test_repair_result_returned_correctly() -> None:
    executor = _FakeCsvRepairExecutor(
        CsvExecutionOutcome(success=True, confidence=0.77, message="ok")
    )
    agent = CsvRepairAgent(executor)
    error = WrongDelimiterError("bad delimiter", file_path="/mnt/landing/x.csv")

    result = agent.handle(error)

    assert isinstance(result, RepairResult)
    assert not isinstance(result, dict)
    assert result.confidence == 0.77
    assert result.message == "ok"


def test_agent_satisfies_repair_agent_protocol() -> None:
    agent = CsvRepairAgent(_FakeCsvRepairExecutor(CsvExecutionOutcome(success=True)))

    assert isinstance(agent, RepairAgent)
    assert isinstance(agent._executor, CsvRepairExecutor)  # sanity: fake satisfies the port too


def test_unsupported_failure_class_reraises_original_pipeline_error() -> None:
    agent = CsvRepairAgent(_ExplodingExecutor())
    error = PipelineError("not a CSV failure", failure_class=FailureClass.UNKNOWN)

    with pytest.raises(PipelineError) as exc_info:
        agent.handle(error)

    assert exc_info.value is error


def test_missing_file_path_returns_unsuccessful_result_without_calling_executor() -> None:
    agent = CsvRepairAgent(_ExplodingExecutor())
    error = WrongDelimiterError("bad delimiter")  # file_path defaults to None

    result = agent.handle(error)

    assert result.success is False
    assert result.applied is False
    assert result.validation_errors == ["missing_file_path"]
