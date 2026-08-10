"""Focused unit tests for `PandasCsvRepairExecutor`.

Most tests write real temporary CSV files (via pytest's `tmp_path`) and
drive the full existing Tier 1 pipeline — `CsvRepairAgent` (Ticket 005)
using this executor — so assertions verify actual repaired file
behavior, not just the executor's return shape. One targeted test spies
on the `pandas.read_csv` call to confirm the prescribed `engine` is
genuinely passed through; every other test exercises real file I/O.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from self_healing_pipeline.domain.exceptions.csv_errors import (
    EngineSelectionError,
    HeaderDetectionError,
    SingleColumnMalformationError,
    WrongDelimiterError,
    WrongEncodingError,
)
from self_healing_pipeline.domain.value_objects.csv_repair_params import (
    CsvEngine,
    CsvRepairParams,
)
from self_healing_pipeline.infrastructure.agents.csv_repair_agent import CsvRepairAgent
from self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor import (
    PandasCsvRepairExecutor,
)


def _write(tmp_path: Path, name: str, content: str, encoding: str = "utf-8") -> str:
    path = tmp_path / name
    path.write_bytes(content.encode(encoding))
    return str(path)


def test_wrong_delimiter_is_repaired(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "wrong_delimiter.csv", "id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n"
    )
    agent = CsvRepairAgent(PandasCsvRepairExecutor())

    result = agent.handle(WrongDelimiterError("bad delimiter", file_path=file_path))

    assert result.success is True
    assert result.applied is True
    assert result.prescription is not None
    assert result.prescription.delimiter == ";"


def test_wrong_encoding_is_repaired(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "wrong_encoding.csv",
        "id,name,value\n1,café,10\n2,naïve,20\n3,façade,30\n",
        encoding="latin-1",
    )
    agent = CsvRepairAgent(PandasCsvRepairExecutor())

    result = agent.handle(WrongEncodingError("bad codec", file_path=file_path))

    assert result.success is True
    assert result.prescription is not None
    assert result.prescription.encoding == "latin-1"


def test_header_detection_is_repaired(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "header_detection.csv",
        "Sales Report - Q3 2026\nid,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )
    agent = CsvRepairAgent(PandasCsvRepairExecutor())

    result = agent.handle(HeaderDetectionError("header not found", file_path=file_path))

    assert result.success is True
    assert result.prescription is not None
    assert result.prescription.header_row == 1


def test_single_column_malformation_is_repaired(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "single_column.csv", "id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n"
    )
    agent = CsvRepairAgent(PandasCsvRepairExecutor())

    result = agent.handle(
        SingleColumnMalformationError("collapsed to one column", file_path=file_path)
    )

    assert result.success is True
    assert result.prescription is not None
    assert result.prescription.delimiter == ";"


def test_engine_selection_prescription_is_applied_on_a_clean_file(tmp_path: Path) -> None:
    """The Tier 1 default ENGINE_SELECTION prescription switches to pyarrow;
    confirm the executor genuinely applies it end to end on real data."""
    file_path = _write(tmp_path, "clean.csv", "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n")
    agent = CsvRepairAgent(PandasCsvRepairExecutor())

    result = agent.handle(EngineSelectionError("no engine could parse the file", file_path=file_path))

    assert result.success is True
    assert result.prescription is not None
    assert result.prescription.engine == CsvEngine.PYARROW


def test_executor_passes_prescribed_engine_through_to_pandas(tmp_path: Path) -> None:
    """White-box check that `params.engine` genuinely reaches `pandas.read_csv`
    (rather than being silently ignored)."""
    file_path = _write(tmp_path, "clean.csv", "id,name,value\n1,alpha,10\n")
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.PYARROW)

    with patch(
        "self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor.pd.read_csv"
    ) as mock_read_csv:
        mock_read_csv.return_value = MagicMock(shape=(1, 3), columns=["id", "name", "value"])
        PandasCsvRepairExecutor().execute(file_path, params)

    assert mock_read_csv.call_args.kwargs["engine"] == "pyarrow"


def test_genuinely_ragged_file_is_not_fixed_by_engine_switch_alone(tmp_path: Path) -> None:
    """Honest negative case: a genuinely ragged row (inconsistent field
    count) cannot be fixed by any single-character-delimiter engine
    switch alone. The executor must report an unsuccessful outcome."""
    file_path = _write(tmp_path, "ragged.csv", "id,name,value\n1,alpha,10\n2,be,ta,20\n3,gamma,30\n")
    agent = CsvRepairAgent(PandasCsvRepairExecutor())

    result = agent.handle(EngineSelectionError("no engine could parse the file", file_path=file_path))

    assert result.success is False
    assert result.applied is False
    assert result.validation_errors


def test_invalid_repair_returns_unsuccessful_outcome_for_missing_file() -> None:
    executor = PandasCsvRepairExecutor()
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.C)

    outcome = executor.execute("/nonexistent/path/does_not_exist.csv", params)

    assert outcome.success is False
    assert outcome.confidence is None
    assert outcome.validation_errors


def test_repair_result_reports_readable_columns_and_data(tmp_path: Path) -> None:
    """Successful repair must yield a result whose message reflects real,
    readable row/column counts, not a canned or arbitrary string."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", "id;name;value\n1;alpha;10\n2;beta;20\n")
    agent = CsvRepairAgent(PandasCsvRepairExecutor())

    result = agent.handle(WrongDelimiterError("bad delimiter", file_path=file_path))

    assert result.success is True
    assert result.message is not None
    assert "2 rows" in result.message
    assert "3 columns" in result.message
