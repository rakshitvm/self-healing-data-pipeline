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


# --- `execute` physically mutates; `verify` is strictly read-only -------


def test_execute_physically_rewrites_the_file_to_canonical_form(tmp_path: Path) -> None:
    """The live-demo bug: `execute` (apply) must genuinely rewrite the
    file on disk into canonical CSV, not merely re-parse it in memory."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", "id;name;value\n1;alpha;10\n2;beta;20\n")
    params = CsvRepairParams(delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    on_disk = Path(file_path).read_text(encoding="utf-8")
    assert ";" not in on_disk
    assert on_disk == "id,name,value\n1,alpha,10\n2,beta,20\n"


def test_execute_does_not_write_when_the_prescription_fails_to_resolve_the_file(
    tmp_path: Path,
) -> None:
    """No fabricated success: if the corrected prescription still doesn't
    produce a sane multi-column result, `execute` must report failure
    and must not touch the original file at all."""
    file_path = _write(tmp_path, "still_broken.csv", "id name value\n1 alpha 10\n")
    original_bytes = Path(file_path).read_bytes()
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is False
    assert Path(file_path).read_bytes() == original_bytes


def test_a_failed_write_never_corrupts_the_original_file(tmp_path: Path) -> None:
    """Atomic write safety: if the write step itself fails after a
    successful read, the original file must remain exactly as it was —
    proven by making `DataFrame.to_csv` raise mid-repair."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", "id;name;value\n1;alpha;10\n2;beta;20\n")
    original_bytes = Path(file_path).read_bytes()
    params = CsvRepairParams(delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    with patch("pandas.DataFrame.to_csv", side_effect=OSError("simulated disk failure")):
        outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is False
    assert Path(file_path).read_bytes() == original_bytes


def test_verify_never_writes_to_the_file(tmp_path: Path) -> None:
    """`verify` (reverify) must be strictly read-only — unlike `execute`,
    it must never rewrite the file, even on a successful check."""
    file_path = _write(tmp_path, "clean.csv", "id,name,value\n1,alpha,10\n2,beta,20\n")
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)
    before_bytes = Path(file_path).read_bytes()
    before_mtime_ns = Path(file_path).stat().st_mtime_ns

    outcome = PandasCsvRepairExecutor().verify(file_path, params)

    assert outcome.success is True
    assert Path(file_path).read_bytes() == before_bytes
    assert Path(file_path).stat().st_mtime_ns == before_mtime_ns


def test_verify_reports_failure_for_a_still_malformed_file(tmp_path: Path) -> None:
    """`verify` independently re-reads with plain defaults (the file is
    expected to already be canonical after `execute`) — if it still isn't
    a sane multi-column CSV, verification must fail, read-only."""
    file_path = _write(tmp_path, "still_semicolons.csv", "id;name;value\n1;alpha;10\n")
    params = CsvRepairParams(delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)
    before_bytes = Path(file_path).read_bytes()

    outcome = PandasCsvRepairExecutor().verify(file_path, params)

    assert outcome.success is False
    assert Path(file_path).read_bytes() == before_bytes  # still read-only even on failure


def test_execute_then_verify_round_trip_on_real_malformed_data(tmp_path: Path) -> None:
    """End-to-end proof at the executor level: apply physically repairs
    the file, and a subsequent read-only verify independently confirms
    it, exactly mirroring the workflow's apply -> reverify sequence."""
    file_path = _write(
        tmp_path, "wrong_delimiter.csv", "id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n"
    )
    params = CsvRepairParams(delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)
    executor = PandasCsvRepairExecutor()

    apply_outcome = executor.execute(file_path, params)
    assert apply_outcome.success is True

    after_apply_bytes = Path(file_path).read_bytes()
    verify_outcome = executor.verify(file_path, params)

    assert verify_outcome.success is True
    assert Path(file_path).read_bytes() == after_apply_bytes  # verify did not touch the file
