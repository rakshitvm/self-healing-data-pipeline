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


# --- `execute` never modifies the source; it writes a separate output ---
# --- file. `verify` is strictly read-only. -------------------------------


def test_execute_creates_separate_repaired_output_and_leaves_source_untouched(
    tmp_path: Path,
) -> None:
    """The live-demo bug: `execute` (apply) used to physically overwrite
    the source file in place. It must instead create a *separate*
    repaired output file — in a `repaired/` subdirectory alongside the
    source, preserving the original filename — while the source stays
    byte-for-byte identical."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", "id;name;value\n1;alpha;10\n2;beta;20\n")
    original_bytes = Path(file_path).read_bytes()
    params = CsvRepairParams(delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert Path(file_path).read_bytes() == original_bytes  # source untouched

    assert outcome.output_path is not None
    output_path = Path(outcome.output_path)
    assert output_path != Path(file_path)
    assert output_path.name == "wrong_delimiter.csv"  # filename preserved
    assert output_path.parent == Path(file_path).resolve().parent / "repaired"

    on_disk_output = output_path.read_text(encoding="utf-8")
    assert ";" not in on_disk_output
    assert on_disk_output == "id,name,value\n1,alpha,10\n2,beta,20\n"


def test_execute_creates_the_output_directory_if_missing(tmp_path: Path) -> None:
    """The `repaired/` output directory does not exist yet before the
    first repair — `execute` must create it rather than fail."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", "id;name;value\n1;alpha;10\n")
    params = CsvRepairParams(delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)
    assert not (tmp_path / "repaired").exists()

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert (tmp_path / "repaired").is_dir()
    assert outcome.output_path is not None
    assert Path(outcome.output_path).is_file()


def test_execute_does_not_write_when_the_prescription_fails_to_resolve_the_file(
    tmp_path: Path,
) -> None:
    """No fabricated success: if the corrected prescription still doesn't
    produce a sane multi-column result, `execute` must report failure
    and must not touch the source or create any output at all."""
    file_path = _write(tmp_path, "still_broken.csv", "id name value\n1 alpha 10\n")
    original_bytes = Path(file_path).read_bytes()
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is False
    assert outcome.output_path is None
    assert Path(file_path).read_bytes() == original_bytes
    assert not (tmp_path / "repaired").exists()


def test_a_failed_write_never_corrupts_the_source_or_leaves_a_partial_output(
    tmp_path: Path,
) -> None:
    """Atomic write safety: if the write step itself fails after a
    successful read, neither the source nor a partial output file must
    be left behind — proven by making `DataFrame.to_csv` raise
    mid-repair."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", "id;name;value\n1;alpha;10\n2;beta;20\n")
    original_bytes = Path(file_path).read_bytes()
    params = CsvRepairParams(delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    with patch("pandas.DataFrame.to_csv", side_effect=OSError("simulated disk failure")):
        outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is False
    assert Path(file_path).read_bytes() == original_bytes  # source untouched
    output_dir = tmp_path / "repaired"
    if output_dir.exists():
        assert list(output_dir.iterdir()) == []  # no partial/temp file left behind


def test_refuse_if_output_collides_with_source_direct_unit_test() -> None:
    """Direct test of the safety-net check itself (requirement: never
    overwrite the source even if output-path resolution accidentally
    points back at it) — exercised directly since the real resolution
    logic can't naturally produce a collision to test through the public
    API alone."""
    from self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor import (
        _refuse_if_output_collides_with_source,
    )

    outcome = _refuse_if_output_collides_with_source("/tmp/x.csv", Path("/tmp/x.csv"))

    assert outcome is not None
    assert outcome.success is False
    assert "collides_with_source" in outcome.validation_errors[0]


def test_execute_fails_safely_when_output_path_would_collide_with_source(tmp_path: Path) -> None:
    """Integration-level proof through the real `execute()` entry point:
    if output-path resolution is forced to collide with the source, the
    source must be left completely untouched and no write attempted."""
    from self_healing_pipeline.infrastructure.csv import pandas_csv_repair_executor as module

    file_path = _write(tmp_path, "wrong_delimiter.csv", "id;name;value\n1;alpha;10\n2;beta;20\n")
    original_bytes = Path(file_path).read_bytes()
    params = CsvRepairParams(delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    with patch.object(module, "_resolve_output_path", return_value=Path(file_path)):
        outcome = module.PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is False
    assert outcome.output_path is None
    assert Path(file_path).read_bytes() == original_bytes


def test_verify_never_writes_to_the_file(tmp_path: Path) -> None:
    """`verify` (reverify) must be strictly read-only — unlike `execute`,
    it must never write to the file it's given (in the real workflow,
    the repaired output file), even on a successful check."""
    file_path = _write(tmp_path, "clean.csv", "id,name,value\n1,alpha,10\n2,beta,20\n")
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)
    before_bytes = Path(file_path).read_bytes()
    before_mtime_ns = Path(file_path).stat().st_mtime_ns

    outcome = PandasCsvRepairExecutor().verify(file_path, params)

    assert outcome.success is True
    assert Path(file_path).read_bytes() == before_bytes
    assert Path(file_path).stat().st_mtime_ns == before_mtime_ns


def test_verify_reports_failure_for_a_still_malformed_file(tmp_path: Path) -> None:
    """`verify` independently re-reads with plain defaults (a repaired
    output file is expected to already be canonical) — if it still isn't
    a sane multi-column CSV, verification must fail, read-only."""
    file_path = _write(tmp_path, "still_semicolons.csv", "id;name;value\n1;alpha;10\n")
    params = CsvRepairParams(delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)
    before_bytes = Path(file_path).read_bytes()

    outcome = PandasCsvRepairExecutor().verify(file_path, params)

    assert outcome.success is False
    assert Path(file_path).read_bytes() == before_bytes  # still read-only even on failure


def test_execute_then_verify_round_trip_on_real_malformed_data(tmp_path: Path) -> None:
    """End-to-end proof at the executor level, mirroring the workflow's
    apply -> reverify sequence: `execute` creates the repaired output
    (source untouched), and `verify`, called with that *output* path
    (not the source), independently confirms it read-only."""
    file_path = _write(
        tmp_path, "wrong_delimiter.csv", "id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n"
    )
    original_bytes = Path(file_path).read_bytes()
    params = CsvRepairParams(delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)
    executor = PandasCsvRepairExecutor()

    apply_outcome = executor.execute(file_path, params)
    assert apply_outcome.success is True
    assert apply_outcome.output_path is not None
    assert Path(file_path).read_bytes() == original_bytes  # still untouched after apply

    after_apply_output_bytes = Path(apply_outcome.output_path).read_bytes()
    verify_outcome = executor.verify(apply_outcome.output_path, params)

    assert verify_outcome.success is True
    assert Path(file_path).read_bytes() == original_bytes  # still untouched after reverify
    assert Path(apply_outcome.output_path).read_bytes() == after_apply_output_bytes  # verify didn't write


# --- row-level field-count verification (pandas silently NaN-pads a ------
# --- short row rather than raising; independently cross-checked here) ---


def test_execute_rejects_a_row_with_genuinely_fewer_fields_than_expected(
    tmp_path: Path,
) -> None:
    """The real bug found manually via test_samples/sampletest.csv: a row
    collapsed by a wrong delimiter (fewer raw fields than the header)
    must not be silently NaN-padded and reported as a success."""
    file_path = _write(
        tmp_path,
        "collapsed_row.csv",
        "id,name,age,country,continent\n"
        "1,Alice,30,india,asia\n"
        "6;Fiona;33,india,asia\n"
        "7,George,29,india,asia\n",
    )
    original_bytes = Path(file_path).read_bytes()
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.C)

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is False
    assert outcome.output_path is None
    assert any("row 3" in e for e in outcome.validation_errors)
    assert Path(file_path).read_bytes() == original_bytes  # source untouched
    assert not (tmp_path / "repaired").exists()  # nothing written


def test_verify_rejects_a_pre_existing_output_with_a_short_row(tmp_path: Path) -> None:
    """Direct proof that `verify` independently re-checks the *output*
    file, not just trusting whatever `execute` produced — a hand-written
    "already repaired" file with a short row must still fail verify."""
    output_path = _write(
        tmp_path, "bad_output.csv", "id,name,age\n1,Alice,30\n2,Bob\n"
    )
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.C)

    outcome = PandasCsvRepairExecutor().verify(output_path, params)

    assert outcome.success is False
    assert any("row 3" in e for e in outcome.validation_errors)


def test_a_legitimately_empty_trailing_value_is_not_a_false_positive(tmp_path: Path) -> None:
    """A field that is genuinely present but empty (a real trailing comma)
    must not be confused with a field that's missing entirely — the
    fixed check counts raw tokens via csv.reader, so an empty string is
    still counted as one field, not zero."""
    file_path = _write(
        tmp_path, "empty_trailing_field.csv", "id,name,age\n1,Alice,30\n2,Bob,\n"
    )
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert outcome.output_path is not None


def test_header_row_is_honored_by_the_row_level_check(tmp_path: Path) -> None:
    """HEADER_DETECTION's own shape: a junk title line before the real
    header (header_row=1). A short data row *after* the real header must
    still be caught; the junk line itself must never be mistaken for a
    data row."""
    file_path = _write(
        tmp_path,
        "header_detection_short_row.csv",
        "Sales Report - Q3 2026\nid,name,value\n1,alpha,10\n2,beta\n3,gamma,30\n",
    )
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=1, engine=CsvEngine.PYTHON)

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is False
    # row 4 = "2,beta" (junk title=1, real header=2, "1,alpha,10"=3, "2,beta"=4)
    assert any("row 4" in e for e in outcome.validation_errors)


def test_header_row_none_treats_every_line_as_data(tmp_path: Path) -> None:
    """`header_row=None` (pandas `header=None`, no header row at all) —
    every line is a data row, including the first."""
    file_path = _write(tmp_path, "headerless_short_row.csv", "1,alpha,10\n2,beta\n")
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=None, engine=CsvEngine.PYTHON)

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is False
    assert any("row 2" in e for e in outcome.validation_errors)  # "2,beta" is line 2


def test_surplus_row_still_fails_via_the_pre_existing_pandas_parser_error(
    tmp_path: Path,
) -> None:
    """Regression proof, not new behavior: a row with MORE fields than
    expected is already caught by pandas' own ParserError (confirmed
    directly for all three engines) — the new row-level check is a
    deficit-only addition and must never double-report or change this
    existing, already-correct failure path."""
    file_path = _write(
        tmp_path, "surplus_row.csv", "id,name,value\n1,alpha,10\n2,be,ta,20\n3,gamma,30\n"
    )
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.C)

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is False
    # the ParserError message, not our own "expected N fields" message —
    # proves pandas' own exception path fired first, ours was never reached
    assert not any("expected" in e.lower() and "fields, found" in e for e in outcome.validation_errors)


def test_sampletest_csv_fixture_now_correctly_fails_instead_of_false_success() -> None:
    """The actual test_samples/sampletest.csv file, run through the real
    executor directly (bypassing MIXED_DELIMITER-aware routing, proving
    this executor's own check in isolation): this is the user-facing bug
    that prompted this whole fix — manually running this file previously
    reported `success: True` while silently writing corrupted rows. It
    must now correctly fail rather than silently pad short rows with
    NaN. This is a shared, manually-edited fixture (its exact content
    has drifted more than once during development), so the exact row
    count isn't asserted — only that at least one genuine mismatch is
    always caught, never silently accepted."""
    fixture_path = Path(__file__).resolve().parents[3] / "test_samples" / "sampletest.csv"
    assert fixture_path.is_file(), f"expected fixture at {fixture_path}"
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.C)

    outcome = PandasCsvRepairExecutor().execute(str(fixture_path), params)

    assert outcome.success is False
    assert outcome.output_path is None
    assert len(outcome.validation_errors) >= 1


def test_execute_strips_invisible_characters_from_column_names(tmp_path: Path) -> None:
    """A UTF-8 BOM plus a zero-width space glued onto the first column
    name must not survive into the repaired output — unconditional, not
    gated on any particular failure class."""
    file_path = _write(
        tmp_path,
        "bom_and_zwsp.csv",
        "﻿​store_id,store_name\nS001,Tesco\nS002,Sainsbury's\n",
    )
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert outcome.output_path is not None
    header_line = Path(outcome.output_path).read_text(encoding="utf-8").splitlines()[0]
    assert header_line == "store_id,store_name"
    assert "﻿" not in header_line
    assert "​" not in header_line


def test_no_header_repair_preserves_integer_column_names(tmp_path: Path) -> None:
    """Invisible-character stripping must never coerce pandas' own
    default integer column names (a `header_row=None` / NO_HEADER read)
    into strings — only string column names are ever touched."""
    file_path = _write(tmp_path, "headerless.csv", "1,Alice,30\n2,Bob,25\n")
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=None, engine=CsvEngine.PYTHON)

    outcome = PandasCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert outcome.output_path is not None
    header_line = Path(outcome.output_path).read_text(encoding="utf-8").splitlines()[0]
    assert header_line == "0,1,2"


def test_verify_rejects_output_with_invisible_characters_in_columns(tmp_path: Path) -> None:
    """Direct proof `verify` independently re-confirms no invisible
    character remains in the *output* file's own column names, rather
    than trusting that `execute`'s in-memory strip actually took. Uses a
    zero-width space rather than a BOM: pandas' own `read_csv` already
    strips a real UTF-8 BOM automatically on the plain default read
    `verify` uses, so a BOM alone could never actually reach this check
    — a zero-width space is not auto-stripped and does."""
    output_path = _write(
        tmp_path, "contaminated_output.csv", "​id,name\n1,Alice\n2,Bob\n"
    )
    params = CsvRepairParams(delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    outcome = PandasCsvRepairExecutor().verify(output_path, params)

    assert outcome.success is False
    assert any("invisible" in e.lower() for e in outcome.validation_errors)


def test_execute_respects_the_prescribed_encoding_for_the_row_level_check(
    tmp_path: Path,
) -> None:
    """The independent raw-line field-count check must decode the source
    with the *prescribed* encoding, not always UTF-8 — otherwise a
    genuinely non-UTF-8 file (e.g. a real WRONG_ENCODING repair) gets
    misread as corrupted by this defense-in-depth check and a correct
    repair is falsely rejected. Regression test for the real bug found
    against the CSV test kit's UTF-16 fixture."""
    file_path = tmp_path / "latin1.csv"
    file_path.write_bytes(
        "id,name,value\n1,café,10\n2,naïve,20\n3,façade,30\n".encode("latin-1")
    )
    params = CsvRepairParams(
        delimiter=",", encoding="latin-1", header_row=0, engine=CsvEngine.PYTHON
    )

    outcome = PandasCsvRepairExecutor().execute(str(file_path), params)

    assert outcome.success is True
    assert outcome.output_path is not None
