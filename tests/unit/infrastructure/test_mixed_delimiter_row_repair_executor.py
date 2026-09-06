"""Focused unit tests for `MixedDelimiterCsvRepairExecutor`.

Every test writes a real temporary CSV file and drives the real
executor — no mocking of file I/O. Mirrors
`test_pandas_csv_repair_executor.py`'s conventions (source-untouched /
separate-output-file tests, atomic write) for the parts that overlap,
plus the row-level-specific behavior this executor adds: applying only
approved affected-row repairs, preserving every other row verbatim, and
strengthened per-row verification.
"""

from pathlib import Path

from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvEngine, CsvRepairParams
from self_healing_pipeline.domain.value_objects.mixed_delimiter_row_repair import (
    MixedDelimiterRowRepair,
)
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
)
from self_healing_pipeline.infrastructure.csv.mixed_delimiter_row_repair_executor import (
    MixedDelimiterCsvRepairExecutor,
)


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _params_from_detector(file_path: str) -> CsvRepairParams:
    """Build a real `CsvRepairParams` the way the workflow's `propose`
    node would — deterministic detector evidence, not hand-authored, so
    these tests exercise the real detector -> executor handoff."""
    established_delimiter, rows = LocalCsvFailureDetector().detect_mixed_delimiter_evidence(
        file_path
    )
    return CsvRepairParams(
        delimiter=established_delimiter,
        encoding="utf-8",
        header_row=0,
        engine=CsvEngine.PYTHON,
        mixed_delimiter_rows=rows,
    )


def test_only_the_flagged_row_is_repaired_other_rows_preserved(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "panel.csv",
        "id,name,age\n1,Alice,30\n2,Bob,25\n6;Fiona;33\n7,George,29\n",
    )
    params = _params_from_detector(file_path)

    outcome = MixedDelimiterCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert outcome.output_path is not None
    repaired = Path(outcome.output_path).read_text(encoding="utf-8")
    assert repaired == "id,name,age\n1,Alice,30\n2,Bob,25\n6,Fiona,33\n7,George,29\n"


def test_source_is_never_modified(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "panel.csv", "id,name,age\n1,Alice,30\n6;Fiona;33\n7,George,29\n"
    )
    original_bytes = Path(file_path).read_bytes()
    params = _params_from_detector(file_path)

    outcome = MixedDelimiterCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert Path(file_path).read_bytes() == original_bytes
    assert outcome.output_path is not None
    assert Path(outcome.output_path) != Path(file_path)
    assert Path(outcome.output_path).parent == Path(file_path).resolve().parent / "repaired"


def test_multiple_malformed_rows_are_all_repaired(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "multi.csv",
        "id,name,age\n1,Alice,30\n2;Bob;25\n3,Charlie,35\n4;Diana;28\n5,Ethan,40\n",
    )
    params = _params_from_detector(file_path)

    outcome = MixedDelimiterCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert outcome.output_path is not None
    repaired = Path(outcome.output_path).read_text(encoding="utf-8")
    assert repaired == (
        "id,name,age\n1,Alice,30\n2,Bob,25\n3,Charlie,35\n4,Diana,28\n5,Ethan,40\n"
    )


def test_different_delimiters_in_different_rows_are_both_repaired(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "mixed_delims.csv",
        "id,name,age\n1,Alice,30\n2;Bob;25\n3,Charlie,35\n4|Diana|28\n",
    )
    params = _params_from_detector(file_path)
    assert {r.observed_delimiter for r in params.mixed_delimiter_rows} == {";", "|"}

    outcome = MixedDelimiterCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert outcome.output_path is not None
    repaired = Path(outcome.output_path).read_text(encoding="utf-8")
    assert repaired == "id,name,age\n1,Alice,30\n2,Bob,25\n3,Charlie,35\n4,Diana,28\n"


def test_header_and_row_count_and_order_are_preserved(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "order.csv",
        "id,name,age\n1,Alice,30\n2,Bob,25\n6;Fiona;33\n7,George,29\n8,Hannah,31\n",
    )
    params = _params_from_detector(file_path)

    outcome = MixedDelimiterCsvRepairExecutor().execute(file_path, params)

    assert outcome.output_path is not None
    lines = Path(outcome.output_path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id,name,age"
    assert len(lines) == 6  # header + 5 data rows, none dropped
    assert [line.split(",")[0] for line in lines[1:]] == ["1", "2", "6", "7", "8"]


def test_no_flagged_rows_repairs_the_file_unchanged(tmp_path: Path) -> None:
    """Defensive: an empty `mixed_delimiter_rows` (shouldn't normally
    reach this executor, but must not corrupt data if it does) leaves
    every row exactly as it was."""
    file_path = _write(tmp_path, "clean.csv", "id,name,age\n1,Alice,30\n2,Bob,25\n")
    params = CsvRepairParams(
        delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON
    )

    outcome = MixedDelimiterCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert outcome.output_path is not None
    assert (
        Path(outcome.output_path).read_text(encoding="utf-8")
        == "id,name,age\n1,Alice,30\n2,Bob,25\n"
    )


# --- strengthened verification ----------------------------------------------


def test_verify_passes_for_a_fully_repaired_file(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "panel.csv", "id,name,age\n1,Alice,30\n6;Fiona;33\n7,George,29\n"
    )
    params = _params_from_detector(file_path)
    executor = MixedDelimiterCsvRepairExecutor()
    outcome = executor.execute(file_path, params)
    assert outcome.output_path is not None

    verify_outcome = executor.verify(outcome.output_path, params)

    assert verify_outcome.success is True


def test_verify_fails_if_a_malformed_row_still_remains(tmp_path: Path) -> None:
    """The strengthened check (Part 14): unlike `PandasCsvRepairExecutor`
    (which only checks `frame.shape[1] < 2`), a file that still has a
    row not matching the header's field count must fail verification —
    proven directly by verifying a file that was never actually
    repaired."""
    file_path = _write(
        tmp_path, "still_broken.csv", "id,name,age\n1,Alice,30\n6;Fiona;33\n"
    )
    params = CsvRepairParams(
        delimiter=",",
        encoding="utf-8",
        header_row=0,
        engine=CsvEngine.PYTHON,
        mixed_delimiter_rows=(
            MixedDelimiterRowRepair(
                row_number=3,
                expected_field_count=3,
                actual_field_count=1,
                observed_delimiter=";",
                original_text="6;Fiona;33",
                repaired_text="6,Fiona,33",
            ),
        ),
    )

    # Verify against the *unrepaired* source directly (simulating apply
    # somehow not having run) — the still-malformed row must be caught.
    verify_outcome = MixedDelimiterCsvRepairExecutor().verify(file_path, params)

    assert verify_outcome.success is False
    assert any("row 3" in e for e in verify_outcome.validation_errors)


def test_execute_refuses_when_an_unflagged_row_is_also_malformed(tmp_path: Path) -> None:
    """Safety: `apply` must never silently write a partially-repaired
    file. A row that's malformed but wasn't in the approved
    `mixed_delimiter_rows` list must cause a failure, not a
    best-effort partial write."""
    file_path = _write(
        tmp_path,
        "partial.csv",
        "id,name,age\n1,Alice,30\n6;Fiona;33\n9;Extra;40\n",
    )
    # Only row 3 (Fiona) was approved; row 4 (Extra) was not.
    params = CsvRepairParams(
        delimiter=",",
        encoding="utf-8",
        header_row=0,
        engine=CsvEngine.PYTHON,
        mixed_delimiter_rows=(
            MixedDelimiterRowRepair(
                row_number=3,
                expected_field_count=3,
                actual_field_count=1,
                observed_delimiter=";",
                original_text="6;Fiona;33",
                repaired_text="6,Fiona,33",
            ),
        ),
    )

    outcome = MixedDelimiterCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is False
    assert outcome.output_path is None
    assert not (Path(file_path).parent / "repaired").exists()


def test_execute_creates_the_output_directory_if_missing(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "panel.csv", "id,name,age\n1,Alice,30\n6;Fiona;33\n"
    )
    params = _params_from_detector(file_path)
    assert not (tmp_path / "repaired").exists()

    outcome = MixedDelimiterCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert (tmp_path / "repaired").is_dir()


# --- header_row-aware (garbled HEADER_DETECTION header line) ----------------


def _params_from_garbled_header(file_path: str) -> CsvRepairParams:
    """Mirrors `_params_from_detector`, but for the garbled-header-line
    evidence path instead of the genuine-MIXED_DELIMITER one."""
    detector = LocalCsvFailureDetector()
    header_row = detector.detect_header_offset_evidence(file_path)
    result = detector.detect_garbled_header_repair(file_path)
    assert result is not None
    delimiter, repair = result
    return CsvRepairParams(
        delimiter=delimiter,
        encoding="utf-8",
        header_row=header_row,
        engine=CsvEngine.PYTHON,
        mixed_delimiter_rows=(repair,),
    )


def test_preamble_lines_are_dropped_and_garbled_header_is_rewritten(tmp_path: Path) -> None:
    """The live-discovered case, at the executor level: 2 junk preamble
    lines must be dropped entirely from the output, and the garbled
    header line rewritten — not promoting the first data row into the
    column names."""
    file_path = _write(
        tmp_path,
        "garbled_header.csv",
        "VDVSDVSDVSD\nSNJSDHSDH\nid;name|age,state,country\n"
        "1,Alice,30,Georgia,USA\n2,Bob,25,Georgia,USA\n3,Charlie,35,Georgia,USA\n",
    )
    params = _params_from_garbled_header(file_path)
    assert params.header_row == 2

    outcome = MixedDelimiterCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert outcome.output_path is not None
    repaired = Path(outcome.output_path).read_text(encoding="utf-8")
    assert repaired == (
        "id,name,age,state,country\n"
        "1,Alice,30,Georgia,USA\n"
        "2,Bob,25,Georgia,USA\n"
        "3,Charlie,35,Georgia,USA\n"
    )

    verify_outcome = MixedDelimiterCsvRepairExecutor().verify(outcome.output_path, params)
    assert verify_outcome.success is True


def test_genuine_mixed_delimiter_case_is_unaffected_by_header_row_awareness(
    tmp_path: Path,
) -> None:
    """Regression guard: a real MIXED_DELIMITER prescription always has
    `header_row=0` — confirm the header-row-aware generalization is a
    strict no-op for it (identical output to the pre-existing
    `test_only_the_flagged_row_is_repaired_other_rows_preserved`)."""
    file_path = _write(
        tmp_path,
        "panel.csv",
        "id,name,age\n1,Alice,30\n2,Bob,25\n6;Fiona;33\n7,George,29\n",
    )
    params = _params_from_detector(file_path)
    assert params.header_row == 0

    outcome = MixedDelimiterCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is True
    assert outcome.output_path is not None
    repaired = Path(outcome.output_path).read_text(encoding="utf-8")
    assert repaired == "id,name,age\n1,Alice,30\n2,Bob,25\n6,Fiona,33\n7,George,29\n"


def test_header_row_out_of_range_fails_safely(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "short.csv", "id,name,age\n1,Alice,30\n")
    params = CsvRepairParams(
        delimiter=",",
        encoding="utf-8",
        header_row=5,
        engine=CsvEngine.PYTHON,
        mixed_delimiter_rows=(
            MixedDelimiterRowRepair(
                row_number=6,
                expected_field_count=3,
                actual_field_count=1,
                observed_delimiter=";",
                original_text="x",
                repaired_text="x,y,z",
            ),
        ),
    )

    outcome = MixedDelimiterCsvRepairExecutor().execute(file_path, params)

    assert outcome.success is False
    assert outcome.output_path is None
