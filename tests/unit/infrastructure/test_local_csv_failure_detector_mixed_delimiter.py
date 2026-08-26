"""Focused unit tests for `LocalCsvFailureDetector`'s MIXED_DELIMITER
detection: dynamic candidate discovery, false-positive protection,
quoted-value protection, whole-file scanning, and the regression
boundary against the pre-existing WRONG_DELIMITER/ENGINE_SELECTION
classifications.

Kept separate from `test_local_csv_failure_detector.py` (left untouched)
and `test_local_csv_failure_detector_multi_error.py` (left untouched),
mirroring how multi-error detection already got its own file. Every test
writes a real temporary CSV file and runs the actual detector — no
mocking of file I/O, `csv.Sniffer`, or `csv.reader`.
"""

from pathlib import Path

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
)


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


# --- TEST 1: clean file stays healthy -------------------------------------


def test_clean_comma_csv_is_healthy(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "clean.csv", "id,name,age\n1,Alice,30\n2,Bob,25\n")

    assert LocalCsvFailureDetector().detect(file_path) is None


# --- TEST 2-4: single anomalous row, start/middle/end ----------------------


def test_semicolon_row_in_the_middle_is_mixed_delimiter(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "middle.csv",
        "id,name,age\n1,Alice,30\n2,Bob,25\n6;Fiona;33\n7,George,29\n8,Hannah,31\n",
    )

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.MIXED_DELIMITER


def test_semicolon_row_as_first_data_row_is_mixed_delimiter(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "first.csv", "id,name,age\n6;Fiona;33\n2,Bob,25\n3,Charlie,35\n"
    )

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.MIXED_DELIMITER


def test_semicolon_row_as_last_data_row_is_mixed_delimiter(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "last.csv", "id,name,age\n1,Alice,30\n2,Bob,25\n6;Fiona;33\n"
    )

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.MIXED_DELIMITER


# --- TEST 5: multiple malformed rows, same delimiter ------------------------


def test_multiple_semicolon_rows_are_all_found(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "multi.csv",
        "id,name,age\n1,Alice,30\n2;Bob;25\n3,Charlie,35\n4;Diana;28\n",
    )

    evidence = LocalCsvFailureDetector().detect_mixed_delimiter_evidence(file_path)

    assert evidence[0] == ","
    affected_rows = {r.row_number for r in evidence[1]}
    assert affected_rows == {3, 5}
    assert all(r.observed_delimiter == ";" for r in evidence[1])


# --- TEST 6-9: pipe / tab / colon / one dynamically-discovered novel char --


def test_pipe_delimited_row_is_mixed_delimiter(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "pipe.csv", "id,name,age\n1,Alice,30\n2|Bob|25\n")

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.MIXED_DELIMITER
    _, rows = LocalCsvFailureDetector().detect_mixed_delimiter_evidence(file_path)
    assert rows[0].observed_delimiter == "|"
    assert rows[0].repaired_text == "2,Bob,25"


def test_tab_delimited_row_is_mixed_delimiter(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "tab.csv", "id,name,age\n1,Alice,30\n2\tBob\t25\n")

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.MIXED_DELIMITER
    _, rows = LocalCsvFailureDetector().detect_mixed_delimiter_evidence(file_path)
    assert rows[0].observed_delimiter == "\t"
    assert rows[0].repaired_text == "2,Bob,25"


def test_colon_delimited_row_is_mixed_delimiter(tmp_path: Path) -> None:
    """Colon is not in `_CANDIDATE_DELIMITERS = ",;\\t|"` at all — this
    proves dynamic discovery, not a longer hardcoded list."""
    file_path = _write(tmp_path, "colon.csv", "id,name,age\n1,Alice,30\n2:Bob:25\n")

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.MIXED_DELIMITER
    _, rows = LocalCsvFailureDetector().detect_mixed_delimiter_evidence(file_path)
    assert rows[0].observed_delimiter == ":"


def test_dynamically_discovered_novel_delimiter_not_in_original_candidate_list(
    tmp_path: Path,
) -> None:
    """`~` was never part of `_CANDIDATE_DELIMITERS` and never will be
    added to it (that constant stays untouched) — this proves the
    MIXED_DELIMITER path discovers candidates from the row itself, not
    from any fixed list at all."""
    file_path = _write(tmp_path, "novel.csv", "id,name,age\n1,Alice,30\n2~Bob~25\n")

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.MIXED_DELIMITER
    _, rows = LocalCsvFailureDetector().detect_mixed_delimiter_evidence(file_path)
    assert rows[0].observed_delimiter == "~"
    assert rows[0].repaired_text == "2,Bob,25"


# --- TEST 10: entire file consistently semicolon -> WRONG_DELIMITER --------


def test_entire_file_semicolon_delimited_stays_wrong_delimiter(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "all_semicolon.csv", "id;name;age\n1;Alice;30\n2;Bob;25\n3;Charlie;35\n"
    )

    result = LocalCsvFailureDetector().detect(file_path)

    assert result == FailureClass.WRONG_DELIMITER
    # and MIXED_DELIMITER must never also be reported for this file
    assert FailureClass.MIXED_DELIMITER not in LocalCsvFailureDetector().detect_all(file_path)


# --- TEST 11: entire file pipe-delimited -> WRONG_DELIMITER ----------------


def test_entire_file_pipe_delimited_stays_wrong_delimiter(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "all_pipe.csv", "id|name|age\n1|Alice|30\n2|Bob|25\n")

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.WRONG_DELIMITER


# --- TEST 12-14: quoted alternate-delimiter characters must NOT trigger ---


def test_quoted_semicolon_is_not_mixed_delimiter(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "quoted_semicolon.csv", 'id,name,age\n1,"Smith; John",30\n'
    )

    assert LocalCsvFailureDetector().detect(file_path) is None


def test_quoted_pipe_is_not_mixed_delimiter(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "quoted_pipe.csv", 'id,name,age\n1,"Smith|John",30\n')

    assert LocalCsvFailureDetector().detect(file_path) is None


def test_quoted_colon_is_not_mixed_delimiter(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "quoted_colon.csv", 'id,name,age\n1,"Smith: John",30\n')

    assert LocalCsvFailureDetector().detect(file_path) is None


# --- TEST 15: ordinary punctuation false-positive protection ---------------


def test_ordinary_punctuation_is_not_mixed_delimiter(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "punctuation.csv",
        "id,name,age\n1,John-Smith,30\n2,john@example.com,31\n3,abc_def,32\n",
    )

    assert LocalCsvFailureDetector().detect(file_path) is None


def test_period_in_data_is_not_mixed_delimiter(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "period.csv", "id,name,age\n1,John Smith Jr.,30\n")

    assert LocalCsvFailureDetector().detect(file_path) is None


# --- TEST 16-17: anomaly far beyond line 5 / thousands of rows in ---------


def test_anomaly_far_beyond_first_five_lines_is_still_found(tmp_path: Path) -> None:
    rows = "".join(f"{i},Person{i},{20 + i}\n" for i in range(1, 20))
    content = "id,name,age\n" + rows + "999;Anomaly;99\n" + "20,Person20,40\n"
    file_path = _write(tmp_path, "far.csv", content)

    result = LocalCsvFailureDetector().detect(file_path)
    _, rows_evidence = LocalCsvFailureDetector().detect_mixed_delimiter_evidence(file_path)

    assert result == FailureClass.MIXED_DELIMITER
    assert any(r.original_text == "999;Anomaly;99" for r in rows_evidence)


def test_anomaly_after_thousands_of_valid_rows_is_found_deterministically(
    tmp_path: Path,
) -> None:
    """No LLM/sample-cap involvement at all — proven by using only the
    detector directly, with a file far larger than any LLM sample."""
    rows = "".join(f"{i},Person{i},{20 + (i % 50)}\n" for i in range(1, 3001))
    content = "id,name,age\n" + rows + "3001;Anomaly;99\n"
    file_path = _write(tmp_path, "thousands.csv", content)

    result = LocalCsvFailureDetector().detect(file_path)
    _, rows_evidence = LocalCsvFailureDetector().detect_mixed_delimiter_evidence(file_path)

    assert result == FailureClass.MIXED_DELIMITER
    assert len(rows_evidence) == 1
    assert rows_evidence[0].row_number == 3002  # header + 3000 data rows + 1


# --- TEST 18: empty lines / trailing newline --------------------------------


def test_blank_lines_do_not_cause_false_positive(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "blanks.csv", "id,name,age\n1,Alice,30\n\n2,Bob,25\n\n\n3,Charlie,35\n"
    )

    assert LocalCsvFailureDetector().detect(file_path) is None


def test_blank_line_before_anomaly_does_not_shift_the_reported_row_number(
    tmp_path: Path,
) -> None:
    """`row_number` counts non-blank lines (matching the detector's own
    existing blank-line filtering) — proven directly, since a wrong
    row_number here would make the executor repair the wrong line."""
    file_path = _write(
        tmp_path,
        "blank_then_anomaly.csv",
        "id,name,age\n1,Alice,30\n\n2,Bob,25\n6;Fiona;33\n\n7,George,29\n",
    )

    _, rows = LocalCsvFailureDetector().detect_mixed_delimiter_evidence(file_path)

    assert len(rows) == 1
    assert rows[0].row_number == 4  # header(1), "1,Alice,30"(2), "2,Bob,25"(3), anomaly(4)
    assert rows[0].original_text == "6;Fiona;33"


# --- TEST 21: fewer fields due to alternate delimiter ----------------------


def test_row_with_fewer_fields_due_to_alternate_delimiter_is_repaired_evidence(
    tmp_path: Path,
) -> None:
    file_path = _write(tmp_path, "fewer.csv", "id,name,age\n1,Alice,30\n6;Fiona;33\n")

    _, rows = LocalCsvFailureDetector().detect_mixed_delimiter_evidence(file_path)

    assert rows[0].actual_field_count == 1
    assert rows[0].expected_field_count == 3
    assert rows[0].repaired_text == "6,Fiona,33"


# --- TEST 20 / 22: regression — existing ENGINE_SELECTION case is untouched


def test_existing_engine_selection_case_is_not_reclassified(tmp_path: Path) -> None:
    """The exact fixture from `test_local_csv_failure_detector.py`'s
    `test_detects_engine_selection_failure`: a row with an extra comma
    and no other punctuation at all. No candidate delimiter character is
    even present in the row, so this must remain ENGINE_SELECTION."""
    file_path = _write(
        tmp_path, "engine_selection.csv", "id,name,value\n1,alpha,10\n2,be,ta,20\n3,gamma,30\n"
    )

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.ENGINE_SELECTION
    assert (
        FailureClass.MIXED_DELIMITER not in LocalCsvFailureDetector().detect_all(file_path)
    )


def test_more_fields_with_no_resolvable_alternate_delimiter_stays_engine_selection(
    tmp_path: Path,
) -> None:
    """More fields than expected (not fewer), and again no candidate
    character present at all — must not be misclassified either."""
    file_path = _write(
        tmp_path, "more_fields.csv", "id,name,age\n1,Alice,30\n2,Bob,Extra,25\n"
    )

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.ENGINE_SELECTION


# --- TEST 24: different malformed rows using different alternate delimiters


def test_different_rows_can_use_different_alternate_delimiters(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "mixed_delims.csv",
        "id,name,age\n1,Alice,30\n2;Bob;25\n3,Charlie,35\n4|Diana|28\n",
    )

    result = LocalCsvFailureDetector().detect(file_path)
    _, rows = LocalCsvFailureDetector().detect_mixed_delimiter_evidence(file_path)
    by_row = {r.row_number: r.observed_delimiter for r in rows}

    assert result == FailureClass.MIXED_DELIMITER
    assert by_row == {3: ";", 5: "|"}
