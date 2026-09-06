"""Focused unit tests for `LocalCsvFailureDetector`.

Every test writes a real temporary CSV file (via pytest's `tmp_path`) and
runs the actual detector against it — no mocking of file I/O.
"""

from pathlib import Path

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
)


def _write(tmp_path: Path, name: str, content: str, encoding: str = "utf-8") -> str:
    path = tmp_path / name
    path.write_bytes(content.encode(encoding))
    return str(path)


def test_detects_wrong_delimiter(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "wrong_delimiter.csv",
        "id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n",
    )

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.WRONG_DELIMITER


def test_detects_wrong_encoding(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "wrong_encoding.csv",
        "id,name,value\n1,café,10\n2,naïve,20\n3,façade,30\n",
        encoding="latin-1",
    )

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.WRONG_ENCODING


def test_detects_header_detection_failure(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "header_detection.csv",
        "Sales Report - Q3 2026\nid,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.HEADER_DETECTION


def test_detects_engine_selection_failure(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "engine_selection.csv",
        "id,name,value\n1,alpha,10\n2,be,ta,20\n3,gamma,30\n",
    )

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.ENGINE_SELECTION


def test_detects_single_column_malformation(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "single_column.csv",
        "id  name    value\n1   alpha   10\n2  beta     20\n3    gamma  30\n",
    )

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.SINGLE_COLUMN_MALFORMATION


def test_well_formed_csv_has_no_detected_failure(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "well_formed.csv",
        "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )

    assert LocalCsvFailureDetector().detect(file_path) is None


def test_detects_multi_line_preamble_header_detection(tmp_path: Path) -> None:
    """A 3-line preamble (title + metadata, no blank separator needed —
    blank lines are already filtered before this analysis runs) must
    still be recognized as HEADER_DETECTION, not fall through to
    ENGINE_SELECTION: the original implementation only ever handled
    exactly one leading preamble line."""
    file_path = _write(
        tmp_path,
        "multi_line_preamble.csv",
        "Weekly Sales Report\n"
        "Reporting Period: 2026-01-01 to 2026-01-14\n"
        "Prepared by: Data Team\n"
        "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )

    detector = LocalCsvFailureDetector()
    assert detector.detect(file_path) == FailureClass.HEADER_DETECTION
    assert detector.detect_header_offset_evidence(file_path) == 3


def test_multi_line_preamble_with_non_comma_delimiter(tmp_path: Path) -> None:
    """A 2-line, non-delimited preamble ahead of a semicolon-delimited
    header+data region can defeat `csv.Sniffer`'s whole-file confidence
    entirely (empirically verified against the real failing file this
    guards against) — the retry-on-the-tail fallback must still resolve
    both the delimiter and the header offset correctly. Uses enough data
    rows that the tail-retry window (the last few lines) lands entirely
    within the consistent semicolon-delimited region, matching the
    real-world file's own proportions."""
    file_path = _write(
        tmp_path,
        "multi_line_preamble_semicolon.csv",
        "SALES EXPORT\nSource: SAP BW\nid;name;value\n"
        "1;alpha;10\n2;beta;20\n3;gamma;30\n4;delta;40\n5;epsilon;50\n6;zeta;60\n",
    )

    detector = LocalCsvFailureDetector()
    failures = detector.detect_all(file_path)
    assert failures == frozenset({FailureClass.WRONG_DELIMITER, FailureClass.HEADER_DETECTION})
    assert detector.detect_header_offset_evidence(file_path) == 2


def test_mid_file_anomaly_is_not_misread_as_preamble(tmp_path: Path) -> None:
    """A single malformed row near the START of an otherwise-consistent
    file must never be (mis)explained away as "everything up to and
    including it was a preamble" — that would silently discard genuine
    data rows. This is the key regression guard for the N-line-preamble
    generalization: rows 0-2 are good (matching the data's own width),
    row 3 is genuinely malformed (a surplus field, so MIXED_DELIMITER's
    deficit-only resolution can't fix it either), rows 4+ are good again
    — the file must still fall through to ENGINE_SELECTION exactly as it
    did before this feature existed."""
    file_path = _write(
        tmp_path,
        "mid_file_anomaly.csv",
        "id,name,value\n"
        "1,alpha,10\n"
        "2,beta,20\n"
        "3,gamma,30\n"
        "4,delta,extra,40\n"
        "5,epsilon,50\n"
        "6,zeta,60\n"
        "7,eta,70\n",
    )

    detector = LocalCsvFailureDetector()
    assert detector.detect(file_path) == FailureClass.ENGINE_SELECTION
    assert detector.detect_header_offset_evidence(file_path) is None


def test_header_offset_evidence_is_none_for_a_healthy_file(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "well_formed.csv",
        "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )

    assert LocalCsvFailureDetector().detect_header_offset_evidence(file_path) is None


def test_garbled_header_line_is_repaired_not_treated_as_junk(tmp_path: Path) -> None:
    """The live-discovered case: 2 junk preamble lines, then a header
    that itself uses three different delimiters at once
    (`;`, `|`, `,`). Without this fix, `_find_header_offset` would treat
    the garbled header as more junk to skip and promote the first real
    data row into the column names instead."""
    file_path = _write(
        tmp_path,
        "garbled_header.csv",
        "VDVSDVSDVSD\n"
        "SNJSDHSDH\n"
        "id;name|age,state,country\n"
        "1,Alice,30,Georgia,USA\n"
        "2,Bob,25,Georgia,USA\n"
        "3,Charlie,35,Georgia,USA\n",
    )

    detector = LocalCsvFailureDetector()
    assert detector.detect_all(file_path) == frozenset({FailureClass.HEADER_DETECTION})
    assert detector.detect_header_offset_evidence(file_path) == 2

    result = detector.detect_garbled_header_repair(file_path)
    assert result is not None
    delimiter, repair = result
    assert delimiter == ","
    assert repair.row_number == 3
    assert repair.original_text == "id;name|age,state,country"
    assert repair.repaired_text == "id,name,age,state,country"
    assert repair.expected_field_count == 5
    assert repair.actual_field_count == 3


def test_garbled_header_with_no_preceding_junk_still_offsets_to_zero(tmp_path: Path) -> None:
    """A garbled header with no real preamble before it at all (offset
    shifts from 1 down to exactly 0) must still classify as
    HEADER_DETECTION — never NO_HEADER, since the problem is the
    header's *text*, not its absence."""
    file_path = _write(
        tmp_path,
        "garbled_header_no_preamble.csv",
        "id;name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )

    detector = LocalCsvFailureDetector()
    assert detector.detect_all(file_path) == frozenset({FailureClass.HEADER_DETECTION})
    assert detector.detect_header_offset_evidence(file_path) == 0

    result = detector.detect_garbled_header_repair(file_path)
    assert result is not None
    _, repair = result
    assert repair.row_number == 1
    assert repair.repaired_text == "id,name,value"


def test_garbled_header_repair_never_fires_on_genuine_prose_preambles(tmp_path: Path) -> None:
    """Regression guard: every existing multi-line-preamble fixture's
    boundary line is ordinary prose, not a garbled header — the new
    check must find nothing and leave the original offset untouched."""
    file_path = _write(
        tmp_path,
        "multi_line_preamble.csv",
        "Weekly Sales Report\n"
        "Reporting Period: 2026-01-01 to 2026-01-14\n"
        "Prepared by: Data Team\n"
        "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )

    detector = LocalCsvFailureDetector()
    assert detector.detect_header_offset_evidence(file_path) == 3  # unchanged from before
    assert detector.detect_garbled_header_repair(file_path) is None


def test_garbled_header_repair_is_none_when_no_offset_exists(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "well_formed.csv",
        "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )

    assert LocalCsvFailureDetector().detect_garbled_header_repair(file_path) is None
