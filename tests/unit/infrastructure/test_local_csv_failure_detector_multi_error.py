"""Focused unit tests for `LocalCsvFailureDetector.detect_all` (multi-error
detection). Kept separate from `test_local_csv_failure_detector.py` (left
untouched, still exercising the original single-value `detect`).

Every test writes a real temporary file and runs the actual detector —
no mocking of file I/O or of `chardet`/`csv.Sniffer`.
"""

from pathlib import Path

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
)
from tests.fixtures.multi_error_csv import (
    encoding_and_single_column_bytes,
    two_failure_bytes,
)


def _write_bytes(tmp_path: Path, name: str, content: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


def test_detect_all_returns_empty_set_for_healthy_file(tmp_path: Path) -> None:
    file_path = _write_bytes(
        tmp_path, "healthy.csv", b"id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n"
    )

    assert LocalCsvFailureDetector().detect_all(file_path) == frozenset()


def test_detect_all_returns_single_value_set_for_ordinary_single_failure(tmp_path: Path) -> None:
    file_path = _write_bytes(
        tmp_path, "wrong_delimiter.csv", b"id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n"
    )

    assert LocalCsvFailureDetector().detect_all(file_path) == frozenset(
        {FailureClass.WRONG_DELIMITER}
    )


def test_detect_all_finds_two_simultaneous_failures_encoding_and_delimiter(
    tmp_path: Path,
) -> None:
    file_path = _write_bytes(tmp_path, "multi_error.csv", two_failure_bytes())

    result = LocalCsvFailureDetector().detect_all(file_path)

    assert result == frozenset({FailureClass.WRONG_ENCODING, FailureClass.WRONG_DELIMITER})


def test_detect_all_finds_two_simultaneous_failures_encoding_and_single_column(
    tmp_path: Path,
) -> None:
    file_path = _write_bytes(tmp_path, "multi_error2.csv", encoding_and_single_column_bytes())

    result = LocalCsvFailureDetector().detect_all(file_path)

    assert result == frozenset(
        {FailureClass.WRONG_ENCODING, FailureClass.SINGLE_COLUMN_MALFORMATION}
    )


def test_detect_derives_backward_compatible_primary_value_from_detect_all(
    tmp_path: Path,
) -> None:
    """`detect()` must still return exactly one value — the same
    priority-ordered primary result existing callers already depend on —
    even though `detect_all()` now finds two."""
    file_path = _write_bytes(tmp_path, "multi_error.csv", two_failure_bytes())
    detector = LocalCsvFailureDetector()

    assert detector.detect(file_path) == FailureClass.WRONG_ENCODING
    assert detector.detect_all(file_path) == frozenset(
        {FailureClass.WRONG_ENCODING, FailureClass.WRONG_DELIMITER}
    )


def test_missing_file_returns_empty_set() -> None:
    assert LocalCsvFailureDetector().detect_all("/nonexistent/path.csv") == frozenset()
