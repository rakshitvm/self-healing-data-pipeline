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
