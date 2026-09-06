"""Focused unit tests for `LocalCsvFailureDetector`'s NO_HEADER and
INVISIBLE_CHARACTERS dimensions.

Every test writes a real temporary CSV file (via pytest's `tmp_path`) and
runs the actual detector against it — no mocking of file I/O. Both
dimensions only apply to a file that is otherwise structurally healthy
(field counts already fully consistent from line 1, i.e.
`_find_header_offset` returns `0`) — see the module docstring of
`local_csv_failure_detector.py`.
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


def test_detects_no_header(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "no_header.csv",
        "1,Alice,30\n2,Bob,25\n3,Carol,40\n4,Dave,22\n",
    )

    assert LocalCsvFailureDetector().detect(file_path) == FailureClass.NO_HEADER


def test_real_header_is_not_misdetected_as_no_header(tmp_path: Path) -> None:
    """A genuine header's text labels (`id`, `name`, `age`) must never
    parse as the numeric type inferred from the data rows below them —
    false-positive guard."""
    file_path = _write(
        tmp_path,
        "real_header.csv",
        "id,name,age\n1,Alice,30\n2,Bob,25\n3,Carol,40\n",
    )

    assert LocalCsvFailureDetector().detect(file_path) is None


def test_all_text_file_is_never_flagged_no_header(tmp_path: Path) -> None:
    """An all-text-column file is undecidable this way (a real header's
    labels are indistinguishable from a data row's text values) — same
    honest scope boundary Tier 2's `looks_headerless` documents."""
    file_path = _write(
        tmp_path,
        "all_text.csv",
        "Alice,Manager,Marketing\nBob,Engineer,Sales\nCarol,Director,Finance\n",
    )

    assert LocalCsvFailureDetector().detect(file_path) is None


def test_detects_invisible_characters_from_utf8_bom(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "bom.csv",
        "﻿id,name,value\n1,alpha,10\n2,beta,20\n",
    )

    failures = LocalCsvFailureDetector().detect_all(file_path)
    assert failures == frozenset({FailureClass.INVISIBLE_CHARACTERS})


def test_detects_invisible_characters_from_zero_width_space(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "zwsp.csv",
        "​id,name,value\n1,alpha,10\n2,beta,20\n",
    )

    failures = LocalCsvFailureDetector().detect_all(file_path)
    assert failures == frozenset({FailureClass.INVISIBLE_CHARACTERS})


def test_healthy_file_has_no_invisible_characters_failure(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "clean.csv",
        "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n",
    )

    assert FailureClass.INVISIBLE_CHARACTERS not in LocalCsvFailureDetector().detect_all(
        file_path
    )


def test_no_header_and_invisible_characters_can_co_occur(tmp_path: Path) -> None:
    """A BOM-prefixed, genuinely headerless file exhibits both
    dimensions at once — neither excludes the other."""
    file_path = _write(
        tmp_path,
        "bom_and_no_header.csv",
        "﻿Alice,1,Marketing\nBob,2,Sales\nCarol,3,Finance\nDave,4,Ops\n",
    )

    failures = LocalCsvFailureDetector().detect_all(file_path)
    assert failures == frozenset({FailureClass.NO_HEADER, FailureClass.INVISIBLE_CHARACTERS})
