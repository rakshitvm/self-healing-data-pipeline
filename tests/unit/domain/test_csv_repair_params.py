"""Focused unit tests for `CsvRepairParams`.

Regression coverage for a real defect found during live E2E validation:
model-wide `str_strip_whitespace=True` silently stripped a whitespace
`delimiter` (e.g. a tab) down to an empty string before `min_length=1`
ever ran, rejecting an otherwise valid, LLM-proposed tab delimiter for
whitespace-separated ("single column malformation") data.
"""

import pytest
from pydantic import ValidationError

from self_healing_pipeline.domain.value_objects.csv_repair_params import (
    CsvEngine,
    CsvRepairParams,
)


def test_tab_delimiter_is_accepted() -> None:
    params = CsvRepairParams(delimiter="\t", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    assert params.delimiter == "\t"


def test_space_delimiter_is_accepted() -> None:
    """Space is also a legitimate whitespace delimiter, not just tab."""
    params = CsvRepairParams(delimiter=" ", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    assert params.delimiter == " "


def test_empty_delimiter_is_still_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        CsvRepairParams(delimiter="", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)


def test_multi_character_delimiter_is_still_rejected() -> None:
    with pytest.raises(ValidationError, match="at most 1 character"):
        CsvRepairParams(delimiter=";;", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)


def test_ordinary_single_character_delimiter_still_works() -> None:
    params = CsvRepairParams(delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON)

    assert params.delimiter == ";"


def test_encoding_whitespace_is_still_stripped() -> None:
    """Preserved behavior: `encoding` (unlike `delimiter`) should still
    tolerate incidental LLM-formatting whitespace around a real value."""
    params = CsvRepairParams(delimiter=",", encoding="  utf-8  ", header_row=0, engine=CsvEngine.PYTHON)

    assert params.encoding == "utf-8"


def test_unknown_encoding_is_still_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown text encoding"):
        CsvRepairParams(delimiter=",", encoding="not-a-real-codec", header_row=0, engine=CsvEngine.PYTHON)
