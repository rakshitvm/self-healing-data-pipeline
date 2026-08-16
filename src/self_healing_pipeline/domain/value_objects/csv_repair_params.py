"""CSV repair parameters value object.

Represents the complete set of LLM-derived parameters used to parse or
repair a CSV file. Every LLM-derived repair parameter in Tier 1 must be
representable by `CsvRepairParams`.
"""

import codecs
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CsvEngine(str, Enum):
    """Parsing engine identifier for CSV repair execution."""

    PYTHON = "python"
    C = "c"
    PYARROW = "pyarrow"


class CsvRepairParams(BaseModel):
    """Immutable, LLM-derived parameters for repairing a CSV file.

    Whitespace is deliberately *not* stripped at the model level: a valid
    `delimiter` may itself be whitespace (e.g. `"\\t"` for tab-separated
    data), and model-wide stripping would silently collapse that to an
    empty string before `min_length=1` ever runs, rejecting an otherwise
    correct proposal. `encoding` still has its own whitespace stripped
    (via its field validator below), since a codec name is never
    meaningfully whitespace itself and stray padding there is only ever
    incidental LLM formatting noise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    delimiter: str = Field(min_length=1, max_length=1)
    encoding: str
    header_row: int | None = Field(default=0, ge=0)
    engine: CsvEngine = CsvEngine.PYTHON

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        """Strip incidental whitespace, then ensure the encoding name is a
        Python-recognized codec."""
        value = value.strip()
        try:
            codecs.lookup(value)
        except LookupError as exc:
            raise ValueError(f"Unknown text encoding: {value!r}") from exc
        return value
