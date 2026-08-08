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
    """Immutable, LLM-derived parameters for repairing a CSV file."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    delimiter: str = Field(min_length=1, max_length=1)
    encoding: str
    header_row: int | None = Field(default=0, ge=0)
    engine: CsvEngine = CsvEngine.PYTHON

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        """Ensure the encoding name is a Python-recognized codec."""
        try:
            codecs.lookup(value)
        except LookupError as exc:
            raise ValueError(f"Unknown text encoding: {value!r}") from exc
        return value
