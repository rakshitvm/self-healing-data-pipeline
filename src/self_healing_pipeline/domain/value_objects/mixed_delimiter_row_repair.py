"""Row-level evidence/repair for a single MIXED_DELIMITER-affected row.

Unlike every other Tier 1 dimension, MIXED_DELIMITER repair is not
expressible as a single whole-file `CsvRepairParams` field — it is a set
of per-row exceptions layered on top of the file's established
delimiter. `MixedDelimiterRowRepair` is that per-row exception: it is
produced deterministically by `LocalCsvFailureDetector` (never by the
LLM) and carries everything needed to both display the repair for human
approval and apply it byte-for-byte identically to what was approved.
"""

from pydantic import BaseModel, ConfigDict


class MixedDelimiterRowRepair(BaseModel):
    """Deterministic evidence and precomputed fix for one malformed row.

    `row_number` is the 1-based line number in the file (matching how a
    human would count lines, including the header) — not a data-row
    index. `repaired_text` is precomputed at detection time (by
    re-splitting `original_text` with `observed_delimiter` and rejoining
    with the file's established delimiter) so the exact same text shown
    at human approval is what the executor writes — no re-derivation, no
    risk of the two drifting apart, and no path for the LLM to invent or
    modify it (it is never consulted for this value).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=False)

    row_number: int
    expected_field_count: int
    actual_field_count: int
    observed_delimiter: str
    original_text: str
    repaired_text: str
