"""Deterministic, local CSV failure detector.

Implements `CsvFailureDetector` using only the standard library (file I/O
and `csv.Sniffer` / `csv.reader`) — no pandas, no LLM, no external
service. Detection is purely structural: encoding decodability, delimiter
sniffability, and per-line field-count consistency are enough to
distinguish all five Tier 1 CSV failure dimensions deterministically:

- Undecodable bytes under UTF-8               -> WRONG_ENCODING
- A non-comma delimiter is confidently sniffed -> WRONG_DELIMITER
- Every line has exactly one comma-separated
  field, and no better delimiter was sniffed   -> SINGLE_COLUMN_MALFORMATION
- The first line's field count differs from a
  consistent field count on every other line   -> HEADER_DETECTION
- Field counts are inconsistent among the data
  lines themselves                             -> ENGINE_SELECTION
- None of the above                            -> no Tier 1 failure (None)
"""

import csv

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass

_CANDIDATE_DELIMITERS = ",;\t|"
_DEFAULT_DELIMITER = ","


class LocalCsvFailureDetector:
    """Concrete `CsvFailureDetector` for local CSV files."""

    def detect(self, file_path: str) -> FailureClass | None:
        try:
            with open(file_path, "rb") as fh:
                raw = fh.read()
        except OSError:
            return None

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return FailureClass.WRONG_ENCODING

        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return None

        try:
            dialect = csv.Sniffer().sniff(text, delimiters=_CANDIDATE_DELIMITERS)
            if dialect.delimiter != _DEFAULT_DELIMITER:
                return FailureClass.WRONG_DELIMITER
        except csv.Error:
            pass

        field_counts = [len(row) for row in csv.reader(lines, delimiter=_DEFAULT_DELIMITER)]

        if all(count == 1 for count in field_counts):
            return FailureClass.SINGLE_COLUMN_MALFORMATION

        header_count, *data_counts = field_counts
        if not data_counts:
            return None

        if all(count == data_counts[0] for count in data_counts):
            if header_count != data_counts[0]:
                return FailureClass.HEADER_DETECTION
            return None

        return FailureClass.ENGINE_SELECTION
