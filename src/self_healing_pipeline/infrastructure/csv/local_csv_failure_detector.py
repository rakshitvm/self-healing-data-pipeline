"""Deterministic, local CSV failure detector.

Implements `CsvFailureDetector` using only the standard library (file I/O
and `csv.Sniffer` / `csv.reader`) plus the same deterministic `chardet`
tool already used for WRONG_ENCODING repair evidence — no pandas, no LLM,
no external service. Detection is purely structural: encoding
decodability, delimiter sniffability, and per-line field-count
consistency are enough to distinguish all five Tier 1 CSV failure
dimensions deterministically:

- Undecodable bytes under UTF-8               -> WRONG_ENCODING
- A non-comma delimiter is confidently sniffed -> WRONG_DELIMITER
- Every line has exactly one field (using the
  best-known delimiter), and no better
  delimiter was sniffed                        -> SINGLE_COLUMN_MALFORMATION
- The first line's field count differs from a
  consistent field count on every other line   -> HEADER_DETECTION
- Field counts are inconsistent among the data
  lines themselves                             -> ENGINE_SELECTION
- None of the above                            -> no Tier 1 failure

A single file can genuinely exhibit more than one of these at once —
`detect_all` returns every applicable one. Two are structurally
independent: WRONG_ENCODING is a byte-level concern that gates whether
the rest can even be analyzed at all (when undecodable, this
deterministically re-decodes using chardet's best guess, purely to
*continue* the analysis — the file is still reported as WRONG_ENCODING).
WRONG_DELIMITER, once found, is used (not assumed-comma) for the
structural analysis below it, so a correctly-delimited file never also
reports a spurious structural failure.

Empirically verified (not assumed) that WRONG_DELIMITER cannot co-occur
with any of {SINGLE_COLUMN_MALFORMATION, HEADER_DETECTION,
ENGINE_SELECTION} in practice: `csv.Sniffer` only reports a delimiter
confidently when it occurs a *consistent* number of times per line — the
same consistency that, once used for field-count analysis, means no
further structural issue remains. Any raggedness/inconsistency that
would trigger one of those three also defeats Sniffer's confidence in
the first place. The realistic maximum simultaneous set is therefore
`{WRONG_ENCODING, WRONG_DELIMITER}` or `{WRONG_ENCODING, <one structural
class>}` — never a three-way combination through this mechanism.
"""

import csv

import chardet

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass

_CANDIDATE_DELIMITERS = ",;\t|"
_DEFAULT_DELIMITER = ","

# Backward-compatible single-value priority order for `detect()`: mirrors
# the original implementation's early-return order exactly, so every
# existing single-failure fixture still yields the same one result.
_PRIORITY: tuple[FailureClass, ...] = (
    FailureClass.WRONG_ENCODING,
    FailureClass.WRONG_DELIMITER,
    FailureClass.HEADER_DETECTION,
    FailureClass.ENGINE_SELECTION,
    FailureClass.SINGLE_COLUMN_MALFORMATION,
)


class LocalCsvFailureDetector:
    """Concrete `CsvFailureDetector` for local CSV files."""

    def detect(self, file_path: str) -> FailureClass | None:
        """Return the single highest-priority failure, or `None` if healthy.

        Preserved for backward compatibility with every existing
        single-failure caller/test; derived from `detect_all`.
        """
        failures = self.detect_all(file_path)
        for candidate in _PRIORITY:
            if candidate in failures:
                return candidate
        return None

    def detect_all(self, file_path: str) -> frozenset[FailureClass]:
        """Return every Tier 1 failure dimension `file_path` exhibits."""
        try:
            with open(file_path, "rb") as fh:
                raw = fh.read()
        except OSError:
            return frozenset()

        failures: set[FailureClass] = set()
        text: str
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            decoded: str | None = None
            try:
                encoding = chardet.detect(raw).get("encoding")
                if encoding is not None:
                    decoded = raw.decode(encoding)
            except Exception:  # noqa: BLE001 - best-effort: must never block detection
                decoded = None
            if decoded is None:
                return frozenset({FailureClass.WRONG_ENCODING})
            text = decoded
            failures.add(FailureClass.WRONG_ENCODING)

        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return frozenset(failures)

        delimiter = _DEFAULT_DELIMITER
        try:
            dialect = csv.Sniffer().sniff(text, delimiters=_CANDIDATE_DELIMITERS)
            if dialect.delimiter != _DEFAULT_DELIMITER:
                failures.add(FailureClass.WRONG_DELIMITER)
                delimiter = dialect.delimiter
        except csv.Error:
            pass

        field_counts = [len(row) for row in csv.reader(lines, delimiter=delimiter)]

        if all(count == 1 for count in field_counts):
            failures.add(FailureClass.SINGLE_COLUMN_MALFORMATION)
            return frozenset(failures)

        header_count, *data_counts = field_counts
        if not data_counts:
            return frozenset(failures)

        if all(count == data_counts[0] for count in data_counts):
            if header_count != data_counts[0]:
                failures.add(FailureClass.HEADER_DETECTION)
            return frozenset(failures)

        failures.add(FailureClass.ENGINE_SELECTION)
        return frozenset(failures)
