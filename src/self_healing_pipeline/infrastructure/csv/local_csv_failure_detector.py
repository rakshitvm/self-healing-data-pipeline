"""Deterministic, local CSV failure detector.

Implements `CsvFailureDetector` using only the standard library (file I/O
and `csv.Sniffer` / `csv.reader`) plus the same deterministic `chardet`
tool already used for WRONG_ENCODING repair evidence — no pandas, no LLM,
no external service. Detection is purely structural: encoding
decodability, delimiter sniffability, and per-line field-count
consistency are enough to distinguish all Tier 1 CSV failure dimensions
deterministically:

- Undecodable bytes under UTF-8               -> WRONG_ENCODING
- A non-comma delimiter is confidently sniffed -> WRONG_DELIMITER
- Every line has exactly one field (using the
  best-known delimiter), and no better
  delimiter was sniffed                        -> SINGLE_COLUMN_MALFORMATION
- The first line's field count differs from a
  consistent field count on every other line   -> HEADER_DETECTION
- Field counts are inconsistent among the data
  lines themselves, and every disagreeing row
  resolves unambiguously to a single alternate
  delimiter producing the expected width        -> MIXED_DELIMITER
- Field counts are inconsistent among the data
  lines themselves, and at least one disagreeing
  row does NOT resolve unambiguously             -> ENGINE_SELECTION
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
ENGINE_SELECTION, MIXED_DELIMITER} in practice: `csv.Sniffer` only
reports a delimiter confidently when it occurs a *consistent* number of
times per line — the same consistency that, once used for field-count
analysis, means no further structural issue remains. Any
raggedness/inconsistency that would trigger one of those four also
defeats Sniffer's confidence in the first place. The realistic maximum
simultaneous set is therefore `{WRONG_ENCODING, WRONG_DELIMITER}` or
`{WRONG_ENCODING, <one structural class>}` — never a three-way
combination through this mechanism.

MIXED_DELIMITER (a minority of rows using a different delimiter than the
file's established one) is a further split of what would otherwise fall
into ENGINE_SELECTION: whenever every row whose field count disagrees
with the header can be unambiguously resolved to a single alternate
delimiter that produces exactly the expected width, the file is
MIXED_DELIMITER, not ENGINE_SELECTION. If even one disagreeing row
cannot be resolved this way (zero or more than one qualifying candidate
character), the file falls back to ENGINE_SELECTION exactly as before —
MIXED_DELIMITER is never claimed on ambiguous or unresolvable evidence.
Candidate delimiters for this resolution are discovered dynamically from
the characters actually present in each specific suspicious row — never
a fixed list, and never a character common inside real data values (see
`_EXCLUDED_FROM_DELIMITER_CANDIDACY`) — so this in no way changes what
`_CANDIDATE_DELIMITERS` means for the pre-existing WRONG_DELIMITER/
healthy-file sniffing above.
"""

import csv
import io

import chardet

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.mixed_delimiter_row_repair import (
    MixedDelimiterRowRepair,
)

_CANDIDATE_DELIMITERS = ",;\t|"
_DEFAULT_DELIMITER = ","

# Characters that legitimately occur *inside* ordinary data values and
# must never be treated as a candidate row-level delimiter, even if a
# suspicious row happens to contain one. This is a belt-and-suspenders
# safeguard: the dominant false-positive protection is that a row is
# only ever examined for candidates at all if its field count under the
# established delimiter already disagrees with the header's (see
# `_find_mixed_delimiter_rows`) — a row like `1,John-Smith,30` or
# `2,john@example.com,31` already matches the header's field count under
# the established delimiter, so it is never even considered "suspicious"
# and candidate search never runs on it in the first place.
_EXCLUDED_FROM_DELIMITER_CANDIDACY = frozenset({"-", "_", ".", "'", '"', "@", " "})

# Backward-compatible single-value priority order for `detect()`: mirrors
# the original implementation's early-return order exactly, so every
# existing single-failure fixture still yields the same one result.
# MIXED_DELIMITER is placed adjacent to ENGINE_SELECTION because the two
# are mutually exclusive by construction (see module docstring) — the
# file's data rows either all resolve unambiguously (MIXED_DELIMITER) or
# they don't (ENGINE_SELECTION), never both at once — so their relative
# priority ordering here never actually matters in practice.
_PRIORITY: tuple[FailureClass, ...] = (
    FailureClass.WRONG_ENCODING,
    FailureClass.WRONG_DELIMITER,
    FailureClass.HEADER_DETECTION,
    FailureClass.MIXED_DELIMITER,
    FailureClass.ENGINE_SELECTION,
    FailureClass.SINGLE_COLUMN_MALFORMATION,
)


def _resolve_row_delimiter(
    raw_line: str, established_delimiter: str, expected_field_count: int
) -> str | None:
    """Return the single alternate delimiter that resolves `raw_line` to
    exactly `expected_field_count` fields, or `None` if zero or more than
    one candidate qualifies.

    Candidates are the non-alphanumeric characters actually present in
    `raw_line`, minus `_EXCLUDED_FROM_DELIMITER_CANDIDACY` and minus
    `established_delimiter` itself — never a fixed list. Each candidate
    is tried via `csv.reader` (so quoting is respected: a candidate
    character that only appears inside an already-quoted field would
    still not split the row differently, since `csv.reader` never splits
    inside quotes). Never guesses: more than one qualifying candidate is
    treated exactly like zero — unresolved.
    """
    candidates = {
        ch
        for ch in raw_line
        if not ch.isalnum()
        and ch not in _EXCLUDED_FROM_DELIMITER_CANDIDACY
        and ch != established_delimiter
    }

    qualifying: list[str] = []
    for candidate in sorted(candidates):
        try:
            fields = next(csv.reader([raw_line], delimiter=candidate))
        except csv.Error:
            continue
        if len(fields) == expected_field_count:
            qualifying.append(candidate)

    if len(qualifying) == 1:
        return qualifying[0]
    return None


def _write_csv_row(fields: list[str], delimiter: str) -> str:
    """Serialize `fields` with `delimiter`, CSV-quoting as needed."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="")
    writer.writerow(fields)
    return buffer.getvalue()


def _find_mixed_delimiter_rows(
    numbered_data_lines: list[tuple[int, str]],
    established_delimiter: str,
    expected_field_count: int,
) -> tuple[MixedDelimiterRowRepair, ...] | None:
    """Resolve every data row whose field count (under
    `established_delimiter`) disagrees with `expected_field_count`.

    Returns `None` — never a partial/guessed result — if any disagreeing
    row cannot be unambiguously resolved to a single alternate delimiter.
    Returns an empty tuple if no row disagrees at all. `row_number` in
    each result is the 1-based line number in the original file
    (including the header and any blank lines that were filtered out
    before this function is called), matching `numbered_data_lines`.
    """
    repairs: list[MixedDelimiterRowRepair] = []
    for row_number, raw_line in numbered_data_lines:
        actual_fields = next(csv.reader([raw_line], delimiter=established_delimiter))
        actual_count = len(actual_fields)
        if actual_count == expected_field_count:
            continue

        observed_delimiter = _resolve_row_delimiter(
            raw_line, established_delimiter, expected_field_count
        )
        if observed_delimiter is None:
            return None

        resolved_fields = next(csv.reader([raw_line], delimiter=observed_delimiter))
        repaired_text = _write_csv_row(resolved_fields, established_delimiter)
        repairs.append(
            MixedDelimiterRowRepair(
                row_number=row_number,
                expected_field_count=expected_field_count,
                actual_field_count=actual_count,
                observed_delimiter=observed_delimiter,
                original_text=raw_line,
                repaired_text=repaired_text,
            )
        )
    return tuple(repairs)


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

        # Field counts disagree among the data rows themselves. Before
        # falling back to the pre-existing ENGINE_SELECTION verdict, see
        # whether every disagreeing row resolves unambiguously to a
        # single alternate delimiter — `lines`, `delimiter`, and
        # `header_count` are already exactly what's needed, computed
        # above, unchanged.
        numbered_data_lines = list(enumerate(lines, start=1))[1:]
        resolved = _find_mixed_delimiter_rows(numbered_data_lines, delimiter, header_count)
        if resolved is not None and len(resolved) > 0:
            failures.add(FailureClass.MIXED_DELIMITER)
            return frozenset(failures)

        failures.add(FailureClass.ENGINE_SELECTION)
        return frozenset(failures)

    def detect_mixed_delimiter_evidence(
        self, file_path: str
    ) -> tuple[str, tuple[MixedDelimiterRowRepair, ...]]:
        """Return `(established_delimiter, row_evidence)` for a
        MIXED_DELIMITER file, or `(established_delimiter, ())` if the
        file has no such evidence.

        Additive capability (mirroring `detect_all`), used by the
        workflow's `propose` node to inject deterministic row-level
        evidence — and to force-override whatever the LLM proposed for
        `delimiter` — never derived from or overridden by the LLM.
        Performs the same decode/sniff/field-count analysis as
        `detect_all`, independently, so this detector's public methods
        each remain a single, self-contained, read-only pass over the
        file.
        """
        try:
            with open(file_path, "rb") as fh:
                raw = fh.read()
        except OSError:
            return (_DEFAULT_DELIMITER, ())

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            encoding = chardet.detect(raw).get("encoding")
            if encoding is None:
                return (_DEFAULT_DELIMITER, ())
            try:
                text = raw.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                return (_DEFAULT_DELIMITER, ())

        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return (_DEFAULT_DELIMITER, ())

        delimiter = _DEFAULT_DELIMITER
        try:
            dialect = csv.Sniffer().sniff(text, delimiters=_CANDIDATE_DELIMITERS)
            if dialect.delimiter != _DEFAULT_DELIMITER:
                delimiter = dialect.delimiter
        except csv.Error:
            pass

        field_counts = [len(row) for row in csv.reader(lines, delimiter=delimiter)]
        if all(count == 1 for count in field_counts):
            return (delimiter, ())

        header_count, *data_counts = field_counts
        if not data_counts or all(count == data_counts[0] for count in data_counts):
            return (delimiter, ())

        numbered_data_lines = list(enumerate(lines, start=1))[1:]
        resolved = _find_mixed_delimiter_rows(numbered_data_lines, delimiter, header_count)
        return (delimiter, resolved or ())
