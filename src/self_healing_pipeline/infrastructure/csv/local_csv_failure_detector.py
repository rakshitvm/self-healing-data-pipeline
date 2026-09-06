"""Deterministic, local CSV failure detector.

Implements `CsvFailureDetector` using only the standard library
(`csv.Sniffer` / `csv.reader`) plus `chardet` for encoding detection —
no pandas, no LLM, no external service. Detection is purely structural:
encoding decodability, delimiter sniffability, and per-line field-count
consistency distinguish the Tier 1 failure dimensions:

- Undecodable bytes under UTF-8               -> WRONG_ENCODING
- A non-comma delimiter is confidently sniffed -> WRONG_DELIMITER
- Every line has exactly one field (using the
  best-known delimiter), and no better
  delimiter was sniffed                        -> SINGLE_COLUMN_MALFORMATION
- Some smallest N-line leading preamble (N >= 1,
  bounded) can be dropped so every remaining
  line shares one consistent field count        -> HEADER_DETECTION
- No leading preamble is needed (field counts
  already consistent from line 1), but the
  first line itself parses as data — every
  column that looks numeric in the other lines
  also parses as numeric on line 1              -> NO_HEADER
- No leading preamble is needed, and either a
  UTF-8 BOM opens the file or the first line
  contains a zero-width/invisible character     -> INVISIBLE_CHARACTERS
- Field counts are inconsistent among the data
  lines themselves (no leading preamble resolves
  it), and every disagreeing row resolves to a
  single alternate delimiter producing the
  expected width                                -> MIXED_DELIMITER
- Field counts are inconsistent among the data
  lines, and at least one disagreeing row does
  NOT resolve unambiguously                     -> ENGINE_SELECTION
- None of the above                            -> no Tier 1 failure

NO_HEADER and INVISIBLE_CHARACTERS only apply to an otherwise
structurally healthy file (no preamble offset needed) — they can't
co-occur with HEADER_DETECTION, MIXED_DELIMITER, or ENGINE_SELECTION,
since those require a field-count inconsistency a preamble offset
doesn't resolve.

HEADER_DETECTION also covers a header line malformed by several
delimiter characters used interchangeably (e.g.
`id;name|age,state,country`), which `_find_header_offset` alone would
misread as more junk to skip. `_resolve_garbled_header_line` checks
whether the line right before the offset normalizes cleanly (every
stray character swapped for the established delimiter) to the target
width; if so, that line is the real header, and
`detect_garbled_header_repair` returns the precomputed rewrite the same
way `detect_mixed_delimiter_evidence` does for data rows, reusing
`MixedDelimiterCsvRepairExecutor` to apply it.

`detect_all` returns every dimension a file exhibits — a file can hit
more than one at once. WRONG_ENCODING gates the rest: an undecodable
file is re-decoded with chardet's best guess purely to continue the
structural analysis, and is still reported as WRONG_ENCODING.
WRONG_DELIMITER, once found, is used (not assumed-comma) for the
structural checks below it, so a correctly-delimited file doesn't also
report a spurious structural failure.

WRONG_DELIMITER doesn't co-occur with the four structural dimensions in
practice: `csv.Sniffer` only reports a delimiter confidently when it's
consistent per line, and that same consistency rules out a structural
issue. Realistic combinations are therefore `{WRONG_ENCODING,
WRONG_DELIMITER}` or `{WRONG_ENCODING, <one structural class>}`.

MIXED_DELIMITER is the resolvable half of what would otherwise be
ENGINE_SELECTION: if every disagreeing row can be fixed unambiguously,
it's MIXED_DELIMITER; if even one can't, it's ENGINE_SELECTION.
Candidate delimiters are discovered per-row from the characters actually
present (never a fixed list, and never a character common in real data —
see `_EXCLUDED_FROM_DELIMITER_CANDIDACY`).

Two resolution tiers, in order:
1. Whole-row (`_resolve_row_delimiter`): the entire row re-parses under
   one alternate delimiter (e.g. a row with zero commas, entirely
   semicolon-delimited instead).
2. Single-field sub-split (`_resolve_single_field_subsplit`), only for a
   field-count deficit: one field internally uses a different character
   — e.g. `id,name,age` where a field is really `"6;Fiona;33"`. Never
   attempted for a surplus (too many fields), since merging fields back
   together means guessing which two to join and with what — a surplus
   row always falls back to ENGINE_SELECTION.

Both tiers: more than one qualifying resolution is treated as zero —
unresolved rather than guessed.
"""

import csv
import io

import chardet

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.mixed_delimiter_row_repair import (
    MixedDelimiterRowRepair,
)

_CANDIDATE_DELIMITERS = ",;\t|:"
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

# Bound on how many leading lines `_find_header_offset` will consider
# dropping as a preamble. Kept small and explicit: a genuine preamble
# (report titles, generation metadata, blank separators) is realistically
# a handful of lines; capping this prevents a large, unrelated anomaly
# deep in the file from ever being (mis)explained away as "everything
# before it was a preamble."
_MAX_PREAMBLE_LINES_SEARCHED = 10

# Known zero-width / invisible Unicode characters that legitimately never
# belong in a column name: BOM-as-character (left over after UTF-8
# decoding an actual BOM byte sequence), zero-width space, zero-width
# non-joiner, zero-width joiner, and word joiner. A small, explicit set —
# never a heuristic guess.
_INVISIBLE_CHARACTERS = frozenset({"﻿", "​", "‌", "‍", "⁠"})
_UTF8_BOM = b"\xef\xbb\xbf"

# Backward-compatible single-value priority order for `detect()`: mirrors
# the original implementation's early-return order exactly, so every
# existing single-failure fixture still yields the same one result.
# MIXED_DELIMITER is placed adjacent to ENGINE_SELECTION because the two
# are mutually exclusive by construction (see module docstring) — the
# file's data rows either all resolve unambiguously (MIXED_DELIMITER) or
# they don't (ENGINE_SELECTION), never both at once — so their relative
# priority ordering here never actually matters in practice. NO_HEADER
# and INVISIBLE_CHARACTERS are likewise mutually exclusive with
# HEADER_DETECTION/MIXED_DELIMITER/ENGINE_SELECTION (see module
# docstring), so their placement relative to those four never matters
# either — only relative to each other and to WRONG_ENCODING/
# WRONG_DELIMITER, with which they can genuinely co-occur.
_PRIORITY: tuple[FailureClass, ...] = (
    FailureClass.WRONG_ENCODING,
    FailureClass.WRONG_DELIMITER,
    FailureClass.HEADER_DETECTION,
    FailureClass.NO_HEADER,
    FailureClass.INVISIBLE_CHARACTERS,
    FailureClass.MIXED_DELIMITER,
    FailureClass.ENGINE_SELECTION,
    FailureClass.SINGLE_COLUMN_MALFORMATION,
)


def _find_header_offset(field_counts: list[int]) -> int | None:
    """Return the smallest `N` such that `field_counts[N:]` is fully
    consistent (every remaining line shares one field count) and has
    length >= 2 (a header plus at least one data row), searching `N`
    from 0 up to `_MAX_PREAMBLE_LINES_SEARCHED`.

    `N=0` means the file is already consistent from line 1 (not itself
    evidence of an offset; callers only treat a positive `N` as
    HEADER_DETECTION). Returns `None` if no such `N` exists within the
    search bound — an anomaly in the middle or end of the file (not a
    leading preamble) falls through to the MIXED_DELIMITER/
    ENGINE_SELECTION analysis instead.

    A candidate `N` is only accepted if every discarded line
    (`field_counts[:N]`) has a field count that differs from the
    retained block's. Otherwise a genuinely malformed row near the start
    of an otherwise normal file could be misread as "the good rows
    before it were all preamble."
    """
    limit = min(_MAX_PREAMBLE_LINES_SEARCHED, len(field_counts) - 2)
    for candidate in range(0, limit + 1):
        tail = field_counts[candidate:]
        if len(tail) < 2 or not all(count == tail[0] for count in tail):
            continue
        preamble = field_counts[:candidate]
        if any(count == tail[0] for count in preamble):
            continue
        return candidate
    return None


def _resolve_garbled_header_line(
    raw_line: str, established_delimiter: str, target_field_count: int
) -> tuple[str, str] | None:
    """Return `(repaired_text, observed_delimiter_display)` if
    normalizing *every* stray delimiter-like character in `raw_line` to
    `established_delimiter` produces exactly `target_field_count`
    fields, or `None` otherwise.

    Handles a header row that uses several different delimiter
    characters interchangeably — e.g. `id;name|age,state,country`
    (semicolon, pipe, and comma all in one line). Candidates are
    discovered the same way as elsewhere in this module (non-
    alphanumeric, not in `_EXCLUDED_FROM_DELIMITER_CANDIDACY`, not
    `established_delimiter` itself). Unlike `_resolve_row_delimiter`
    (which picks one of several candidates and bails on ambiguity), this
    normalizes all of them at once, since the premise is that several
    characters were all standing in for the same delimiter on one line.
    Returns `None` when there are no candidates at all (an ordinary
    prose preamble line is left untouched).
    """
    candidates = {
        ch
        for ch in raw_line
        if not ch.isalnum()
        and ch not in _EXCLUDED_FROM_DELIMITER_CANDIDACY
        and ch != established_delimiter
    }
    if not candidates:
        return None

    normalized = raw_line
    for ch in candidates:
        normalized = normalized.replace(ch, established_delimiter)

    try:
        fields = next(csv.reader([normalized], delimiter=established_delimiter))
    except csv.Error:
        return None
    if len(fields) != target_field_count:
        return None

    return normalized, "/".join(sorted(candidates))


def _parses_as_number(value: str) -> bool:
    """`True` if `value` parses as a Python float (covers int-looking
    strings too, e.g. `"312"`, without needing a separate int check)."""
    try:
        float(value)
    except ValueError:
        return False
    return True


def _infer_numeric_columns(data_rows: list[list[str]]) -> set[int]:
    """Return the column indices where every value across `data_rows`
    parses as a number — a column's expected type, inferred purely from
    the data rows themselves (never from the candidate header row), the
    same reasoning `looks_headerless` (Tier 2) applies against an
    external baseline instead."""
    if not data_rows:
        return set()
    width = len(data_rows[0])
    numeric_columns: set[int] = set()
    for index in range(width):
        values = [row[index] for row in data_rows if index < len(row)]
        if values and all(_parses_as_number(v) for v in values):
            numeric_columns.add(index)
    return numeric_columns


def _looks_headerless(header_fields: list[str], data_rows: list[list[str]]) -> bool:
    """`True` if `header_fields` (the file's first row) itself looks
    like a data row rather than a real header: every column inferred as
    numeric from `data_rows` also parses as numeric on `header_fields`.

    Requires at least one numeric column to exist at all — an all-text
    file is undecidable this way (a real header's text labels are
    indistinguishable from a data row's text values) and always returns
    `False`, an honest scope boundary, not a bug to fix later — same
    boundary Tier 2's `looks_headerless` already documents.
    """
    numeric_columns = _infer_numeric_columns(data_rows)
    if not numeric_columns:
        return False
    return all(
        index < len(header_fields) and _parses_as_number(header_fields[index])
        for index in numeric_columns
    )


# How many trailing lines to retry sniffing on when the whole-file sniff
# fails outright — see `_sniff_delimiter`. Deliberately small: the point
# is to exclude a leading preamble (which `_find_header_offset` bounds at
# `_MAX_PREAMBLE_LINES_SEARCHED`), so a short, fixed window from the very
# end of the file is far more likely to land entirely inside the
# homogeneous data region than a large one is — a large window risks
# re-including the very preamble lines that broke the first attempt.
_SNIFF_RETRY_TAIL_LINES = 5


def _sniff_delimiter(text: str, lines: list[str]) -> str:
    """Best-effort delimiter sniff; never raises, defaults to
    `_DEFAULT_DELIMITER` if nothing can be determined.

    Tries the whole file first. A leading preamble (title/metadata lines
    with zero occurrences of the real delimiter) can defeat `csv.Sniffer`
    entirely, since it needs a broadly consistent delimiter count across
    every line. If the whole-file attempt fails, retry on just the last
    `_SNIFF_RETRY_TAIL_LINES` of `lines` — a preamble is only ever at the
    front of a file, so the tail gives Sniffer a cleaner signal.
    """
    try:
        return csv.Sniffer().sniff(text, delimiters=_CANDIDATE_DELIMITERS).delimiter
    except csv.Error:
        pass

    tail_text = "\n".join(lines[-min(len(lines), _SNIFF_RETRY_TAIL_LINES) :])
    try:
        return csv.Sniffer().sniff(tail_text, delimiters=_CANDIDATE_DELIMITERS).delimiter
    except csv.Error:
        return _DEFAULT_DELIMITER


def _contains_invisible_characters(raw: bytes, header_fields: list[str]) -> bool:
    """`True` if `raw` opens with a UTF-8 BOM, or any field of
    `header_fields` contains a character from `_INVISIBLE_CHARACTERS`."""
    if raw.startswith(_UTF8_BOM):
        return True
    return any(ch in field for field in header_fields for ch in _INVISIBLE_CHARACTERS)


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


def _resolve_single_field_subsplit(
    fields: list[str], established_delimiter: str, deficit: int
) -> tuple[int, str, list[str]] | None:
    """Find the one `(field_index, candidate_delimiter, sub_fields)`
    where re-splitting `fields[field_index]` by `candidate_delimiter`
    adds exactly `deficit` fields to the row.

    Handles a row that is *mostly* correctly delimited except that one
    field internally uses a different character — e.g. `id,name,age`
    where one field is really `"6;Fiona;33"`, three semicolon-joined
    values collapsed into what `established_delimiter` sees as a single
    field. Never guesses: if more than one `(field, candidate)` pair
    would each independently close the deficit — whether in different
    fields or via different candidate characters within the same field
    — the row is left unresolved, mirroring `_resolve_row_delimiter`'s
    own ambiguity policy exactly. Only ever resolves one field via one
    character; a resulting sub-field is never itself re-examined for a
    further nested sub-split, and this is never tried for a surplus
    (too many fields) — merging fields back together would require
    guessing which two to join and with what glue, a materially
    riskier operation than splitting one further.
    """
    matches: list[tuple[int, str, list[str]]] = []
    for index, field in enumerate(fields):
        candidates = {
            ch
            for ch in field
            if not ch.isalnum()
            and ch not in _EXCLUDED_FROM_DELIMITER_CANDIDACY
            and ch != established_delimiter
        }
        for candidate in sorted(candidates):
            try:
                sub_fields = next(csv.reader([field], delimiter=candidate))
            except csv.Error:
                continue
            if len(sub_fields) - 1 == deficit:
                matches.append((index, candidate, sub_fields))

    if len(matches) == 1:
        return matches[0]
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
    row cannot be unambiguously resolved. Two resolution tiers are
    tried, in order, per disagreeing row: first, whether the *entire*
    row re-parses correctly under one alternate delimiter
    (`_resolve_row_delimiter` — e.g. a row with zero commas at all);
    second, only for a field-count deficit, whether exactly one field
    internally uses a different character that closes the gap
    (`_resolve_single_field_subsplit` — e.g. a row that's otherwise
    correctly comma-delimited except one field is really several
    semicolon-joined values). If neither tier resolves a row
    unambiguously, the whole result is `None` — never a partial fix.
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
        if observed_delimiter is not None:
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
            continue

        if actual_count < expected_field_count:
            deficit = expected_field_count - actual_count
            subsplit = _resolve_single_field_subsplit(
                actual_fields, established_delimiter, deficit
            )
            if subsplit is not None:
                field_index, candidate, sub_fields = subsplit
                resolved_fields = (
                    actual_fields[:field_index] + sub_fields + actual_fields[field_index + 1 :]
                )
                repaired_text = _write_csv_row(resolved_fields, established_delimiter)
                repairs.append(
                    MixedDelimiterRowRepair(
                        row_number=row_number,
                        expected_field_count=expected_field_count,
                        actual_field_count=actual_count,
                        observed_delimiter=candidate,
                        original_text=raw_line,
                        repaired_text=repaired_text,
                    )
                )
                continue

        return None
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
            except Exception:  
                decoded = None
            if decoded is None:
                return frozenset({FailureClass.WRONG_ENCODING})
            text = decoded
            failures.add(FailureClass.WRONG_ENCODING)

        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return frozenset(failures)

        delimiter = _sniff_delimiter(text, lines)
        if delimiter != _DEFAULT_DELIMITER:
            failures.add(FailureClass.WRONG_DELIMITER)

        parsed_rows = list(csv.reader(lines, delimiter=delimiter))
        field_counts = [len(row) for row in parsed_rows]

        if all(count == 1 for count in field_counts):
            failures.add(FailureClass.SINGLE_COLUMN_MALFORMATION)
            return frozenset(failures)

        offset = _find_header_offset(field_counts)
        if offset is not None:
            # The line immediately before the accepted boundary might
            # not be genuine junk at all — it might be the real header,
            # just mangled by several different delimiter characters
            # used interchangeably (see `_resolve_garbled_header_line`).
            # Only checked when there's a boundary to check at all
            # (`offset > 0`); can legitimately push the effective offset
            # down to exactly 0 (the garbled header was the very first
            # line, no real preamble before it).
            garbled = (
                _resolve_garbled_header_line(lines[offset - 1], delimiter, field_counts[offset])
                if offset > 0
                else None
            )
            effective_offset = offset - 1 if garbled is not None else offset

            if effective_offset > 0 or garbled is not None:
                failures.add(FailureClass.HEADER_DETECTION)
            else:
                # Fully consistent from line 1, and no garbled header
                # found either — no preamble to skip. Only now do the
                # two dimensions that only make sense on an otherwise-
                # structurally-healthy file apply: the "header" row
                # might actually be data (NO_HEADER), or it might carry
                # invisible/BOM contamination (INVISIBLE_CHARACTERS).
                # Neither excludes the other.
                if _looks_headerless(parsed_rows[0], parsed_rows[1:]):
                    failures.add(FailureClass.NO_HEADER)
                if _contains_invisible_characters(raw, parsed_rows[0]):
                    failures.add(FailureClass.INVISIBLE_CHARACTERS)
            return frozenset(failures)

        # No small leading preamble resolves the inconsistency. Before
        # falling back to the pre-existing ENGINE_SELECTION verdict, see
        # whether every disagreeing row resolves unambiguously to a
        # single alternate delimiter — `lines`, `delimiter`, and the
        # first line's own field count are already exactly what's
        # needed, computed above, unchanged.
        numbered_data_lines = list(enumerate(lines, start=1))[1:]
        resolved = _find_mixed_delimiter_rows(numbered_data_lines, delimiter, field_counts[0])
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

        delimiter = _sniff_delimiter(text, lines)

        field_counts = [len(row) for row in csv.reader(lines, delimiter=delimiter)]
        if all(count == 1 for count in field_counts):
            return (delimiter, ())

        header_count, *data_counts = field_counts
        if not data_counts or all(count == data_counts[0] for count in data_counts):
            return (delimiter, ())

        numbered_data_lines = list(enumerate(lines, start=1))[1:]
        resolved = _find_mixed_delimiter_rows(numbered_data_lines, delimiter, header_count)
        return (delimiter, resolved or ())

    def detect_header_offset_evidence(self, file_path: str) -> int | None:
        """Return the deterministic `header_row` for a HEADER_DETECTION
        file (the number of leading preamble lines to skip), or `None`
        if this detector finds no such offset.

        Additive capability (mirroring `detect_mixed_delimiter_evidence`),
        used by the workflow's `propose` node to force-override whatever
        the LLM proposed for `header_row` — never left to the LLM to
        guess, since a multi-line preamble can push the real header
        beyond what the (5-line-capped) sample even shows it. Performs
        the same decode/sniff/field-count analysis as `detect_all`,
        independently, so this detector's public methods each remain a
        single, self-contained, read-only pass over the file.

        Accounts for a garbled header line the same way `detect_all`
        does (see `_resolve_garbled_header_line`): if the line right
        before the raw offset turns out to be a mangled real header
        rather than junk, the returned offset is one smaller — including
        possibly `0`, still correctly non-`None` here, since that means
        "no preamble, but the header itself still needs its text fixed."
        """
        try:
            with open(file_path, "rb") as fh:
                raw = fh.read()
        except OSError:
            return None

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            encoding = chardet.detect(raw).get("encoding")
            if encoding is None:
                return None
            try:
                text = raw.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                return None

        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return None

        delimiter = _sniff_delimiter(text, lines)

        field_counts = [len(row) for row in csv.reader(lines, delimiter=delimiter)]
        if all(count == 1 for count in field_counts):
            return None

        offset = _find_header_offset(field_counts)
        if offset is None:
            return None

        garbled = (
            _resolve_garbled_header_line(lines[offset - 1], delimiter, field_counts[offset])
            if offset > 0
            else None
        )
        effective_offset = offset - 1 if garbled is not None else offset
        return effective_offset if effective_offset > 0 or garbled is not None else None

    def detect_garbled_header_repair(
        self, file_path: str
    ) -> tuple[str, MixedDelimiterRowRepair] | None:
        """Return `(established_delimiter, repair)` for a HEADER_DETECTION
        file whose header line itself uses several different delimiter
        characters interchangeably (e.g. `id;name|age,state,country`),
        or `None` if no such line is found.

        Additive capability (mirroring `detect_mixed_delimiter_evidence`
        and `detect_header_offset_evidence`), used by the workflow's
        `propose` node to force-correct both `delimiter` and
        `mixed_delimiter_rows` — never left to the LLM to guess, same
        reasoning as every other deterministic override in this
        codebase. `repair.row_number` is the 1-based line number of the
        header line itself (the same convention `MixedDelimiterRowRepair`
        already uses everywhere else), so the existing
        `MixedDelimiterCsvRepairExecutor` can apply it with no new
        executor needed for MIXED_DELIMITER's own use of the same
        mechanism. Performs the same decode/sniff/field-count analysis
        as `detect_all`, independently, so this detector's public
        methods each remain a single, self-contained, read-only pass
        over the file.
        """
        try:
            with open(file_path, "rb") as fh:
                raw = fh.read()
        except OSError:
            return None

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            encoding = chardet.detect(raw).get("encoding")
            if encoding is None:
                return None
            try:
                text = raw.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                return None

        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return None

        delimiter = _sniff_delimiter(text, lines)

        field_counts = [len(row) for row in csv.reader(lines, delimiter=delimiter)]
        if all(count == 1 for count in field_counts):
            return None

        offset = _find_header_offset(field_counts)
        if offset is None or offset == 0:
            return None

        garbled = _resolve_garbled_header_line(lines[offset - 1], delimiter, field_counts[offset])
        if garbled is None:
            return None
        repaired_text, observed_delimiter = garbled

        original_text = lines[offset - 1]
        return delimiter, MixedDelimiterRowRepair(
            row_number=offset,  # 1-based: `lines[offset - 1]` is line `offset`
            expected_field_count=field_counts[offset],
            actual_field_count=field_counts[offset - 1],
            observed_delimiter=observed_delimiter,
            original_text=original_text,
            repaired_text=repaired_text,
        )
