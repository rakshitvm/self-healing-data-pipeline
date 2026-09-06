"""Pandas-backed CSV repair executor.

Concrete `CsvRepairExecutor` for local CSV files. The only place in the
codebase allowed to import pandas for CSV repair — domain models,
`ErrorRouter`, and `CsvRepairAgent` depend only on the
`CsvRepairExecutor` abstraction.

`execute` never modifies the source file: it reads `file_path` with the
given `CsvRepairParams`, and only if that read is sane (multi-column,
every row has the expected field count — see below) atomically writes a
separate repaired file at `<source directory>/repaired/<source
filename>` via a temp-file-then-`os.replace` swap. `verify` never
writes; it independently re-reads the given path with pandas' plain
defaults (a freshly-written repaired file is already canonical) and
confirms it's still a sane multi-column CSV with every row intact.

Row-count-vs-field-count check (`_find_row_field_count_mismatches`): a
successful pandas parse alone isn't sufficient evidence of a correct
repair. When a row has fewer raw tokens than the header, pandas' `c`/
`python` engines silently pad the missing fields with NaN rather than
raising — `frame.shape[1]` looks fine, but the row is corrupted. (A
surplus row is already caught: all three engines raise `ParserError`
for that case, so this check is deficit-only, mirroring
`MixedDelimiterRowRepair`'s own scope.) This is a second, independent
read of the raw lines via `csv.reader`, applied in both `execute`
(against the source, before writing) and `verify` (against the output).

Invisible-character stripping (`_strip_invisible_characters`): a small,
explicit set of zero-width/BOM characters (mirrors
`local_csv_failure_detector._INVISIBLE_CHARACTERS`, defined
independently here so this executor doesn't depend on the detector) is
stripped from string column names on every repair — a no-op when none
are present. `verify` independently re-confirms none remain in the
output's column names.
"""

import csv
import os
import tempfile
from pathlib import Path

import pandas as pd

from self_healing_pipeline.domain.interfaces.services.csv_repair_executor import (
    CsvExecutionOutcome,
)
from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams

_REPAIRED_SUBDIR_NAME = "repaired"

# Same known, explicit invisible/zero-width-character set as
# `local_csv_failure_detector._INVISIBLE_CHARACTERS` (BOM-as-character,
# zero-width space, zero-width non-joiner, zero-width joiner, word
# joiner) — kept as an independent constant here (not imported) since
# this executor must never depend on the detector, only on
# `CsvRepairParams` (Dependency Inversion: detection and repair are
# separate ports).
_INVISIBLE_CHARACTERS = ("﻿", "​", "‌", "‍", "⁠")


def _strip_invisible_characters(frame: pd.DataFrame) -> pd.DataFrame:
    """Return `frame` with any known invisible/zero-width character
    stripped from its column names. Applied on every repair, not just
    INVISIBLE_CHARACTERS ones — a no-op when no column name contains one.

    Non-string column names (pandas' default integer names for a
    `header=None` / NO_HEADER read) are left untouched — coercing them to
    strings would defeat NO_HEADER's "integer column names" contract.
    """
    cleaned_columns: list[object] = []
    for column in frame.columns:
        if not isinstance(column, str):
            cleaned_columns.append(column)
            continue
        name = column
        for ch in _INVISIBLE_CHARACTERS:
            name = name.replace(ch, "")
        cleaned_columns.append(name)
    frame.columns = pd.Index(cleaned_columns)
    return frame


def _find_row_field_count_mismatches(
    file_path: str,
    delimiter: str,
    header_row: int,
    expected_field_count: int,
    encoding: str = "utf-8",
) -> list[str]:
    """Independently re-read `file_path`'s raw lines (bypassing pandas'
    own NaN-padding) via `csv.reader`, and return one message per data
    row whose raw field count under `delimiter` doesn't equal
    `expected_field_count`.

    Lines up to `header_row` plus blank lines are skipped, matching
    pandas' `header=` semantics. Read failures here are treated as "no
    mismatches found" — this is a defense-in-depth check on top of a
    pandas read that already succeeded.

    `encoding` must match whatever the caller's pandas read used
    (default `"utf-8"`, correct for `verify`'s canonical output file) —
    reading a genuinely non-UTF-8 source as UTF-8 here would turn a
    correct repair into a false failure. Falls back to
    `errors="replace"` only if the given encoding itself fails to open
    the file.
    """
    try:
        with open(file_path, encoding=encoding, errors="replace") as fh:
            lines = [line for line in fh.read().splitlines() if line.strip()]
    except (LookupError, OSError):
        try:
            with open(file_path, encoding="utf-8", errors="replace") as fh:
                lines = [line for line in fh.read().splitlines() if line.strip()]
        except OSError:
            return []

    data_lines = lines[header_row + 1 :]
    mismatches: list[str] = []
    for offset, raw_line in enumerate(data_lines):
        fields = next(csv.reader([raw_line], delimiter=delimiter))
        if len(fields) != expected_field_count:
            mismatches.append(
                f"row {header_row + 2 + offset}: expected {expected_field_count} "
                f"fields, found {len(fields)}"
            )
    return mismatches


def _resolve_output_path(source_path: str) -> Path:
    """Compute the repaired-output path for `source_path`: the same
    filename, in a `repaired/` subdirectory alongside the source. Pure
    and read-only — does not touch the filesystem."""
    source = Path(source_path).resolve()
    return source.parent / _REPAIRED_SUBDIR_NAME / source.name


def _refuse_if_output_collides_with_source(
    source_path: str, output_path: Path
) -> CsvExecutionOutcome | None:
    """Safety net (defense in depth, independently unit-tested): if the
    resolved output path would ever equal the resolved source path, fail
    safely rather than write anything — the source must never be
    overwritten, no matter how the output path was computed."""
    if output_path.resolve() == Path(source_path).resolve():
        return CsvExecutionOutcome(
            success=False,
            validation_errors=["output_path_collides_with_source"],
            message=(
                f"Refusing to repair {source_path!r}: the computed output path "
                "equals the source path. The source file is never modified."
            ),
        )
    return None


class PandasCsvRepairExecutor:
    """Concrete `CsvRepairExecutor` that repairs local CSV files with pandas.

    Reads the source file; writes a separate repaired output file. The
    source is never opened for writing anywhere in this class.
    """

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        try:
            frame = pd.read_csv(
                file_path,
                sep=params.delimiter,
                encoding=params.encoding,
                header=params.header_row,
                engine=params.engine.value,
            )
        except Exception as exc:  # noqa: BLE001 - any read failure is a valid, reportable outcome
            return CsvExecutionOutcome(
                success=False,
                validation_errors=[f"{type(exc).__name__}: {exc}"],
                message=f"Failed to read {file_path!r} with the given prescription.",
            )

        if frame.shape[1] < 2:
            return CsvExecutionOutcome(
                success=False,
                validation_errors=["single_column_result"],
                message=(
                    f"Read {file_path!r} but the result still has a single column "
                    f"{list(frame.columns)!r}; the prescription did not resolve it."
                ),
            )

        frame = _strip_invisible_characters(frame)

        # header_row=None means "no header at all" (pandas header=None) —
        # every line is a data line, i.e. skip zero lines before the
        # first data row (see _find_row_field_count_mismatches' offset
        # arithmetic for why -1 achieves that).
        header_row = params.header_row if params.header_row is not None else -1
        row_mismatches = _find_row_field_count_mismatches(
            file_path, params.delimiter, header_row, frame.shape[1], encoding=params.encoding
        )
        if row_mismatches:
            return CsvExecutionOutcome(
                success=False,
                validation_errors=row_mismatches,
                message=(
                    f"Read {file_path!r} without error, but {len(row_mismatches)} "
                    "row(s) do not genuinely have the expected field count — pandas "
                    "silently padded missing fields rather than the prescription "
                    "actually resolving the file."
                ),
            )

        output_path = _resolve_output_path(file_path)
        collision = _refuse_if_output_collides_with_source(file_path, output_path)
        if collision is not None:
            return collision

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_csv(frame, output_path)
        except Exception as exc:  # noqa: BLE001 - a failed write is a valid, reportable outcome
            return CsvExecutionOutcome(
                success=False,
                validation_errors=[f"{type(exc).__name__}: {exc}"],
                message=f"Parsed {file_path!r} successfully but failed to write the repaired output.",
            )

        return CsvExecutionOutcome(
            success=True,
            confidence=1.0,
            output_path=str(output_path),
            message=(
                f"Created repaired output {str(output_path)!r} from source "
                f"{file_path!r}: {frame.shape[0]} rows, {frame.shape[1]} columns."
            ),
        )

    def verify(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        try:
            frame = pd.read_csv(file_path)
        except Exception as exc:  # noqa: BLE001 - any read failure is a valid, reportable outcome
            return CsvExecutionOutcome(
                success=False,
                validation_errors=[f"{type(exc).__name__}: {exc}"],
                message=f"Failed to re-read {file_path!r} for verification.",
            )

        if frame.shape[1] < 2:
            return CsvExecutionOutcome(
                success=False,
                validation_errors=["single_column_result"],
                message=(
                    f"Re-read {file_path!r} but the result still has a single column "
                    f"{list(frame.columns)!r}."
                ),
            )

        contaminated_columns = [
            column
            for column in frame.columns
            if isinstance(column, str) and any(ch in column for ch in _INVISIBLE_CHARACTERS)
        ]
        if contaminated_columns:
            return CsvExecutionOutcome(
                success=False,
                validation_errors=["invisible_characters_in_output_columns"],
                message=(
                    f"Re-read {file_path!r} but {len(contaminated_columns)} column "
                    "name(s) still contain invisible/zero-width characters: "
                    f"{contaminated_columns!r}."
                ),
            )

        # Plain defaults, matching the read above: a freshly-written
        # repaired file is always canonical comma-separated with the
        # header on line 0, regardless of what params.delimiter/
        # header_row were for the original source.
        row_mismatches = _find_row_field_count_mismatches(file_path, ",", 0, frame.shape[1])
        if row_mismatches:
            return CsvExecutionOutcome(
                success=False,
                validation_errors=row_mismatches,
                message=(
                    f"Re-read {file_path!r} without error, but {len(row_mismatches)} "
                    "row(s) do not genuinely have the expected field count."
                ),
            )

        return CsvExecutionOutcome(
            success=True,
            confidence=1.0,
            message=f"Verified {file_path!r}: {frame.shape[0]} rows, {frame.shape[1]} columns.",
        )


def _atomic_write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    """Write `frame` to `output_path` as canonical CSV without risking a
    partially-written or corrupted output on failure: write to a fresh
    temp file in the same directory first, then atomically swap it into
    place with `os.replace` — `output_path` is only ever touched by that
    final, atomic step, so a failure at any earlier point leaves it
    exactly as it was (and never touches anything else, in particular
    never the source file, which lives at a different path entirely).
    """
    directory = output_path.parent
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-repair-", suffix=".csv")
    os.close(fd)
    try:
        frame.to_csv(tmp_path, index=False, encoding="utf-8")
        os.replace(tmp_path, output_path)
    except Exception:
        os.remove(tmp_path)
        raise
