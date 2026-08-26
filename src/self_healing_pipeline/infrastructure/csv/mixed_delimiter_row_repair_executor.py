"""Row-level CSV repair execution for MIXED_DELIMITER.

The only place allowed to apply a `mixed_delimiter_rows` prescription.
Unlike `PandasCsvRepairExecutor`, this never reads the whole file with
one global pandas delimiter — it processes the file line by line with
the stdlib `csv` module, substitutes only the specific rows flagged in
`params.mixed_delimiter_rows` with their precomputed, already-approved
`repaired_text`, and leaves every other row exactly as it was (preserved
verbatim, not rewritten). Never touches the source file; writes a
separate repaired output, mirroring `PandasCsvRepairExecutor`'s own
never-modify-the-source contract exactly.

Row numbering contract: `row_number` (both here and on
`MixedDelimiterRowRepair`) means "the Nth non-blank line," matching
`LocalCsvFailureDetector`'s own `lines = [... if line.strip()]`
filtering exactly — blank lines are skipped when numbering, on both the
detection and execution sides, so a repair always lands on the row it
was actually computed for even when the file contains blank lines
earlier in it.
"""

import csv
import os
import tempfile
from pathlib import Path

from self_healing_pipeline.domain.interfaces.services.csv_repair_executor import (
    CsvExecutionOutcome,
)
from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams

_REPAIRED_SUBDIR_NAME = "repaired"


def _resolve_output_path(source_path: str) -> Path:
    """Compute the repaired-output path for `source_path`: the same
    filename, in a `repaired/` subdirectory alongside the source. Pure
    and read-only — does not touch the filesystem."""
    source = Path(source_path).resolve()
    return source.parent / _REPAIRED_SUBDIR_NAME / source.name


def _refuse_if_output_collides_with_source(
    source_path: str, output_path: Path
) -> CsvExecutionOutcome | None:
    """Safety net (defense in depth, mirrors `PandasCsvRepairExecutor`'s
    own guard): if the resolved output path would ever equal the
    resolved source path, fail safely rather than write anything."""
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


def _atomic_write_text(text: str, output_path: Path) -> None:
    """Write `text` to `output_path` without risking a partially-written
    or corrupted output on failure: write to a fresh temp file in the
    same directory first, then atomically swap it into place with
    `os.replace`."""
    directory = output_path.parent
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=".tmp-mixed-delimiter-repair-", suffix=".csv"
    )
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp_path, output_path)
    except Exception:
        os.remove(tmp_path)
        raise


def _read_non_blank_lines(file_path: str, encoding: str) -> list[str]:
    with open(file_path, encoding=encoding) as fh:
        raw_lines = fh.read().splitlines()
    return [line for line in raw_lines if line.strip()]


def _field_count(raw_line: str, delimiter: str) -> int:
    return len(next(csv.reader([raw_line], delimiter=delimiter)))


class MixedDelimiterCsvRepairExecutor:
    """Concrete `CsvRepairExecutor` for row-level MIXED_DELIMITER repair."""

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        try:
            lines = _read_non_blank_lines(file_path, params.encoding)
        except Exception as exc:  # noqa: BLE001 - any read failure is a valid, reportable outcome
            return CsvExecutionOutcome(
                success=False,
                validation_errors=[f"{type(exc).__name__}: {exc}"],
                message=f"Failed to read {file_path!r} for mixed-delimiter repair.",
            )

        if not lines:
            return CsvExecutionOutcome(
                success=False,
                validation_errors=["empty_file"],
                message=f"{file_path!r} is empty; nothing to repair.",
            )

        repairs_by_row = {r.row_number: r.repaired_text for r in params.mixed_delimiter_rows}
        expected_field_count = _field_count(lines[0], params.delimiter)

        repaired_lines: list[str] = []
        errors: list[str] = []
        for row_number, raw_line in enumerate(lines, start=1):
            if row_number in repairs_by_row:
                # Approved repair, applied verbatim — the exact text a
                # human already saw at approval, never re-derived here.
                repaired_lines.append(repairs_by_row[row_number])
                continue
            # Not a flagged row: must already match the expected width.
            # Preserved exactly as-is — never rewritten.
            if _field_count(raw_line, params.delimiter) != expected_field_count:
                errors.append(
                    f"row {row_number}: expected {expected_field_count} fields, "
                    "found a different count and this row was not an approved repair"
                )
                continue
            repaired_lines.append(raw_line)

        if errors:
            return CsvExecutionOutcome(
                success=False,
                validation_errors=errors,
                message=(
                    f"Mixed-delimiter repair of {file_path!r} left malformed row(s) "
                    "unresolved; refusing to write a partially-repaired output."
                ),
            )

        output_path = _resolve_output_path(file_path)
        collision = _refuse_if_output_collides_with_source(file_path, output_path)
        if collision is not None:
            return collision

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text("\n".join(repaired_lines) + "\n", output_path)
        except Exception as exc:  # noqa: BLE001 - a failed write is a valid, reportable outcome
            return CsvExecutionOutcome(
                success=False,
                validation_errors=[f"{type(exc).__name__}: {exc}"],
                message=f"Repaired {file_path!r} in memory but failed to write the output.",
            )

        return CsvExecutionOutcome(
            success=True,
            confidence=1.0,
            output_path=str(output_path),
            message=(
                f"Repaired {len(repairs_by_row)} row(s) of {file_path!r}, writing "
                f"{str(output_path)!r}: {len(repaired_lines)} rows, "
                f"{expected_field_count} columns."
            ),
        )

    def verify(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        try:
            lines = _read_non_blank_lines(file_path, params.encoding)
        except Exception as exc:  # noqa: BLE001
            return CsvExecutionOutcome(
                success=False,
                validation_errors=[f"{type(exc).__name__}: {exc}"],
                message=f"Failed to re-read {file_path!r} for verification.",
            )

        if not lines:
            return CsvExecutionOutcome(
                success=False,
                validation_errors=["empty_file"],
                message=f"{file_path!r} is empty.",
            )

        expected_field_count = _field_count(lines[0], params.delimiter)

        errors = [
            f"row {row_number}: expected {expected_field_count} fields, found {count}"
            for row_number, raw_line in enumerate(lines, start=1)
            if (count := _field_count(raw_line, params.delimiter)) != expected_field_count
        ]

        if errors:
            return CsvExecutionOutcome(
                success=False,
                validation_errors=errors,
                message=f"Verification failed for {file_path!r}: malformed row(s) remain.",
            )

        return CsvExecutionOutcome(
            success=True,
            confidence=1.0,
            message=(
                f"Verified {file_path!r}: {len(lines)} rows, {expected_field_count} "
                "columns, no malformed rows remain."
            ),
        )
