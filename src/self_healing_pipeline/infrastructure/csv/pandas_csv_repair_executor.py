"""Pandas-backed CSV repair executor.

Concrete `CsvRepairExecutor` for local CSV files. This is the only place
in the codebase allowed to import pandas for CSV repair purposes — the
domain models, `ErrorRouter`, and `CsvRepairAgent` never import it
directly, only this adapter, satisfying Dependency Inversion: they all
depend on the `CsvRepairExecutor` abstraction, not on pandas.

`execute` **never modifies the source file**: it reads `file_path` with
the given `CsvRepairParams`, and — only if that read produces a sane
multi-column result — atomically writes a *separate* repaired file at
`<source directory>/repaired/<source filename>` (creating that directory
if needed) via a temp-file-then-`os.replace` swap, so a failed write can
never leave a partially-written or corrupted output file, and the source
is never touched at any point. `verify` never writes; given a path (in
practice, the `output_path` `execute` just produced), it independently
re-reads that file with pandas' plain defaults — deliberately ignoring
the original prescription's dialect, since a freshly-written repaired
file is already canonical — and confirms it is still a sane multi-column
CSV.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd

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
