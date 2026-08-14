"""Pandas-backed CSV repair executor.

Concrete `CsvRepairExecutor` for local CSV files. This is the only place
in the codebase allowed to import pandas for CSV repair purposes — the
domain models, `ErrorRouter`, and `CsvRepairAgent` never import it
directly, only this adapter, satisfying Dependency Inversion: they all
depend on the `CsvRepairExecutor` abstraction, not on pandas.

`execute` mutates the file: it reads `file_path` with the given
`CsvRepairParams`, and — only if that read produces a sane multi-column
result — atomically rewrites `file_path` into canonical (comma-delimited,
UTF-8, header on row 0) CSV form via a temp-file-then-`os.replace` swap,
so a failed write can never leave the original partially overwritten or
corrupted. `verify` never writes; it independently re-reads the
(now-rewritten) file with pandas' plain defaults — deliberately ignoring
the original prescription's dialect, since after `execute` the file is
already canonical — and confirms it is still a sane multi-column CSV.
"""

import os
import tempfile

import pandas as pd

from self_healing_pipeline.domain.interfaces.services.csv_repair_executor import (
    CsvExecutionOutcome,
)
from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams


class PandasCsvRepairExecutor:
    """Concrete `CsvRepairExecutor` that reads and rewrites local CSV files with pandas."""

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

        try:
            _atomic_write_csv(frame, file_path)
        except Exception as exc:  # noqa: BLE001 - a failed write is a valid, reportable outcome
            return CsvExecutionOutcome(
                success=False,
                validation_errors=[f"{type(exc).__name__}: {exc}"],
                message=f"Parsed {file_path!r} successfully but failed to write the repair.",
            )

        return CsvExecutionOutcome(
            success=True,
            confidence=1.0,
            message=f"Repaired and rewrote {file_path!r}: {frame.shape[0]} rows, {frame.shape[1]} columns.",
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


def _atomic_write_csv(frame: pd.DataFrame, file_path: str) -> None:
    """Write `frame` to `file_path` as canonical CSV without risking a
    partially-written or corrupted original on failure: write to a fresh
    temp file in the same directory first, then atomically swap it into
    place with `os.replace` — the original is only ever touched by that
    final, atomic step, so a failure at any earlier point leaves it
    exactly as it was.
    """
    directory = os.path.dirname(file_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-repair-", suffix=".csv")
    os.close(fd)
    try:
        frame.to_csv(tmp_path, index=False, encoding="utf-8")
        os.replace(tmp_path, file_path)
    except Exception:
        os.remove(tmp_path)
        raise
