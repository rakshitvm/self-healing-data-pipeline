"""Pandas-backed CSV repair executor.

Concrete `CsvRepairExecutor` for local CSV files. This is the only place
in the codebase allowed to import pandas for CSV repair purposes — the
domain models, `ErrorRouter`, and `CsvRepairAgent` never import it
directly, only this adapter, satisfying Dependency Inversion: they all
depend on the `CsvRepairExecutor` abstraction, not on pandas.

A single, dimension-agnostic strategy is used: attempt to read the file
with the given `CsvRepairParams`, and report whether that produced a
sane multi-column result. This works uniformly across all five Tier 1
failure dimensions because `CsvRepairParams` (delimiter, encoding,
header_row, engine) already parametrizes exactly what each dimension's
repair needs to try.
"""

import pandas as pd

from self_healing_pipeline.domain.interfaces.services.csv_repair_executor import (
    CsvExecutionOutcome,
)
from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams


class PandasCsvRepairExecutor:
    """Concrete `CsvRepairExecutor` that reads local CSV files with pandas."""

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

        return CsvExecutionOutcome(
            success=True,
            confidence=1.0,
            message=f"Read {file_path!r}: {frame.shape[0]} rows, {frame.shape[1]} columns.",
        )
