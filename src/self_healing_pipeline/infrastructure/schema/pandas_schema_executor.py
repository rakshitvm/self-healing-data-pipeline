"""Pandas-backed schema repair execution.

The only place allowed to import pandas for schema repair purposes,
mirroring `PandasCsvRepairExecutor`'s isolation of pandas to a single
adapter. `execute` mutates the file in place (rename/cast/drop/
add_default, applied in the caller-supplied order — callers are
responsible for having already normalized it via
`normalize_operation_order`). `verify` never re-applies anything; it
re-reads the (now-modified) file and independently confirms each
operation's target end-state actually holds.
"""

import pandas as pd

from self_healing_pipeline.domain.interfaces.services.schema_executor import SchemaExecutionOutcome
from self_healing_pipeline.domain.value_objects.schema_repair_operations import (
    OperationType,
    SchemaRepairOperation,
)

_PANDAS_CAST_TYPE: dict[str, str] = {
    "int64": "int64",
    "float64": "float64",
    "bool": "bool",
    "string": "string",
    "datetime": "datetime64[ns]",
}


class PandasSchemaExecutor:
    """Concrete `SchemaExecutor` that mutates local CSV files with pandas."""

    def execute(
        self, file_path: str, operations: tuple[SchemaRepairOperation, ...]
    ) -> SchemaExecutionOutcome:
        try:
            frame = pd.read_csv(file_path)
            for operation in operations:
                frame = _apply_operation(frame, operation)
            frame.to_csv(file_path, index=False)
        except Exception as exc:  # noqa: BLE001 - any failure is a valid, reportable outcome
            return SchemaExecutionOutcome(
                success=False,
                validation_errors=[f"{type(exc).__name__}: {exc}"],
                message=f"Failed to apply schema repair to {file_path!r}.",
            )
        return SchemaExecutionOutcome(
            success=True,
            message=f"Applied {len(operations)} operation(s) to {file_path!r}: "
            f"{list(frame.columns)}.",
        )

    def verify(
        self, file_path: str, operations: tuple[SchemaRepairOperation, ...]
    ) -> SchemaExecutionOutcome:
        try:
            frame = pd.read_csv(file_path)
        except Exception as exc:  # noqa: BLE001
            return SchemaExecutionOutcome(
                success=False,
                validation_errors=[f"{type(exc).__name__}: {exc}"],
                message=f"Failed to re-read {file_path!r} for verification.",
            )

        errors: list[str] = []
        for operation in operations:
            if operation.op is OperationType.RENAME:
                if operation.column in frame.columns:
                    errors.append(f"rename: {operation.column!r} still present")
                if operation.target_column not in frame.columns:
                    errors.append(f"rename: {operation.target_column!r} missing")
            elif operation.op is OperationType.CAST:
                if operation.column not in frame.columns:
                    errors.append(f"cast: {operation.column!r} missing")
                expected = _PANDAS_CAST_TYPE.get(operation.target_type or "", "")
                if operation.column in frame.columns and expected:
                    actual = str(frame[operation.column].dtype)
                    if not (actual == expected or actual.startswith(expected.split("[")[0])):
                        errors.append(
                            f"cast: {operation.column!r} is {actual!r}, expected {expected!r}"
                        )
            elif operation.op is OperationType.DROP:
                if operation.column in frame.columns:
                    errors.append(f"drop: {operation.column!r} still present")
            elif operation.op is OperationType.ADD_DEFAULT:
                if operation.column not in frame.columns:
                    errors.append(f"add_default: {operation.column!r} missing")

        if errors:
            return SchemaExecutionOutcome(
                success=False,
                validation_errors=errors,
                message=f"Verification failed for {file_path!r}.",
            )
        return SchemaExecutionOutcome(
            success=True,
            message=f"Verified {len(operations)} operation(s) against {file_path!r}: "
            f"{list(frame.columns)}.",
        )


def _apply_operation(frame: pd.DataFrame, operation: SchemaRepairOperation) -> pd.DataFrame:
    if operation.op is OperationType.RENAME:
        return frame.rename(columns={operation.column: operation.target_column})
    if operation.op is OperationType.CAST:
        target = _PANDAS_CAST_TYPE.get(operation.target_type or "", None)
        if target is None:
            raise ValueError(f"unsupported cast target type: {operation.target_type!r}")
        # `target` is always one of `_PANDAS_CAST_TYPE`'s known literal
        # values at runtime; mypy's pandas-stubs `astype` overloads only
        # accept `Literal[...]`, not a plain `str` variable.
        frame[operation.column] = frame[operation.column].astype(target)  # type: ignore[call-overload]
        return frame
    if operation.op is OperationType.DROP:
        return frame.drop(columns=[operation.column])
    if operation.op is OperationType.ADD_DEFAULT:
        frame[operation.column] = operation.default
        return frame
    raise ValueError(f"unsupported operation: {operation.op!r}")  # pragma: no cover
