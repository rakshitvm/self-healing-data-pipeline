"""Pandas-backed schema repair execution.

The only place allowed to import pandas for schema repair purposes,
mirroring `PandasCsvRepairExecutor`'s isolation of pandas to a single
adapter. `execute` **never modifies the source file**: it reads
`file_path`, applies the given operations (rename/cast/drop/
add_default) to an in-memory frame, and — only on success — atomically
writes a *separate* repaired file at `<source directory>/repaired/
<source filename>` (creating that directory if needed), mirroring
`PandasCsvRepairExecutor` exactly. `verify` never writes; given a path
(in practice, the `output_path` `execute` just produced), it
independently re-reads that file and confirms each operation's target
end-state actually holds.
"""

import os
import tempfile
from pathlib import Path

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

_REPAIRED_SUBDIR_NAME = "repaired"


def _resolve_output_path(source_path: str) -> Path:
    """Compute the repaired-output path for `source_path`: the same
    filename, in a `repaired/` subdirectory alongside the source. Pure
    and read-only — does not touch the filesystem."""
    source = Path(source_path).resolve()
    return source.parent / _REPAIRED_SUBDIR_NAME / source.name


def _refuse_if_output_collides_with_source(
    source_path: str, output_path: Path
) -> SchemaExecutionOutcome | None:
    """Safety net (defense in depth, mirrors Tier 1's own guard): if the
    resolved output path would ever equal the resolved source path, fail
    safely rather than write anything — the source must never be
    overwritten, no matter how the output path was computed."""
    if output_path.resolve() == Path(source_path).resolve():
        return SchemaExecutionOutcome(
            success=False,
            validation_errors=["output_path_collides_with_source"],
            message=(
                f"Refusing to repair {source_path!r}: the computed output path "
                "equals the source path. The source file is never modified."
            ),
        )
    return None


def _atomic_write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    """Write `frame` to `output_path` without risking a partially-written
    or corrupted output on failure: write to a fresh temp file in the
    same directory first, then atomically swap it into place with
    `os.replace` — `output_path` is only ever touched by that final,
    atomic step, and the source file (elsewhere entirely) is never
    touched at all."""
    directory = output_path.parent
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-schema-repair-", suffix=".csv")
    os.close(fd)
    try:
        frame.to_csv(tmp_path, index=False, encoding="utf-8")
        os.replace(tmp_path, output_path)
    except Exception:
        os.remove(tmp_path)
        raise


class PandasSchemaExecutor:
    """Concrete `SchemaExecutor` that repairs local CSV files with pandas,
    never touching the source file."""

    def execute(
        self, file_path: str, operations: tuple[SchemaRepairOperation, ...]
    ) -> SchemaExecutionOutcome:
        try:
            # ASSIGN_HEADER means the file has no real header row at all —
            # a plain read would otherwise swallow the first data row as
            # a fake header (exactly the condition that made this
            # detectable in the first place, see `looks_headerless`).
            # Reading with header=None instead recovers it as real data.
            # Every other operation type is completely unaffected: same
            # call as before, byte-for-byte.
            if any(op.op is OperationType.ASSIGN_HEADER for op in operations):
                frame = pd.read_csv(file_path, header=None)
            else:
                frame = pd.read_csv(file_path)
            for operation in operations:
                frame = _apply_operation(frame, operation)
        except Exception as exc:  # noqa: BLE001 - any failure is a valid, reportable outcome
            return SchemaExecutionOutcome(
                success=False,
                validation_errors=[f"{type(exc).__name__}: {exc}"],
                message=f"Failed to apply schema repair to {file_path!r}.",
            )

        output_path = _resolve_output_path(file_path)
        collision = _refuse_if_output_collides_with_source(file_path, output_path)
        if collision is not None:
            return collision

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_csv(frame, output_path)
        except Exception as exc:  # noqa: BLE001 - a failed write is a valid, reportable outcome
            return SchemaExecutionOutcome(
                success=False,
                validation_errors=[f"{type(exc).__name__}: {exc}"],
                message=f"Applied operations to {file_path!r} but failed to write the repaired output.",
            )

        return SchemaExecutionOutcome(
            success=True,
            output_path=str(output_path),
            message=f"Applied {len(operations)} operation(s) from {file_path!r} to "
            f"{str(output_path)!r}: {list(frame.columns)}.",
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
            if operation.op in (OperationType.RENAME, OperationType.ASSIGN_HEADER):
                # After a successful repair the file always has a genuine
                # header now (assigned or renamed), read here with plain
                # defaults — so `operation.column` (the old name, or the
                # positional index for ASSIGN_HEADER) correctly no longer
                # matches anything, and `target_column` correctly does.
                # Same check serves both operation kinds unmodified.
                if operation.column in frame.columns:
                    errors.append(f"{operation.op.value}: {operation.column!r} still present")
                if operation.target_column not in frame.columns:
                    errors.append(f"{operation.op.value}: {operation.target_column!r} missing")
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
    if operation.op is OperationType.ASSIGN_HEADER:
        # `frame` was read with header=None for this operation kind (see
        # execute()), so its column labels are pandas' own default
        # integers (0, 1, 2, ...), not strings — operation.column holds
        # that same index as a string (Pydantic requires str), so it
        # must be cast back to int to actually match.
        return frame.rename(columns={int(operation.column): operation.target_column})
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
