"""Focused unit tests for `PandasSchemaExecutor`.

No dedicated executor-only test file existed for Tier 2 before this —
its behavior was previously only exercised indirectly through
`test_schema_repair_workflow.py`. This file adds direct, executor-level
coverage for `ASSIGN_HEADER`: the one operation kind that changes how
the source file itself is read (`header=None` instead of the default),
so it deserves tests that aren't entangled with the rest of the
workflow. Every test writes a real temporary CSV file and drives the
real executor — no mocking of pandas or file I/O.
"""

from pathlib import Path

import pandas as pd

from self_healing_pipeline.domain.value_objects.schema_repair_operations import (
    OperationType,
    SchemaRepairOperation,
)
from self_healing_pipeline.infrastructure.schema.pandas_schema_executor import PandasSchemaExecutor


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


ASSIGN_HEADER_OPS = (
    SchemaRepairOperation(op=OperationType.ASSIGN_HEADER, column="0", target_column="id"),
    SchemaRepairOperation(op=OperationType.ASSIGN_HEADER, column="1", target_column="name"),
    SchemaRepairOperation(op=OperationType.ASSIGN_HEADER, column="2", target_column="age"),
)


def test_execute_reads_header_none_and_recovers_the_first_row(tmp_path: Path) -> None:
    """The core behavior: with header=0 (the default), '1,Alice,30' would
    be silently swallowed as a fake header, losing that row's data.
    ASSIGN_HEADER reads with header=None instead, so it's recovered as
    a real data row."""
    file_path = _write(tmp_path, "headerless.csv", "1,Alice,30\n2,Bob,25\n3,Charlie,35\n")

    outcome = PandasSchemaExecutor().execute(file_path, ASSIGN_HEADER_OPS)

    assert outcome.success is True
    assert outcome.output_path is not None
    frame = pd.read_csv(outcome.output_path)
    assert list(frame.columns) == ["id", "name", "age"]
    assert len(frame) == 3
    assert frame.iloc[0].to_dict() == {"id": 1, "name": "Alice", "age": 30}


def test_source_is_never_modified(tmp_path: Path) -> None:
    original = "1,Alice,30\n2,Bob,25\n"
    file_path = _write(tmp_path, "headerless.csv", original)

    outcome = PandasSchemaExecutor().execute(file_path, ASSIGN_HEADER_OPS[:2])

    assert outcome.success is True
    assert Path(file_path).read_text(encoding="utf-8") == original


def test_execute_with_only_non_assign_header_ops_is_completely_unaffected(
    tmp_path: Path,
) -> None:
    """Regression: a normal rename operation list (no ASSIGN_HEADER at
    all) must read exactly as before (plain header=0 default) — proving
    the new conditional read logic doesn't change existing behavior."""
    file_path = _write(tmp_path, "customers.csv", "id,customer_name,age\n1,Alice,30\n")
    ops = (
        SchemaRepairOperation(
            op=OperationType.RENAME, column="customer_name", target_column="name"
        ),
    )

    outcome = PandasSchemaExecutor().execute(file_path, ops)

    assert outcome.success is True
    assert outcome.output_path is not None
    frame = pd.read_csv(outcome.output_path)
    assert list(frame.columns) == ["id", "name", "age"]
    assert len(frame) == 1  # header row correctly consumed as a header, not data


def test_verify_confirms_a_successful_assign_header_repair(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "headerless.csv", "1,Alice,30\n2,Bob,25\n")
    executor = PandasSchemaExecutor()
    outcome = executor.execute(file_path, ASSIGN_HEADER_OPS)
    assert outcome.output_path is not None

    verify_outcome = executor.verify(outcome.output_path, ASSIGN_HEADER_OPS)

    assert verify_outcome.success is True


def test_verify_fails_if_the_expected_header_names_are_not_present(tmp_path: Path) -> None:
    """Direct proof that verify's RENAME check now also correctly covers
    ASSIGN_HEADER: given an output that never actually got the baseline's
    column names, verify must fail, not silently pass."""
    still_headerless_output = _write(tmp_path, "still_bad.csv", "1,Alice,30\n2,Bob,25\n")

    outcome = PandasSchemaExecutor().verify(still_headerless_output, ASSIGN_HEADER_OPS)

    assert outcome.success is False
    assert any("id" in e for e in outcome.validation_errors)
