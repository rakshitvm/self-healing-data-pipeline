"""Focused unit tests for `SchemaRepairOperation` and execution ordering."""

import pytest
from pydantic import ValidationError

from self_healing_pipeline.domain.value_objects.schema_repair_operations import (
    OperationType,
    SchemaRepairOperation,
    is_strictly_ordered,
    normalize_operation_order,
)


def test_normalize_operation_order_sorts_into_rename_cast_drop_add_default() -> None:
    add_default = SchemaRepairOperation(
        op=OperationType.ADD_DEFAULT, column="country", target_type="string", default="UNKNOWN"
    )
    drop = SchemaRepairOperation(op=OperationType.DROP, column="obsolete")
    cast = SchemaRepairOperation(op=OperationType.CAST, column="age", target_type="int64")
    rename = SchemaRepairOperation(op=OperationType.RENAME, column="customer_name", target_column="name")

    # deliberately given in a "wrong"/LLM-arbitrary order
    ordered = normalize_operation_order([add_default, drop, cast, rename])

    assert [op.op for op in ordered] == [
        OperationType.RENAME,
        OperationType.CAST,
        OperationType.DROP,
        OperationType.ADD_DEFAULT,
    ]
    assert is_strictly_ordered(ordered)


def test_normalize_operation_order_is_stable_within_the_same_type() -> None:
    cast_a = SchemaRepairOperation(op=OperationType.CAST, column="a", target_type="int64")
    cast_b = SchemaRepairOperation(op=OperationType.CAST, column="b", target_type="float64")

    ordered = normalize_operation_order([cast_b, cast_a])

    assert [op.column for op in ordered] == ["b", "a"]  # relative order preserved


def test_rename_operation_requires_target_column() -> None:
    with pytest.raises(ValidationError, match="target_column"):
        SchemaRepairOperation(op=OperationType.RENAME, column="customer_name")


def test_cast_operation_requires_target_type() -> None:
    with pytest.raises(ValidationError, match="target_type"):
        SchemaRepairOperation(op=OperationType.CAST, column="age")


def test_add_default_operation_requires_target_type() -> None:
    with pytest.raises(ValidationError, match="target_type"):
        SchemaRepairOperation(op=OperationType.ADD_DEFAULT, column="country", default="UNKNOWN")


def test_drop_operation_requires_only_column() -> None:
    op = SchemaRepairOperation(op=OperationType.DROP, column="obsolete")
    assert op.column == "obsolete"
