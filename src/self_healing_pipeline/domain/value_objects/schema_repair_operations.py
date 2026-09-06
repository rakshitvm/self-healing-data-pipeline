"""Structured schema repair operations and their mandatory execution order.

Every repair — however it was derived (deterministic diff, or an
LLM-confirmed rename) — is represented as one of exactly four operation
types. `normalize_operation_order` is the single place execution order is
decided: `rename -> cast -> drop -> add_default`, always, regardless of
what order operations were assembled or proposed in. The LLM never
dictates execution order.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OperationType(str, Enum):
    """The Tier 2 schema repair operation kinds.

    `ASSIGN_HEADER` is a fifth kind, structurally identical to `RENAME`
    (`column` + `target_column`) but semantically distinct: there is no
    real column to rename *from* — a header is being attached to a file
    that doesn't have one at all (`column` holds the positional index,
    as a string, that pandas assigns when reading with `header=None`).
    It only ever appears alone in an operation list — never mixed with
    the other four, which all assume named columns already exist.
    """

    RENAME = "rename"
    CAST = "cast"
    DROP = "drop"
    ADD_DEFAULT = "add_default"
    ASSIGN_HEADER = "assign_header"


_EXECUTION_ORDER: dict[OperationType, int] = {
    OperationType.ASSIGN_HEADER: -1,
    OperationType.RENAME: 0,
    OperationType.CAST: 1,
    OperationType.DROP: 2,
    OperationType.ADD_DEFAULT: 3,
}


class SchemaRepairOperation(BaseModel):
    """A single structured schema repair operation.

    Which fields are required depends on `op` (enforced below): `rename`
    needs `column`+`target_column`; `cast` needs `column`+`target_type`;
    `drop` needs only `column`; `add_default` needs `column`+`target_type`
    (+ optional `default`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    op: OperationType
    column: str = Field(min_length=1)
    target_column: str | None = None
    target_type: str | None = None
    default: str | int | float | bool | None = None

    @model_validator(mode="after")
    def _validate_required_fields_for_op(self) -> "SchemaRepairOperation":
        if self.op in (OperationType.RENAME, OperationType.ASSIGN_HEADER) and not self.target_column:
            raise ValueError(f"{self.op.value} operation requires target_column")
        if self.op is OperationType.CAST and not self.target_type:
            raise ValueError("cast operation requires target_type")
        if self.op is OperationType.ADD_DEFAULT and not self.target_type:
            raise ValueError("add_default operation requires target_type")
        return self


def normalize_operation_order(
    operations: list[SchemaRepairOperation],
) -> list[SchemaRepairOperation]:
    """Return `operations` re-ordered into the mandatory execution order.

    Stable sort: relative order within the same operation type is
    preserved. This is the sole authority on execution order — callers
    (including anything that assembles operations from an LLM-influenced
    source) must always pass their result through this function before
    execution.
    """
    return sorted(operations, key=lambda operation: _EXECUTION_ORDER[operation.op])


def is_strictly_ordered(operations: list[SchemaRepairOperation]) -> bool:
    """True iff `operations` are already in the mandatory execution order."""
    ranks = [_EXECUTION_ORDER[operation.op] for operation in operations]
    return ranks == sorted(ranks)
