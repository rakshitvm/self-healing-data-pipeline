"""The final, ordered schema repair prescription and its confidence.

Assembled from the deterministic `ColumnDiff` plus whatever the nested
rename-resolution subgraph confirmed — never handed to `apply` until it
has passed `normalize_operation_order` and structural Pydantic
validation (every `SchemaRepairOperation` is itself validated on
construction).
"""

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.value_objects.schema_repair_operations import (
    SchemaRepairOperation,
)


class SchemaRepairPrescription(BaseModel):
    """An ordered set of repair operations with an overall confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    table: str = Field(min_length=1)
    operations: tuple[SchemaRepairOperation, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
