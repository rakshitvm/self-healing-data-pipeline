"""Deterministic structural diff between a schema baseline and a current schema.

`ColumnDiff` is computed purely from column names/types (stdlib
`difflib` for rename similarity) — no LLM involvement. The LLM's only
role downstream is *confirming* whether a `RenameHint` genuinely
represents a rename (see `rename_resolution_subgraph`); it never
computes the raw structural facts itself.
"""

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition


class TypeChange(BaseModel):
    """A column present in both baseline and current, with a different type."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    column: str
    from_type: str
    to_type: str


class RenameHint(BaseModel):
    """A deterministic (name-similarity) candidate pairing a removed column
    with an added one — a *hint* only; not yet confirmed as a real rename."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    removed_column: str
    added_column: str
    similarity: float = Field(ge=0.0, le=1.0)


class ConfirmedRename(BaseModel):
    """A `RenameHint` the nested rename-resolution subgraph confirmed,
    with its final blended (fuzzy + LLM) confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    removed_column: str
    added_column: str
    fuzzy_similarity: float = Field(ge=0.0, le=1.0)
    llm_confidence: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class ColumnDiff(BaseModel):
    """The complete deterministic structural diff for one table."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    added: tuple[ColumnDefinition, ...] = ()
    removed: tuple[ColumnDefinition, ...] = ()
    type_changed: tuple[TypeChange, ...] = ()
    rename_hints: tuple[RenameHint, ...] = ()

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.type_changed)
