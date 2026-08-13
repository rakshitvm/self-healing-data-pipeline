"""Versioned schema baseline for Tier 2 schema-drift repair.

A `SchemaBaseline` is the expected, source-controlled shape of a table at
a specific version — the reference every real/current schema is diffed
against. It is a value-like, immutable snapshot (equal baselines compare
equal); a new drift means a *new* version is authored, never an in-place
edit of an existing one (append-only, matching the migration history's
own append-only guarantee).
"""

from pydantic import BaseModel, ConfigDict, Field


class ColumnDefinition(BaseModel):
    """A single column's name, logical type, and nullability."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    nullable: bool = True
    default: str | int | float | bool | None = None


class SchemaBaseline(BaseModel):
    """A versioned, immutable expected schema for one table."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    table: str = Field(min_length=1)
    version: int = Field(ge=1)
    columns: tuple[ColumnDefinition, ...] = Field(min_length=1)

    def column_names(self) -> set[str]:
        return {c.name for c in self.columns}
