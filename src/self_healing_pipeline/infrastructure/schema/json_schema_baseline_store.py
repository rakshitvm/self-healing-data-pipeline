"""Source-controlled, versioned JSON schema baseline storage.

Baselines live at `<root>/<table>.json`, one file per table, hand-authored
and versioned via git — deliberately read-only from the application's
perspective (this store never writes): a baseline is a deliberate,
human-authored reference, not something the tool should silently invent.
"""

import json
from pathlib import Path

from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition, SchemaBaseline


class JsonSchemaBaselineStore:
    """Concrete `SchemaBaselineStore` reading `<root>/<table>.json` files."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def load_latest(self, table: str) -> SchemaBaseline | None:
        path = self._root / f"{table}.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SchemaBaseline(
            table=raw["table"],
            version=raw["version"],
            columns=tuple(
                ColumnDefinition(
                    name=c["name"],
                    type=c["type"],
                    nullable=c.get("nullable", True),
                    default=c.get("default"),
                )
                for c in raw["columns"]
            ),
        )
