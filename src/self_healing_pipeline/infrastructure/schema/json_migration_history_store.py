"""Source-controlled, append-only JSON migration history (Tier 2 requirement 6).

One file per entry, named by `entry_id`, under `<root>/<episode_id>/` —
writing a new file per attempt (rather than read-modify-write on a
shared file) makes "append-only, never overwrite a prior decision" a
filesystem-level guarantee, not just a convention.
"""

import json
from pathlib import Path
from uuid import UUID

from self_healing_pipeline.domain.entities.schema_migration_entry import SchemaMigrationEntry


class JsonMigrationHistoryStore:
    """Concrete `SchemaMigrationHistoryStore` writing one JSON file per entry."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def record(self, entry: SchemaMigrationEntry) -> None:
        episode_dir = self._root / str(entry.episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)
        path = episode_dir / f"{entry.created_at.strftime('%Y%m%dT%H%M%S%f')}_{entry.entry_id}.json"
        path.write_text(
            json.dumps(entry.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load_episode_history(self, episode_id: UUID) -> tuple[SchemaMigrationEntry, ...]:
        episode_dir = self._root / str(episode_id)
        if not episode_dir.exists():
            return ()
        entries = []
        for path in sorted(episode_dir.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            entries.append(SchemaMigrationEntry.model_validate(raw))
        return tuple(entries)
