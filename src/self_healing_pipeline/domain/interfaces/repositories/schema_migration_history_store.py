"""Append-only schema migration history abstraction.

Backs both Tier 2's mandatory source-controlled JSON history and (as a
best-effort mirror, exactly like MLflow tracking is best-effort relative
to the Tier 1 PostgreSQL audit) a PostgreSQL copy. Never overwrites a
prior entry — every attempt, applied or rejected, is a new record.
"""

from typing import Protocol, runtime_checkable
from uuid import UUID

from self_healing_pipeline.domain.entities.schema_migration_entry import SchemaMigrationEntry


@runtime_checkable
class SchemaMigrationHistoryStore(Protocol):
    """Records and recalls append-only schema migration history."""

    def record(self, entry: SchemaMigrationEntry) -> None:
        """Append `entry`. Must never overwrite or mutate a prior entry."""
        ...

    def load_episode_history(self, episode_id: UUID) -> tuple[SchemaMigrationEntry, ...]:
        """Return every entry previously recorded for `episode_id`, in order."""
        ...
