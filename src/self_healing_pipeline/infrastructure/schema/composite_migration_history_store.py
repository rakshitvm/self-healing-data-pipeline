"""Composite migration history store: JSON authoritative, PostgreSQL best-effort.

Writes to the source-controlled JSON store first (authoritative — if
this fails, the exception propagates, matching `PostgresRepairAuditStore`'s
own "a failed write must never masquerade as success" rule for Tier 1's
audit-of-record). The PostgreSQL mirror write is then attempted
best-effort, exactly like MLflow tracking is best-effort relative to
Tier 1's PostgreSQL audit — a mirror failure never invalidates the
already-recorded, authoritative decision.
"""

from uuid import UUID

from self_healing_pipeline.domain.entities.schema_migration_entry import SchemaMigrationEntry
from self_healing_pipeline.domain.interfaces.repositories.schema_migration_history_store import (
    SchemaMigrationHistoryStore,
)
from self_healing_pipeline.infrastructure.logging.logger import get_logger


class CompositeMigrationHistoryStore:
    """`SchemaMigrationHistoryStore` combining a primary and a best-effort mirror."""

    def __init__(
        self,
        primary: SchemaMigrationHistoryStore,
        mirror: SchemaMigrationHistoryStore | None = None,
    ) -> None:
        self._primary = primary
        self._mirror = mirror

    def record(self, entry: SchemaMigrationEntry) -> None:
        self._primary.record(entry)  # authoritative: failures propagate

        if self._mirror is None:
            return
        try:
            self._mirror.record(entry)
        except Exception:  # noqa: BLE001 - mirror is best-effort, never blocks the primary
            logger = get_logger(agent="SchemaRepairAgent", node="history_mirror", table=entry.table)
            logger.warning("mirror_write_failed", episode_id=str(entry.episode_id))

    def load_episode_history(self, episode_id: UUID) -> tuple[SchemaMigrationEntry, ...]:
        return self._primary.load_episode_history(episode_id)
