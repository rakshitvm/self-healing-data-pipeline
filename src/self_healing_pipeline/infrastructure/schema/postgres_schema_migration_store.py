"""PostgreSQL mirror of the schema migration history.

Best-effort relative to the authoritative JSON history — callers should
wrap `record`/`load_episode_history` calls the same way MLflow tracking
calls are wrapped (never let a mirror failure invalidate a real
decision). This class itself does not swallow errors (matching
`PostgresRepairAuditStore`'s own honesty: a failed write must never
masquerade as success) — best-effort wrapping is the caller's job.
"""

from typing import Any
from uuid import UUID

import psycopg2
import psycopg2.extras

from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition
from self_healing_pipeline.domain.entities.schema_migration_entry import SchemaMigrationEntry
from self_healing_pipeline.domain.value_objects.column_diff import ColumnDiff
from self_healing_pipeline.domain.value_objects.schema_repair_prescription import (
    SchemaRepairPrescription,
)
from self_healing_pipeline.domain.value_objects.schema_repair_result import SchemaRepairStatus
from self_healing_pipeline.infrastructure.config.settings import DatabaseSettings
from self_healing_pipeline.infrastructure.persistence.postgres.schema_migration_schema import (
    initialize_schema_migration_schema,
)

_INSERT_SQL = """
INSERT INTO schema_migration_events (
    entry_id, episode_id, table_name, baseline_version, current_columns,
    diff, prescription, confidence, status, applied, human_approved,
    verification_message, trace_id, mlflow_run_id, token_usage, created_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

_SELECT_EPISODE_SQL = """
SELECT entry_id, episode_id, table_name, baseline_version, current_columns,
       diff, prescription, confidence, status, applied, human_approved,
       verification_message, trace_id, mlflow_run_id, token_usage, created_at
FROM schema_migration_events
WHERE episode_id = %s
ORDER BY created_at;
"""


class PostgresSchemaMigrationStore:
    """Concrete `SchemaMigrationHistoryStore` mirror backed by PostgreSQL."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @classmethod
    def from_settings(cls, settings: DatabaseSettings) -> "PostgresSchemaMigrationStore":
        """Build a store with a real `psycopg2` connection, ensuring the
        `schema_migration_events` table exists (idempotent, additive —
        never touches Tier 1's `repair_episodes`/`repair_events`)."""
        connection = psycopg2.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
        )
        initialize_schema_migration_schema(connection)
        return cls(connection)

    def record(self, entry: SchemaMigrationEntry) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                _INSERT_SQL,
                (
                    str(entry.entry_id),
                    str(entry.episode_id),
                    entry.table,
                    entry.baseline_version,
                    psycopg2.extras.Json(
                        [c.model_dump(mode="json") for c in entry.current_schema]
                    ),  # -> current_columns column (avoids the reserved `current_schema` name)
                    psycopg2.extras.Json(entry.diff.model_dump(mode="json")),
                    psycopg2.extras.Json(entry.prescription.model_dump(mode="json"))
                    if entry.prescription is not None
                    else None,
                    entry.confidence,
                    entry.status.value,
                    entry.applied,
                    entry.human_approved,
                    entry.verification_message,
                    entry.trace_id,
                    entry.mlflow_run_id,
                    psycopg2.extras.Json(entry.token_usage) if entry.token_usage else None,
                    entry.created_at,
                ),
            )
        self._connection.commit()

    def load_episode_history(self, episode_id: UUID) -> tuple[SchemaMigrationEntry, ...]:
        with self._connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(_SELECT_EPISODE_SQL, (str(episode_id),))
            rows = cursor.fetchall()

        entries = []
        for row in rows:
            entries.append(
                SchemaMigrationEntry(
                    entry_id=row["entry_id"],
                    episode_id=row["episode_id"],
                    table=row["table_name"],
                    baseline_version=row["baseline_version"],
                    current_schema=tuple(
                        ColumnDefinition.model_validate(c) for c in row["current_columns"]
                    ),
                    diff=ColumnDiff.model_validate(row["diff"]),
                    prescription=SchemaRepairPrescription.model_validate(row["prescription"])
                    if row["prescription"] is not None
                    else None,
                    confidence=row["confidence"],
                    status=SchemaRepairStatus(row["status"]),
                    applied=row["applied"],
                    human_approved=row["human_approved"],
                    verification_message=row["verification_message"],
                    trace_id=row["trace_id"],
                    mlflow_run_id=row["mlflow_run_id"],
                    token_usage=row["token_usage"],
                    created_at=row["created_at"],
                )
            )
        return tuple(entries)
