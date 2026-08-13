"""PostgreSQL schema for the Tier 2 schema-migration audit mirror.

Additive and separate from Tier 1's `repair_episodes` / `repair_events`
(`schema.py`, untouched) — schema-migration episodes are a genuinely
different aggregate (table + baseline version, not file + failure
class), so a dedicated table avoids forcing them into Tier 1's shape.
This table is a best-effort *mirror*; the source-controlled JSON
migration history (`JsonMigrationHistoryStore`) remains authoritative.
"""

from typing import Any

CREATE_SCHEMA_MIGRATION_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migration_events (
    entry_id UUID PRIMARY KEY,
    episode_id UUID NOT NULL,
    table_name TEXT NOT NULL,
    baseline_version INTEGER NOT NULL,
    current_columns JSONB NOT NULL,
    diff JSONB NOT NULL,
    prescription JSONB,
    confidence DOUBLE PRECISION,
    status TEXT NOT NULL,
    applied BOOLEAN NOT NULL DEFAULT FALSE,
    human_approved BOOLEAN,
    verification_message TEXT,
    trace_id TEXT,
    mlflow_run_id TEXT,
    token_usage JSONB,
    created_at TIMESTAMPTZ NOT NULL
);
"""

CREATE_SCHEMA_MIGRATION_EVENTS_EPISODE_ID_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_schema_migration_events_episode_id
    ON schema_migration_events (episode_id);
"""

SCHEMA_STATEMENTS: tuple[str, ...] = (
    CREATE_SCHEMA_MIGRATION_EVENTS_TABLE_SQL,
    CREATE_SCHEMA_MIGRATION_EVENTS_EPISODE_ID_INDEX_SQL,
)


def initialize_schema_migration_schema(connection: Any) -> None:
    """Create the `schema_migration_events` table if absent."""
    with connection.cursor() as cursor:
        for statement in SCHEMA_STATEMENTS:
            cursor.execute(statement)
    connection.commit()
