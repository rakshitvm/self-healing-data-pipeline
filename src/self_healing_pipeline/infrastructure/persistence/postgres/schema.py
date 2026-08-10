"""PostgreSQL schema for the Tier 1 repair audit store.

Minimal, explicit SQL — no ORM, no general migration framework. Two
tables only, directly tied to the Tier 1 audit requirement: every repair
attempt (episode) and every event within it must be recorded.
"""

from typing import Any

CREATE_REPAIR_EPISODES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS repair_episodes (
    episode_id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    table_name TEXT,
    file_path TEXT,
    failure_class TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    last_active_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    agent_last_used TEXT
);
"""

CREATE_REPAIR_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS repair_events (
    event_id UUID PRIMARY KEY,
    episode_id UUID NOT NULL REFERENCES repair_episodes(episode_id),
    node TEXT,
    status TEXT,
    error_type TEXT NOT NULL,
    handler_used TEXT,
    applied BOOLEAN NOT NULL DEFAULT FALSE,
    confidence DOUBLE PRECISION,
    diff TEXT,
    prescription JSONB,
    payload JSONB,
    latency_ms INTEGER,
    token_usage INTEGER,
    cost NUMERIC,
    mlflow_run_id TEXT,
    created_at TIMESTAMPTZ NOT NULL
);
"""

CREATE_REPAIR_EVENTS_EPISODE_ID_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_repair_events_episode_id
    ON repair_events (episode_id);
"""

CREATE_REPAIR_EPISODES_STATUS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_repair_episodes_status
    ON repair_episodes (status);
"""

SCHEMA_STATEMENTS: tuple[str, ...] = (
    CREATE_REPAIR_EPISODES_TABLE_SQL,
    CREATE_REPAIR_EVENTS_TABLE_SQL,
    CREATE_REPAIR_EVENTS_EPISODE_ID_INDEX_SQL,
    CREATE_REPAIR_EPISODES_STATUS_INDEX_SQL,
)


def initialize_schema(connection: Any) -> None:
    """Create the `repair_episodes` / `repair_events` tables if absent.

    `connection` is any psycopg2-connection-like object (real or fake);
    this function only calls `.cursor()`, `.execute()`, and `.commit()`.
    """
    with connection.cursor() as cursor:
        for statement in SCHEMA_STATEMENTS:
            cursor.execute(statement)
    connection.commit()
