"""PostgreSQL implementation of `RepairAuditStore`.

Confined to infrastructure: no domain or application module imports
psycopg2. Explicit, parameterized SQL only — no ORM — against the
`repair_episodes` / `repair_events` tables defined in `schema.py`.
Persistence failures are never caught here: any `psycopg2.Error` (or
other exception) raised by the underlying connection propagates to the
caller unchanged, so a failed write can never masquerade as success.
"""

from typing import Any

import psycopg2
import psycopg2.extras

from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.infrastructure.config.settings import DatabaseSettings

_INSERT_EPISODE_SQL = """
INSERT INTO repair_episodes (
    episode_id, source, table_name, file_path, failure_class,
    status, started_at, last_active_at, is_active, agent_last_used
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

_INSERT_EVENT_SQL = """
INSERT INTO repair_events (
    event_id, episode_id, node, status, error_type, handler_used,
    applied, confidence, diff, prescription, payload,
    latency_ms, token_usage, cost, mlflow_run_id, created_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

_UPDATE_EPISODE_SQL = """
UPDATE repair_episodes
SET status = %s, last_active_at = %s, completed_at = %s,
    is_active = %s, agent_last_used = %s
WHERE episode_id = %s;
"""


class PostgresRepairAuditStore:
    """Concrete `RepairAuditStore` backed by PostgreSQL."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @classmethod
    def from_settings(cls, settings: DatabaseSettings) -> "PostgresRepairAuditStore":
        """Build a store with a real `psycopg2` connection from `settings`."""
        connection = psycopg2.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
        )
        return cls(connection)

    def start_episode(self, episode: RepairEpisode) -> None:
        context = episode.pipeline_context
        with self._connection.cursor() as cursor:
            cursor.execute(
                _INSERT_EPISODE_SQL,
                (
                    str(episode.episode_id),
                    context.source,
                    context.table_name,
                    context.file_path,
                    context.failure_class.value,
                    episode.status.value,
                    episode.started_at,
                    episode.last_active_at,
                    episode.is_active,
                    episode.agent_last_used,
                ),
            )
        self._connection.commit()

    def record_event(self, event: RepairEvent) -> None:
        prescription = (
            psycopg2.extras.Json(event.prescription.model_dump(mode="json"))
            if event.prescription is not None
            else None
        )
        payload = psycopg2.extras.Json(event.payload) if event.payload is not None else None
        with self._connection.cursor() as cursor:
            cursor.execute(
                _INSERT_EVENT_SQL,
                (
                    str(event.event_id),
                    str(event.episode_id),
                    event.node,
                    event.status,
                    event.error_type,
                    event.handler_used,
                    event.applied,
                    event.confidence,
                    event.diff,
                    prescription,
                    payload,
                    event.latency_ms,
                    event.token_usage,
                    float(event.cost) if event.cost is not None else None,
                    event.mlflow_run_id,
                    event.created_at,
                ),
            )
        self._connection.commit()

    def complete_episode(self, episode: RepairEpisode) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                _UPDATE_EPISODE_SQL,
                (
                    episode.status.value,
                    episode.last_active_at,
                    episode.completed_at,
                    episode.is_active,
                    episode.agent_last_used,
                    str(episode.episode_id),
                ),
            )
        self._connection.commit()
