"""One-time PostgreSQL schema initialization for the Tier 1 audit store.

Reuses the existing `initialize_schema` (Ticket 009) — no CREATE TABLE
statements are duplicated here or in Docker Compose — against a
connection built from the existing `DatabaseSettings`. Run this once
after `docker compose up` brings PostgreSQL up, before the first repair
attempt (or at any time; `initialize_schema` uses `CREATE TABLE IF NOT
EXISTS`, so re-running it is harmless).

Usage:
    python scripts/init_db.py
"""

import sys

import psycopg2

from self_healing_pipeline.infrastructure.config.settings import get_settings
from self_healing_pipeline.infrastructure.persistence.postgres.schema import initialize_schema


def main() -> int:
    settings = get_settings().database
    connection = psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )
    try:
        initialize_schema(connection)
    finally:
        connection.close()

    print(
        f"Initialized repair_episodes / repair_events on "
        f"{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
