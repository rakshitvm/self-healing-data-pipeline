"""Focused unit tests for `PostgresRepairAuditStore` and its schema.

Every test uses fake, in-memory connection/cursor doubles that mimic the
psycopg2 API surface this store relies on (`.cursor()` as a context
manager, `.execute(sql, params)`, `.commit()`). No real PostgreSQL server
is started or contacted.
"""

from typing import Any

import psycopg2.extras
import pytest

from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.domain.interfaces.repositories.repair_audit_store import (
    RepairAuditStore,
)
from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.pipeline_context import PipelineContext
from self_healing_pipeline.domain.value_objects.pipeline_status import RepairEpisodeStatus
from self_healing_pipeline.infrastructure.persistence.postgres.repair_audit_store import (
    PostgresRepairAuditStore,
)
from self_healing_pipeline.infrastructure.persistence.postgres.schema import (
    CREATE_REPAIR_EPISODES_TABLE_SQL,
    CREATE_REPAIR_EVENTS_TABLE_SQL,
    SCHEMA_STATEMENTS,
    initialize_schema,
)


class _FakeCursor:
    """Records every `.execute()` call; supports the `with` protocol."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.should_raise: Exception | None = None

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        if self.should_raise is not None:
            raise self.should_raise
        self.executed.append((sql, params))


class _FakeConnection:
    """Records `.commit()` calls; hands out a single `_FakeCursor`."""

    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()
        self.commit_count = 0

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commit_count += 1


def _episode() -> RepairEpisode:
    context = PipelineContext(
        source="local", file_path="/tmp/wrong.csv", failure_class=FailureClass.WRONG_DELIMITER
    )
    return RepairEpisode(pipeline_context=context)


def test_postgres_store_satisfies_repair_audit_store_protocol() -> None:
    store = PostgresRepairAuditStore(connection=_FakeConnection())

    assert isinstance(store, RepairAuditStore)


def test_episode_creation_executes_insert_and_commits() -> None:
    conn = _FakeConnection()
    store = PostgresRepairAuditStore(conn)
    episode = _episode()

    store.start_episode(episode)

    assert conn.commit_count == 1
    sql, params = conn.cursor_obj.executed[0]
    assert "INSERT INTO repair_episodes" in sql
    assert params[0] == str(episode.episode_id)
    assert params[4] == FailureClass.WRONG_DELIMITER.value
    assert params[5] == RepairEpisodeStatus.PENDING.value


def test_event_insertion_executes_insert_and_commits() -> None:
    conn = _FakeConnection()
    store = PostgresRepairAuditStore(conn)
    episode = _episode()
    event = RepairEvent(
        episode_id=episode.episode_id,
        node="apply",
        status="applied",
        error_type=FailureClass.WRONG_DELIMITER.value,
        applied=True,
        confidence=0.9,
        prescription=CsvRepairParams(delimiter=";", encoding="utf-8"),
        payload={"message": "ok", "validation_errors": []},
    )

    store.record_event(event)

    assert conn.commit_count == 1
    sql, params = conn.cursor_obj.executed[0]
    assert "INSERT INTO repair_events" in sql
    assert params[0] == str(event.event_id)


def test_event_belongs_to_the_correct_episode() -> None:
    conn = _FakeConnection()
    store = PostgresRepairAuditStore(conn)
    episode = _episode()
    event = RepairEvent(
        episode_id=episode.episode_id, node="apply", error_type=FailureClass.WRONG_DELIMITER.value
    )

    store.record_event(event)

    _, params = conn.cursor_obj.executed[0]
    assert params[1] == str(episode.episode_id)


def test_episode_completion_executes_update_and_commits() -> None:
    conn = _FakeConnection()
    store = PostgresRepairAuditStore(conn)
    episode = _episode()
    episode.status = RepairEpisodeStatus.SUCCEEDED
    episode.is_active = False

    store.complete_episode(episode)

    assert conn.commit_count == 1
    sql, params = conn.cursor_obj.executed[0]
    assert "UPDATE repair_episodes" in sql
    assert params[0] == RepairEpisodeStatus.SUCCEEDED.value
    assert params[-1] == str(episode.episode_id)


def test_structured_payload_is_serialized_as_json_safe() -> None:
    conn = _FakeConnection()
    store = PostgresRepairAuditStore(conn)
    episode = _episode()
    event = RepairEvent(
        episode_id=episode.episode_id,
        error_type=FailureClass.WRONG_DELIMITER.value,
        prescription=CsvRepairParams(delimiter=";", encoding="utf-8"),
        payload={"validation_errors": ["bad delimiter"], "message": "still broken"},
    )

    store.record_event(event)

    _, params = conn.cursor_obj.executed[0]
    prescription_param, payload_param = params[9], params[10]
    assert isinstance(prescription_param, psycopg2.extras.Json)
    assert prescription_param.adapted == {
        "delimiter": ";",
        "encoding": "utf-8",
        "header_row": 0,
        "engine": "python",
        "mixed_delimiter_rows": [],
    }
    assert isinstance(payload_param, psycopg2.extras.Json)
    assert payload_param.adapted == {
        "validation_errors": ["bad delimiter"],
        "message": "still broken",
    }


def test_persistence_failure_is_not_swallowed() -> None:
    conn = _FakeConnection()
    conn.cursor_obj.should_raise = RuntimeError("simulated connection failure")
    store = PostgresRepairAuditStore(conn)

    with pytest.raises(RuntimeError, match="simulated connection failure"):
        store.start_episode(_episode())

    assert conn.commit_count == 0  # no false "success" via a stray commit


def test_credentials_are_never_hardcoded_in_the_store_module() -> None:
    import self_healing_pipeline.infrastructure.persistence.postgres.repair_audit_store as module

    source_path = module.__file__
    assert source_path is not None
    with open(source_path, encoding="utf-8") as fh:
        content = fh.read()

    # The only credential reference must flow through injected settings
    # (`settings.db_password`), never a literal string value.
    assert 'password="' not in content
    assert "password='" not in content
    assert "settings.db_password" in content


def test_schema_contains_repair_episodes_table() -> None:
    assert "repair_episodes" in CREATE_REPAIR_EPISODES_TABLE_SQL
    assert "CREATE TABLE" in CREATE_REPAIR_EPISODES_TABLE_SQL
    assert "episode_id UUID PRIMARY KEY" in CREATE_REPAIR_EPISODES_TABLE_SQL


def test_schema_contains_repair_events_table() -> None:
    assert "repair_events" in CREATE_REPAIR_EVENTS_TABLE_SQL
    assert "CREATE TABLE" in CREATE_REPAIR_EVENTS_TABLE_SQL
    assert "event_id UUID PRIMARY KEY" in CREATE_REPAIR_EVENTS_TABLE_SQL


def test_schema_represents_foreign_key_from_events_to_episodes() -> None:
    assert "REFERENCES repair_episodes(episode_id)" in CREATE_REPAIR_EVENTS_TABLE_SQL


def test_initialize_schema_executes_every_statement_and_commits() -> None:
    conn = _FakeConnection()

    initialize_schema(conn)

    assert conn.commit_count == 1
    executed_sql = [sql for sql, _ in conn.cursor_obj.executed]
    assert executed_sql == list(SCHEMA_STATEMENTS)
