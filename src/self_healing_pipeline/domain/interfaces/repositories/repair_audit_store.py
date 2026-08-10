"""Provider-agnostic repair audit persistence abstraction.

Defines the boundary between "what happened during a repair episode" and
however that gets durably recorded — PostgreSQL, an in-memory fake for
tests, or any other store. Callers depend only on `RepairAuditStore`; no
database-specific type (psycopg2, SQLAlchemy, or otherwise) is named
here, only the existing `RepairEpisode` / `RepairEvent` domain models
(Open/Closed, Dependency Inversion).

The store only records what already happened — it never influences
routing (`ErrorRouter`'s job) or orchestration (the LangGraph workflow's
job). A persistence failure must propagate as a raised exception, never
be swallowed into a false "recorded successfully" outcome.
"""

from typing import Protocol, runtime_checkable

from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent


@runtime_checkable
class RepairAuditStore(Protocol):
    """Durably records repair episodes and the events within them."""

    def start_episode(self, episode: RepairEpisode) -> None:
        """Persist the start of a new repair episode."""
        ...

    def record_event(self, event: RepairEvent) -> None:
        """Persist a single event that occurred within an episode."""
        ...

    def complete_episode(self, episode: RepairEpisode) -> None:
        """Persist an episode's terminal status (and completion time)."""
        ...
