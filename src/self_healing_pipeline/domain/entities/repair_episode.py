"""RepairEpisode entity.

A `RepairEpisode` is the stateful aggregate tracking a single failure's
self-healing lifecycle, from detection through resolution. It has
identity (`episode_id`) and mutates over time as repair attempts are
made; it holds no infrastructure or repair-execution logic itself.
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.value_objects.pipeline_context import PipelineContext
from self_healing_pipeline.domain.value_objects.pipeline_status import RepairEpisodeStatus


def _utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


class RepairEpisode(BaseModel):
    """Stateful aggregate tracking a single failure's repair lifecycle."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    episode_id: UUID = Field(default_factory=uuid4)
    pipeline_context: PipelineContext
    status: RepairEpisodeStatus = RepairEpisodeStatus.PENDING
    started_at: datetime = Field(default_factory=_utcnow)
    last_active_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    is_active: bool = True
    agent_last_used: str | None = None
