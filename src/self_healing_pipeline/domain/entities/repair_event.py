"""RepairEvent entity.

A `RepairEvent` is an immutable, append-only record of a single action
taken (or attempted) within a `RepairEpisode`. It holds no
infrastructure or repair-execution logic itself.
"""

from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams


def _utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


class RepairEvent(BaseModel):
    """Immutable record of a single repair attempt within an episode."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    event_id: UUID = Field(default_factory=uuid4)
    episode_id: UUID
    error_type: str
    handler_used: str | None = None
    applied: bool = False
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    diff: str | None = None
    prescription: CsvRepairParams | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    token_usage: int | None = Field(default=None, ge=0)
    cost: Decimal | None = Field(default=None, ge=0)
    mlflow_run_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
