"""Provider-agnostic cloud-integration abstraction (repaired-output
upload + downstream job trigger).

This is a strictly optional, best-effort *add-on* to the existing Tier 1
repair workflow — it is never part of it. A repair's own success/failure
is decided entirely by the existing `apply`/`reverify` nodes, exactly as
today; nothing here can influence that outcome. In the same spirit as
`RepairRunTracker`/`TrackingOutcome` (MLflow tracking is observability,
never the audit-of-record), a cloud-integration failure must never raise,
roll back, or corrupt the already-repaired local file — every method
reports whether it actually worked via `CloudIntegrationOutcome`, never
silently mistaken for success.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class CloudIntegrationOutcome(BaseModel):
    """Result of one best-effort "upload repaired output, then trigger a
    downstream job" attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    uploaded: bool
    cloud_path: str | None = None
    upload_error: str | None = None
    triggered: bool = False
    databricks_run_id: int | None = None
    trigger_error: str | None = None
    message: str | None = None


@runtime_checkable
class CloudIntegrationPort(Protocol):
    """Uploads a local repaired file to cloud storage and, if that
    succeeds, triggers a downstream processing job with the resulting
    cloud path."""

    def process(self, *, local_file_path: str) -> CloudIntegrationOutcome:
        """Best-effort: upload `local_file_path`, then trigger the
        configured downstream job. Never raises — any failure is
        reported on the returned outcome, and the local file at
        `local_file_path` is never modified or deleted by this call."""
        ...
