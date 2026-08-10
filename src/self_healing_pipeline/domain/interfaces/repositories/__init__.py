"""Repository-facing domain interfaces (ports)."""

from self_healing_pipeline.domain.interfaces.repositories.repair_audit_store import (
    RepairAuditStore,
)

__all__ = ["RepairAuditStore"]
