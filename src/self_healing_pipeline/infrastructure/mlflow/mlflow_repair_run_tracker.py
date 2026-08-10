"""MLflow implementation of `RepairRunTracker`.

Confined to infrastructure: no domain or application module imports
mlflow. Tracking is strictly best-effort — every method catches any
exception the MLflow client raises and reports it through
`TrackingOutcome` instead of propagating it, so a tracking failure can
never fail, roll back, or invalidate an otherwise successful repair.
MLflow is observability only; PostgreSQL (`RepairAuditStore`) remains
the audit-of-record.
"""

from uuid import UUID

from mlflow.tracking import MlflowClient

from self_healing_pipeline.domain.interfaces.services.repair_run_tracker import TrackingOutcome
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.config.settings import MLflowSettings

_STATUS_TO_MLFLOW_RUN_STATUS = {
    "succeeded": "FINISHED",
    "failed": "FAILED",
}
_DEFAULT_RUN_STATUS = "FINISHED"


class MlflowRepairRunTracker:
    """Concrete `RepairRunTracker` backed by MLflow."""

    def __init__(self, client: MlflowClient, experiment_name: str) -> None:
        self._client = client
        self._experiment_name = experiment_name

    @classmethod
    def from_settings(cls, settings: MLflowSettings) -> "MlflowRepairRunTracker":
        """Build a tracker with a real `MlflowClient` from `settings`."""
        client = MlflowClient(tracking_uri=settings.tracking_uri)
        return cls(client=client, experiment_name=settings.experiment_name)

    def _resolve_experiment_id(self) -> str:
        experiment = self._client.get_experiment_by_name(self._experiment_name)
        if experiment is not None:
            return str(experiment.experiment_id)
        return self._client.create_experiment(self._experiment_name)

    def start_run(self, *, episode_id: UUID, failure_class: FailureClass) -> TrackingOutcome:
        try:
            experiment_id = self._resolve_experiment_id()
            run = self._client.create_run(
                experiment_id,
                tags={"episode_id": str(episode_id), "failure_class": failure_class.value},
            )
        except Exception as exc:  # noqa: BLE001 - best-effort: never break the repair over tracking
            return TrackingOutcome(success=False, error=f"{type(exc).__name__}: {exc}")
        return TrackingOutcome(success=True, run_id=run.info.run_id)

    def log_metrics(
        self, run_id: str, *, latency_ms: int | None = None, token_usage: int | None = None
    ) -> TrackingOutcome:
        try:
            if latency_ms is not None:
                self._client.log_metric(run_id, "latency_ms", float(latency_ms))
            if token_usage is not None:
                self._client.log_metric(run_id, "token_usage", float(token_usage))
        except Exception as exc:  # noqa: BLE001
            return TrackingOutcome(success=False, run_id=run_id, error=f"{type(exc).__name__}: {exc}")
        return TrackingOutcome(success=True, run_id=run_id)

    def end_run(self, run_id: str, *, status: str) -> TrackingOutcome:
        try:
            mlflow_status = _STATUS_TO_MLFLOW_RUN_STATUS.get(status, _DEFAULT_RUN_STATUS)
            self._client.set_terminated(run_id, status=mlflow_status)
        except Exception as exc:  # noqa: BLE001
            return TrackingOutcome(success=False, run_id=run_id, error=f"{type(exc).__name__}: {exc}")
        return TrackingOutcome(success=True, run_id=run_id)
