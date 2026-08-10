"""Focused unit tests for `MlflowRepairRunTracker`.

Every test uses a `MagicMock` in place of the real `mlflow.tracking.
MlflowClient`. No MLflow tracking server, real or local, is started or
contacted.
"""

from unittest.mock import MagicMock
from uuid import uuid4

from self_healing_pipeline.domain.interfaces.services.repair_run_tracker import (
    RepairRunTracker,
    TrackingOutcome,
)
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.mlflow.mlflow_repair_run_tracker import (
    MlflowRepairRunTracker,
)


def _client_with_existing_experiment(experiment_id: str = "exp-1") -> MagicMock:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id=experiment_id)
    run = MagicMock()
    run.info.run_id = "run-123"
    client.create_run.return_value = run
    return client


def test_tracker_satisfies_repair_run_tracker_protocol() -> None:
    tracker = MlflowRepairRunTracker(client=MagicMock(), experiment_name="tier1-csv-repair")

    assert isinstance(tracker, RepairRunTracker)


def test_start_run_creates_run_in_resolved_experiment_and_returns_run_id() -> None:
    client = _client_with_existing_experiment()
    tracker = MlflowRepairRunTracker(client=client, experiment_name="tier1-csv-repair")
    episode_id = uuid4()

    outcome = tracker.start_run(episode_id=episode_id, failure_class=FailureClass.WRONG_DELIMITER)

    assert outcome == TrackingOutcome(success=True, run_id="run-123")
    client.get_experiment_by_name.assert_called_once_with("tier1-csv-repair")
    client.create_experiment.assert_not_called()
    args, kwargs = client.create_run.call_args
    assert args[0] == "exp-1"
    assert kwargs["tags"] == {
        "episode_id": str(episode_id),
        "failure_class": "wrong_delimiter",
    }


def test_start_run_creates_experiment_when_absent() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "new-exp-id"
    run = MagicMock()
    run.info.run_id = "run-456"
    client.create_run.return_value = run
    tracker = MlflowRepairRunTracker(client=client, experiment_name="tier1-csv-repair")

    outcome = tracker.start_run(episode_id=uuid4(), failure_class=FailureClass.WRONG_ENCODING)

    assert outcome.success is True
    client.create_experiment.assert_called_once_with("tier1-csv-repair")
    assert client.create_run.call_args[0][0] == "new-exp-id"


def test_start_run_failure_is_best_effort_and_never_raises() -> None:
    client = MagicMock()
    client.get_experiment_by_name.side_effect = RuntimeError("mlflow server unreachable")
    tracker = MlflowRepairRunTracker(client=client, experiment_name="tier1-csv-repair")

    outcome = tracker.start_run(episode_id=uuid4(), failure_class=FailureClass.HEADER_DETECTION)

    assert outcome.success is False
    assert outcome.run_id is None
    assert outcome.error is not None
    assert "mlflow server unreachable" in outcome.error


def test_log_metrics_logs_latency_and_token_usage_when_provided() -> None:
    client = MagicMock()
    tracker = MlflowRepairRunTracker(client=client, experiment_name="tier1-csv-repair")

    outcome = tracker.log_metrics("run-123", latency_ms=250, token_usage=42)

    assert outcome == TrackingOutcome(success=True, run_id="run-123")
    client.log_metric.assert_any_call("run-123", "latency_ms", 250.0)
    client.log_metric.assert_any_call("run-123", "token_usage", 42.0)
    assert client.log_metric.call_count == 2


def test_log_metrics_omits_absent_values() -> None:
    client = MagicMock()
    tracker = MlflowRepairRunTracker(client=client, experiment_name="tier1-csv-repair")

    tracker.log_metrics("run-123", latency_ms=None, token_usage=None)

    client.log_metric.assert_not_called()


def test_log_metrics_failure_is_best_effort_and_never_raises() -> None:
    client = MagicMock()
    client.log_metric.side_effect = RuntimeError("connection reset")
    tracker = MlflowRepairRunTracker(client=client, experiment_name="tier1-csv-repair")

    outcome = tracker.log_metrics("run-123", latency_ms=250)

    assert outcome.success is False
    assert outcome.run_id == "run-123"
    assert outcome.error is not None and "connection reset" in outcome.error


def test_end_run_maps_succeeded_status_to_finished() -> None:
    client = MagicMock()
    tracker = MlflowRepairRunTracker(client=client, experiment_name="tier1-csv-repair")

    outcome = tracker.end_run("run-123", status="succeeded")

    assert outcome == TrackingOutcome(success=True, run_id="run-123")
    client.set_terminated.assert_called_once_with("run-123", status="FINISHED")


def test_end_run_maps_failed_status_to_failed() -> None:
    client = MagicMock()
    tracker = MlflowRepairRunTracker(client=client, experiment_name="tier1-csv-repair")

    tracker.end_run("run-123", status="failed")

    client.set_terminated.assert_called_once_with("run-123", status="FAILED")


def test_end_run_failure_is_best_effort_and_never_raises() -> None:
    client = MagicMock()
    client.set_terminated.side_effect = RuntimeError("mlflow store error")
    tracker = MlflowRepairRunTracker(client=client, experiment_name="tier1-csv-repair")

    outcome = tracker.end_run("run-123", status="succeeded")

    assert outcome.success is False
    assert outcome.error is not None and "mlflow store error" in outcome.error


def test_from_settings_builds_client_from_mlflow_settings() -> None:
    """`MlflowClient.__init__` eagerly resolves its backing store, so the
    constructor itself is patched here rather than actually invoked —
    this test only checks that `from_settings` wires the right arguments
    through, not that a real (even local) store can be created."""
    from unittest.mock import patch

    from self_healing_pipeline.infrastructure.config.settings import MLflowSettings

    # Constructed via validation_alias, same as settings.py's own loaders;
    # mypy only sees the field names in the generated __init__ signature.
    settings = MLflowSettings(  # type: ignore[call-arg]
        MLFLOW_TRACKING_URI="sqlite:////tmp/does-not-need-to-exist.db",
        MLFLOW_EXPERIMENT_NAME="tier1-csv-repair",
    )

    with patch(
        "self_healing_pipeline.infrastructure.mlflow.mlflow_repair_run_tracker.MlflowClient"
    ) as mock_client_cls:
        tracker = MlflowRepairRunTracker.from_settings(settings)

    mock_client_cls.assert_called_once_with(tracking_uri="sqlite:////tmp/does-not-need-to-exist.db")
    assert tracker._experiment_name == "tier1-csv-repair"  # noqa: SLF001 - white-box wiring check
