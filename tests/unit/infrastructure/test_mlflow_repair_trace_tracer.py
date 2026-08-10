"""Focused unit tests for `MlflowRepairTraceTracer`.

Every test patches `mlflow.start_span` / `mlflow.get_last_active_trace_id`
/ `mlflow.set_trace_tag` directly. No MLflow tracking server, real or
local, is started or contacted.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

from self_healing_pipeline.domain.interfaces.services.repair_trace_tracer import RepairTraceTracer
from self_healing_pipeline.infrastructure.mlflow.mlflow_repair_trace_tracer import (
    MlflowRepairTraceTracer,
)

_MODULE = "self_healing_pipeline.infrastructure.mlflow.mlflow_repair_trace_tracer.mlflow"


def test_tracer_satisfies_repair_trace_tracer_protocol() -> None:
    assert isinstance(MlflowRepairTraceTracer(), RepairTraceTracer)


def test_trace_invocation_calls_invoke_exactly_once_on_success() -> None:
    calls = []

    def invoke() -> str:
        calls.append(1)
        return "repair-result"

    span_cm = MagicMock()
    with patch(f"{_MODULE}.start_span", return_value=span_cm) as mock_start_span:
        with patch(f"{_MODULE}.get_last_active_trace_id", return_value="tr-abc123"):
            with patch(f"{_MODULE}.set_trace_tag") as mock_set_tag:
                result, trace_id = MlflowRepairTraceTracer().trace_invocation(
                    invoke, episode_id=uuid4()
                )

    assert result == "repair-result"
    assert trace_id == "tr-abc123"
    assert len(calls) == 1
    mock_start_span.assert_called_once()
    assert mock_start_span.call_args.kwargs["name"] == "repair_invocation"
    assert mock_start_span.call_args.kwargs["span_type"] == "CHAIN"
    span_cm.__enter__.assert_called_once()
    span_cm.__exit__.assert_called_once_with(None, None, None)
    mock_set_tag.assert_called_once()


def test_trace_invocation_calls_invoke_exactly_once_when_span_creation_fails() -> None:
    calls = []

    def invoke() -> str:
        calls.append(1)
        return "repair-result"

    with patch(f"{_MODULE}.start_span", side_effect=RuntimeError("mlflow unreachable")):
        result, trace_id = MlflowRepairTraceTracer().trace_invocation(invoke, episode_id=uuid4())

    assert result == "repair-result"
    assert trace_id is None
    assert len(calls) == 1


def test_trace_invocation_propagates_invoke_exception_without_double_calling() -> None:
    calls = []

    def invoke() -> str:
        calls.append(1)
        raise ValueError("node failed")

    span_cm = MagicMock()
    with patch(f"{_MODULE}.start_span", return_value=span_cm):
        try:
            MlflowRepairTraceTracer().trace_invocation(invoke, episode_id=uuid4())
            raise AssertionError("expected ValueError to propagate")
        except ValueError as exc:
            assert str(exc) == "node failed"

    assert len(calls) == 1
    span_cm.__exit__.assert_called_once()
    exit_args = span_cm.__exit__.call_args.args
    assert exit_args[0] is ValueError


def test_trace_invocation_survives_get_trace_id_failure() -> None:
    span_cm = MagicMock()
    with patch(f"{_MODULE}.start_span", return_value=span_cm):
        with patch(f"{_MODULE}.get_last_active_trace_id", side_effect=RuntimeError("boom")):
            result, trace_id = MlflowRepairTraceTracer().trace_invocation(
                lambda: "ok", episode_id=uuid4()
            )

    assert result == "ok"
    assert trace_id is None


def test_trace_invocation_survives_span_exit_failure() -> None:
    span_cm = MagicMock()
    span_cm.__exit__.side_effect = RuntimeError("exit failed")
    with patch(f"{_MODULE}.start_span", return_value=span_cm):
        with patch(f"{_MODULE}.get_last_active_trace_id", return_value=None):
            result, trace_id = MlflowRepairTraceTracer().trace_invocation(
                lambda: "ok", episode_id=uuid4()
            )

    assert result == "ok"
    assert trace_id is None


def test_tag_trace_calls_set_trace_tag_per_tag() -> None:
    with patch(f"{_MODULE}.set_trace_tag") as mock_set_tag:
        MlflowRepairTraceTracer().tag_trace("tr-abc", {"failure_class": "wrong_delimiter", "x": "y"})

    assert mock_set_tag.call_count == 2
    mock_set_tag.assert_any_call("tr-abc", "failure_class", "wrong_delimiter")
    mock_set_tag.assert_any_call("tr-abc", "x", "y")


def test_tag_trace_is_best_effort_and_never_raises() -> None:
    with patch(f"{_MODULE}.set_trace_tag", side_effect=RuntimeError("mlflow unreachable")):
        MlflowRepairTraceTracer().tag_trace("tr-abc", {"failure_class": "wrong_delimiter"})
    # no exception raised => test passes
