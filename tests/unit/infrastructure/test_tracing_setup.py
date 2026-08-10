"""Focused unit tests for `enable_tracing` / `flush_traces`.

Mocked — no real MLflow tracking server is contacted. The real,
empirically-verified bounded behavior against a genuinely unreachable
server is covered separately by the real integration test suite
(`tests/integration/test_mlflow_tracing_smoke.py`), since that is a
timing property that a mock cannot meaningfully prove.
"""

import os
from unittest.mock import patch

from self_healing_pipeline.infrastructure.config.settings import MLflowSettings
from self_healing_pipeline.infrastructure.mlflow.tracing_setup import enable_tracing, flush_traces


def _settings() -> MLflowSettings:
    return MLflowSettings(  # type: ignore[call-arg]
        MLFLOW_TRACKING_URI="http://localhost:5000",
        MLFLOW_EXPERIMENT_NAME="tier1-csv-repair",
    )


def test_enable_tracing_returns_true_on_success() -> None:
    with (
        patch("mlflow.set_tracking_uri") as mock_set_uri,
        patch("mlflow.set_experiment") as mock_set_experiment,
        patch("mlflow.langchain.autolog") as mock_langchain_autolog,
        patch("mlflow.openai.autolog") as mock_openai_autolog,
    ):
        result = enable_tracing(_settings())

    assert result is True
    mock_set_uri.assert_called_once_with("http://localhost:5000")
    mock_set_experiment.assert_called_once_with("tier1-csv-repair")
    mock_langchain_autolog.assert_called_once()
    mock_openai_autolog.assert_called_once()


def test_enable_tracing_returns_false_on_failure_without_raising() -> None:
    with patch("mlflow.set_tracking_uri", side_effect=RuntimeError("boom")):
        result = enable_tracing(_settings())

    assert result is False


def test_enable_tracing_sets_bounded_http_timeout_and_retries() -> None:
    saved = {
        k: os.environ.pop(k, None)
        for k in ("MLFLOW_HTTP_REQUEST_TIMEOUT", "MLFLOW_HTTP_REQUEST_MAX_RETRIES")
    }
    try:
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
            patch("mlflow.langchain.autolog"),
            patch("mlflow.openai.autolog"),
        ):
            enable_tracing(_settings())

        assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "5"
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "1"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_enable_tracing_does_not_override_operator_configured_env_vars() -> None:
    saved = {
        k: os.environ.get(k) for k in ("MLFLOW_HTTP_REQUEST_TIMEOUT", "MLFLOW_HTTP_REQUEST_MAX_RETRIES")
    }
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "42"
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "9"
    try:
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
            patch("mlflow.langchain.autolog"),
            patch("mlflow.openai.autolog"),
        ):
            enable_tracing(_settings())

        assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "42"
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "9"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_flush_traces_calls_flush_trace_async_logging() -> None:
    with patch("mlflow.flush_trace_async_logging") as mock_flush:
        flush_traces(timeout_seconds=2.0)

    mock_flush.assert_called_once_with(terminate=True)


def test_flush_traces_is_best_effort_and_never_raises() -> None:
    with patch("mlflow.flush_trace_async_logging", side_effect=RuntimeError("boom")):
        flush_traces(timeout_seconds=1.0)
    # no exception raised => test passes


def test_flush_traces_respects_its_timeout_bound() -> None:
    import time

    def _slow_flush(*, terminate: bool) -> None:
        time.sleep(30)

    with patch("mlflow.flush_trace_async_logging", side_effect=_slow_flush):
        start = time.perf_counter()
        flush_traces(timeout_seconds=0.5)
        elapsed = time.perf_counter() - start

    assert elapsed < 5.0  # must return promptly, not wait for the slow call


def test_flush_traces_uses_a_daemon_thread() -> None:
    with patch("mlflow.flush_trace_async_logging"):
        with patch(
            "self_healing_pipeline.infrastructure.mlflow.tracing_setup.threading.Thread"
        ) as mock_thread_cls:
            mock_thread = mock_thread_cls.return_value
            flush_traces(timeout_seconds=1.0)

    assert mock_thread_cls.call_args.kwargs["daemon"] is True
    mock_thread.start.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=1.0)
