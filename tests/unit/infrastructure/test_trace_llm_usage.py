"""Focused unit tests for `aggregate_chat_model_usage`.

Every test patches `mlflow.get_trace` with a hand-built fake trace/span
object exposing exactly the attributes the function reads — no real
MLflow tracking server, real or local, is started or contacted. The
fake span shapes mirror real, previously-observed MLflow trace data
(`mlflow.chat.tokenUsage`, `mlflow.llm.cost`), not invented ones.
"""

from types import SimpleNamespace
from unittest.mock import patch

from self_healing_pipeline.infrastructure.mlflow.trace_llm_usage import (
    aggregate_chat_model_usage,
)


def _span(span_type: str, attributes: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(span_type=span_type, attributes=attributes)


def _trace(spans: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(spans=spans))


def test_returns_none_when_trace_is_missing() -> None:
    with patch("mlflow.get_trace", return_value=None):
        assert aggregate_chat_model_usage("tr-missing") is None


def test_returns_none_when_mlflow_raises() -> None:
    with patch("mlflow.get_trace", side_effect=RuntimeError("simulated MLflow outage")):
        assert aggregate_chat_model_usage("tr-broken") is None


def test_returns_none_when_no_chat_model_spans_exist() -> None:
    trace = _trace([_span("CHAIN", {}), _span("TOOL", {"foo": "bar"})])
    with patch("mlflow.get_trace", return_value=trace):
        assert aggregate_chat_model_usage("tr-no-llm") is None


def test_sums_a_single_chat_model_span_with_real_usage_and_cost() -> None:
    trace = _trace(
        [
            _span("CHAIN", {}),
            _span(
                "CHAT_MODEL",
                {
                    "mlflow.chat.tokenUsage": {
                        "input_tokens": 243,
                        "output_tokens": 17,
                        "total_tokens": 260,
                    },
                    "mlflow.llm.cost": {
                        "input_cost": 0.00014337,
                        "output_cost": 0.0000134,
                        "total_cost": 0.0001568,
                    },
                },
            ),
        ]
    )

    with patch("mlflow.get_trace", return_value=trace):
        result = aggregate_chat_model_usage("tr-single-call")

    assert result == {
        "prompt_tokens": 243,
        "completion_tokens": 17,
        "total_tokens": 260,
        "total_cost_usd": 0.0001568,
    }


def test_sums_multiple_chat_model_spans_across_retries() -> None:
    usage_a = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    usage_b = {"input_tokens": 120, "output_tokens": 12, "total_tokens": 132}
    trace = _trace(
        [
            _span("CHAT_MODEL", {"mlflow.chat.tokenUsage": usage_a}),
            _span("CHAIN", {}),
            _span("CHAT_MODEL", {"mlflow.chat.tokenUsage": usage_b}),
        ]
    )

    with patch("mlflow.get_trace", return_value=trace):
        result = aggregate_chat_model_usage("tr-two-calls")

    assert result is not None
    assert result["prompt_tokens"] == 220
    assert result["completion_tokens"] == 22
    assert result["total_tokens"] == 242
    assert "total_cost_usd" not in result  # no cost data was ever provided


def test_cost_omitted_not_fabricated_when_mlflow_has_no_pricing_for_the_model() -> None:
    """MLflow only attaches `mlflow.llm.cost` when it has a pricing entry
    for the model — this must never be invented when absent."""
    trace = _trace(
        [
            _span(
                "CHAT_MODEL",
                {"mlflow.chat.tokenUsage": {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55}},
            )
        ]
    )

    with patch("mlflow.get_trace", return_value=trace):
        result = aggregate_chat_model_usage("tr-no-pricing")

    assert result is not None
    assert "total_cost_usd" not in result
