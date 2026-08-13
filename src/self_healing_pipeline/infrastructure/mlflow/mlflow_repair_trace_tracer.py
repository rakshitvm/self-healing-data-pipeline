"""MLflow implementation of `RepairTraceTracer`.

Confined to infrastructure: no domain or application module imports
mlflow directly for tracing. Uses MLflow's fluent tracing functions
(`mlflow.start_span`, `mlflow.get_last_active_trace_id`,
`mlflow.set_trace_tag`) rather than a constructor-injected `MlflowClient`
— empirically verified (Ticket 013) that MLflow 3.15.1's tracing API is
fluent/global-context based, unlike the Runs API `MlflowRepairRunTracker`
uses.

Tracing is strictly best-effort and guarantees `invoke()` is called
EXACTLY ONCE regardless of whether span creation, tagging, or trace-ID
retrieval succeeds or fails — a tracing failure must never cause a
double invocation (which could double-execute a repair) or an
authoritative-result change.
"""

import sys
import time
from collections.abc import Callable
from typing import Any, TypeVar
from uuid import UUID

import mlflow

from self_healing_pipeline.infrastructure.mlflow.trace_llm_usage import aggregate_chat_model_usage

T = TypeVar("T")

_TAG_RETRY_ATTEMPTS = 3
_TAG_RETRY_DELAY_SECONDS = 0.1


class MlflowRepairTraceTracer:
    """Concrete `RepairTraceTracer` backed by MLflow's tracing API."""

    def trace_invocation(self, invoke: Callable[[], T], *, episode_id: UUID) -> tuple[T, str | None]:
        span_cm = None
        try:
            span_cm = mlflow.start_span(
                name="repair_invocation",
                span_type="CHAIN",
                attributes={"episode_id": str(episode_id)},
            )
            span_cm.__enter__()
        except Exception:  # noqa: BLE001 - span creation must never block the repair
            span_cm = None

        if span_cm is None:
            return invoke(), None

        try:
            result = invoke()
        except BaseException:
            exc_info = sys.exc_info()
            try:
                span_cm.__exit__(*exc_info)
            except Exception:  # noqa: BLE001 - must not mask the real exception
                pass
            raise
        try:
            span_cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

        trace_id: str | None = None
        try:
            trace_id = mlflow.get_last_active_trace_id()
        except Exception:  # noqa: BLE001
            trace_id = None

        if trace_id is not None:
            self.tag_trace(trace_id, {"episode_id": str(episode_id)})

        return result, trace_id

    def tag_trace(self, trace_id: str, tags: dict[str, str]) -> None:
        """Set each tag, retrying briefly on failure.

        `mlflow.set_trace_tag` mutates an in-memory, not-yet-exported
        trace directly when available; once the trace has been handed to
        the async export queue (which happens the instant the span
        exits, immediately before this is called) but before it is fully
        queryable on the backend, `set_trace_tag` falls back to an HTTP
        call that can transiently 404 against a trace that exists but
        hasn't landed yet — more likely under concurrent load. A few
        quick retries closes that window; a final failure is still
        swallowed exactly as before (best-effort, never raises).
        """
        for key, value in tags.items():
            for attempt in range(_TAG_RETRY_ATTEMPTS):
                try:
                    mlflow.set_trace_tag(trace_id, key, value)
                    break
                except Exception:  # noqa: BLE001 - tagging must never raise
                    if attempt == _TAG_RETRY_ATTEMPTS - 1:
                        break
                    time.sleep(_TAG_RETRY_DELAY_SECONDS)

    def get_llm_usage(self, trace_id: str) -> dict[str, Any] | None:
        return aggregate_chat_model_usage(trace_id)
