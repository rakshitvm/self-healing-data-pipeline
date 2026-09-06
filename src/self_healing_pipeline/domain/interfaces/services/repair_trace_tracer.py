"""Provider-agnostic repair-invocation tracing abstraction.

MLflow's tracing API (traces/spans) is a different concept from
`RepairRunTracker`'s Runs API — different ID namespace, different
lifecycle, different underlying endpoints (Ticket 013). Rather than
overloading `RepairRunTracker`'s shape, this is a small, separate,
generic port: it wraps one callable invocation in a best-effort trace,
without knowing anything about LangGraph, CSV repair, or any
application-layer type.

Tracing through `RepairTraceTracer` is strictly best-effort: the wrapped
callable is always invoked exactly once, regardless of whether trace
creation, tagging, or trace-ID retrieval succeeds or fails. It must
never be the reason a repair attempt fails, and it must never cause a
double invocation of the wrapped callable.
"""

from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable
from uuid import UUID

T = TypeVar("T")


@runtime_checkable
class RepairTraceTracer(Protocol):
    """Wraps a single callable invocation in a best-effort MLflow trace."""

    def trace_invocation(self, invoke: Callable[[], T], *, episode_id: UUID) -> tuple[T, str | None]:
        """Call `invoke()` exactly once; return `(invoke()'s result, trace_id)`.

        `trace_id` is `None` if tracing failed or was unavailable — never
        raises for tracing-specific failures. If `invoke()` itself
        raises, that exception propagates unchanged (not swallowed).
        """
        ...

    def tag_trace(self, trace_id: str, tags: dict[str, str]) -> None:
        """Best-effort: attach additional tags to an already-created trace."""
        ...

    def get_llm_usage(self, trace_id: str) -> dict[str, Any] | None:
        """Best-effort: real, already-captured LLM token usage/cost for
        `trace_id` (e.g. summed from MLflow's own OpenAI autolog spans)
        — never an invented or estimated figure. `None` if unavailable.
        """
        ...
