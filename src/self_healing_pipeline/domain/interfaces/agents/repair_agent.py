"""Repair agent (handler) abstraction.

Defines the contract the `ErrorRouter` depends on to dispatch a
`PipelineError` to something capable of attempting a repair. The domain
and application layers depend only on this abstraction (Dependency
Inversion) — they know nothing about which concrete agents exist or how
they are implemented (LLM calls, LangGraph nodes, rule-based logic, etc).
"""

from typing import Protocol, runtime_checkable

from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.value_objects.repair_result import RepairResult


@runtime_checkable
class RepairAgent(Protocol):
    """Something that can attempt to repair a `PipelineError`.

    Any object exposing a compatible `handle` method satisfies this
    protocol structurally; no inheritance from this class is required.
    """

    def handle(self, error: PipelineError) -> RepairResult:
        """Attempt to repair the failure described by `error`."""
        ...
