"""Focused unit tests for the Tier 1 `ErrorRouter`.

Uses small stub `RepairAgent` implementations rather than any concrete
repair agent, keeping these tests aligned with the router's own
dependency boundary (it only knows about the `RepairAgent` abstraction).
"""

import pytest

from self_healing_pipeline.application.orchestration.error_router import (
    ErrorRouter,
    NoHandlerRegisteredError,
)
from self_healing_pipeline.domain.exceptions.csv_errors import (
    CsvRepairError,
    HeaderDetectionError,
    WrongDelimiterError,
    WrongEncodingError,
)
from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.repair_result import RepairResult


class _StubAgent:
    """Minimal `RepairAgent`: records the error it was asked to handle."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.handled: list[PipelineError] = []

    def handle(self, error: PipelineError) -> RepairResult:
        self.handled.append(error)
        return RepairResult(success=True, applied=True, message=f"handled by {self.name}")


class _FutureTierError(PipelineError):
    """Stand-in for a not-yet-existing Tier 2/3 error, defined only in this test."""

    def __init__(self, message: str) -> None:
        super().__init__(message, failure_class=FailureClass.UNKNOWN)


def test_exact_match_takes_priority_over_parent() -> None:
    router = ErrorRouter()
    parent_handler = _StubAgent("csv-generic")
    exact_handler = _StubAgent("wrong-delimiter")
    router.register(CsvRepairError, parent_handler)
    router.register(WrongDelimiterError, exact_handler)

    resolved = router.resolve(WrongDelimiterError("bad delimiter"))

    assert resolved is exact_handler


def test_parent_match_used_when_no_exact_registration() -> None:
    router = ErrorRouter()
    parent_handler = _StubAgent("csv-generic")
    router.register(CsvRepairError, parent_handler)

    resolved = router.resolve(WrongEncodingError("bad codec"))

    assert resolved is parent_handler


def test_closest_ancestor_wins_over_farther_ancestor() -> None:
    router = ErrorRouter()
    catch_all_handler = _StubAgent("pipeline-catch-all")
    csv_handler = _StubAgent("csv-generic")
    router.register(PipelineError, catch_all_handler)
    router.register(CsvRepairError, csv_handler)

    resolved = router.resolve(HeaderDetectionError("no header found"))

    assert resolved is csv_handler
    assert resolved is not catch_all_handler


def test_pipeline_error_catch_all_used_for_unregistered_subclass() -> None:
    router = ErrorRouter()
    catch_all_handler = _StubAgent("pipeline-catch-all")
    router.register(PipelineError, catch_all_handler)

    resolved = router.resolve(_FutureTierError("unclassified future failure"))

    assert resolved is catch_all_handler


def test_fallback_used_when_no_type_in_mro_matches() -> None:
    router = ErrorRouter()
    fallback_handler = _StubAgent("fallback")
    router.register_fallback(fallback_handler)

    resolved = router.resolve(WrongDelimiterError("bad delimiter"))

    assert resolved is fallback_handler


def test_exact_and_parent_matches_both_take_priority_over_fallback() -> None:
    router = ErrorRouter()
    fallback_handler = _StubAgent("fallback")
    exact_handler = _StubAgent("wrong-delimiter")
    router.register_fallback(fallback_handler)
    router.register(WrongDelimiterError, exact_handler)

    resolved = router.resolve(WrongDelimiterError("bad delimiter"))

    assert resolved is exact_handler


def test_registering_new_error_type_requires_only_registration() -> None:
    """Proves Open/Closed: a brand-new error type is routable via `register`
    alone, with zero changes to `ErrorRouter`."""
    router = ErrorRouter()
    future_handler = _StubAgent("future-tier-handler")
    router.register(_FutureTierError, future_handler)

    resolved = router.resolve(_FutureTierError("novel failure"))

    assert resolved is future_handler


def test_no_handler_and_no_fallback_raises_clear_error() -> None:
    router = ErrorRouter()

    with pytest.raises(NoHandlerRegisteredError) as exc_info:
        router.resolve(WrongDelimiterError("bad delimiter"))

    assert exc_info.value.error_type is WrongDelimiterError
    assert "WrongDelimiterError" in str(exc_info.value)


def test_route_delegates_to_resolved_handler_and_returns_its_result() -> None:
    router = ErrorRouter()
    handler = _StubAgent("wrong-delimiter")
    router.register(WrongDelimiterError, handler)
    error = WrongDelimiterError("bad delimiter")

    result = router.route(error)

    assert handler.handled == [error]
    assert isinstance(result, RepairResult)
    assert result.message == "handled by wrong-delimiter"
