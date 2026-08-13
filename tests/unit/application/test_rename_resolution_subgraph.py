"""Focused unit tests for the nested rename-resolution subgraph.

Proves it is a genuine, independently-compiled `StateGraph` invoked as a
node — not an ordinary Python function call — via `get_graph(xray=True)`,
and exercises its fuzzy_match -> llm_confirm -> confidence_score pipeline
directly (no parent graph involved).
"""

from typing import Any

from self_healing_pipeline.application.orchestration.rename_resolution_subgraph import (
    build_initial_rename_resolution_state,
    build_rename_resolution_subgraph,
)
from self_healing_pipeline.domain.interfaces.services.rename_confirmation_port import (
    RenameConfirmation,
)
from self_healing_pipeline.domain.value_objects.column_diff import RenameHint


class _FakeConfirmationPort:
    """Fake `RenameConfirmationPort`: canned per-hint responses, records calls."""

    def __init__(self, responses: dict[tuple[str, str], RenameConfirmation]) -> None:
        self._responses = responses
        self.calls: list[RenameHint] = []

    def confirm(self, *, table: str, hint: RenameHint) -> RenameConfirmation:
        self.calls.append(hint)
        return self._responses.get(
            (hint.removed_column, hint.added_column),
            RenameConfirmation(confirmed=False, llm_confidence=0.0),
        )


def test_subgraph_is_genuinely_nested_not_a_function_call() -> None:
    port = _FakeConfirmationPort({})
    subgraph = build_rename_resolution_subgraph(confirmation_port=port)

    node_names = set(subgraph.get_graph().nodes.keys())
    assert {"fuzzy_match", "llm_confirm", "confidence_score"} <= node_names


def test_confirmed_rename_produces_blended_confidence() -> None:
    port = _FakeConfirmationPort(
        {("customer_name", "name"): RenameConfirmation(confirmed=True, llm_confidence=1.0)}
    )
    subgraph = build_rename_resolution_subgraph(confirmation_port=port)
    initial = build_initial_rename_resolution_state(
        table="customers", removed_columns=("customer_name",), added_columns=("name",)
    )

    result: Any = subgraph.invoke(initial)

    assert len(port.calls) == 1  # fuzzy_match found exactly one candidate to confirm
    confirmed = result["confirmed_renames"]
    assert len(confirmed) == 1
    assert confirmed[0].removed_column == "customer_name"
    assert confirmed[0].added_column == "name"
    assert 0.0 < confirmed[0].confidence <= 1.0


def test_rejected_hint_produces_no_confirmed_rename() -> None:
    port = _FakeConfirmationPort(
        {("customer_name", "name"): RenameConfirmation(confirmed=False, llm_confidence=0.9)}
    )
    subgraph = build_rename_resolution_subgraph(confirmation_port=port)
    initial = build_initial_rename_resolution_state(
        table="customers", removed_columns=("customer_name",), added_columns=("name",)
    )

    result: Any = subgraph.invoke(initial)

    assert result["confirmed_renames"] == []


def test_no_candidates_means_zero_llm_calls() -> None:
    """No-change-shaped input (nothing removed/added) must not call the LLM."""
    port = _FakeConfirmationPort({})
    subgraph = build_rename_resolution_subgraph(confirmation_port=port)
    initial = build_initial_rename_resolution_state(
        table="customers", removed_columns=(), added_columns=()
    )

    result: Any = subgraph.invoke(initial)

    assert port.calls == []
    assert result["confirmed_renames"] == []
