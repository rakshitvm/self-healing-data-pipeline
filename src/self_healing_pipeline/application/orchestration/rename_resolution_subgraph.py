"""Nested LangGraph subgraph: fuzzy_match -> llm_confirm -> confidence_score.

A genuine, independently-compiled `StateGraph`, invoked as a single node
of the parent `schema_repair_workflow` graph (LangGraph natively supports
passing a compiled `StateGraph` directly as another graph's node — this
is not simulated with an ordinary function call; `parent.get_graph(xray=True)`
shows this subgraph's own internal nodes nested under the parent node).

Self-contained: given only the raw removed/added column names for a
table, it independently re-derives fuzzy-match candidates (reusing
`compute_rename_hints` — the same deterministic function `schema_diff`
uses, not a second implementation), asks the injected
`RenameConfirmationPort` (the only LLM call in schema repair) to confirm
each candidate, and blends fuzzy + LLM confidence into a final score.
Hints the LLM does not confirm are dropped, not weakened.
"""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from self_healing_pipeline.application.orchestration.schema_diff import compute_rename_hints
from self_healing_pipeline.domain.interfaces.services.rename_confirmation_port import (
    RenameConfirmationPort,
)
from self_healing_pipeline.domain.value_objects.column_diff import ConfirmedRename, RenameHint

_FUZZY_WEIGHT = 0.4
_LLM_WEIGHT = 0.6


class RenameResolutionState(TypedDict):
    """State threaded through the nested rename-resolution subgraph."""

    table: str
    removed_columns: tuple[str, ...]
    added_columns: tuple[str, ...]
    fuzzy_candidates: list[RenameHint]
    confirmed_renames: list[ConfirmedRename]


def build_initial_rename_resolution_state(
    *, table: str, removed_columns: tuple[str, ...], added_columns: tuple[str, ...]
) -> RenameResolutionState:
    """Build the initial state for a fresh subgraph run."""
    return RenameResolutionState(
        table=table,
        removed_columns=removed_columns,
        added_columns=added_columns,
        fuzzy_candidates=[],
        confirmed_renames=[],
    )


def _fuzzy_match(state: RenameResolutionState) -> dict[str, Any]:
    """Deterministic candidate generation — no LLM call."""
    candidates = compute_rename_hints(
        removed=state["removed_columns"], added=state["added_columns"]
    )
    return {"fuzzy_candidates": list(candidates)}


def _make_llm_confirm_node(port: RenameConfirmationPort) -> Any:
    def llm_confirm(state: RenameResolutionState) -> dict[str, Any]:
        confirmed: list[ConfirmedRename] = []
        for hint in state["fuzzy_candidates"]:
            confirmation = port.confirm(table=state["table"], hint=hint)
            if not confirmation.confirmed:
                continue
            final_confidence = (
                _FUZZY_WEIGHT * hint.similarity + _LLM_WEIGHT * confirmation.llm_confidence
            )
            confirmed.append(
                ConfirmedRename(
                    removed_column=hint.removed_column,
                    added_column=hint.added_column,
                    fuzzy_similarity=hint.similarity,
                    llm_confidence=confirmation.llm_confidence,
                    confidence=final_confidence,
                )
            )
        return {"confirmed_renames": confirmed}

    return llm_confirm


def _confidence_score(state: RenameResolutionState) -> dict[str, Any]:
    """Terminal node: the blended confidence was already computed in
    `llm_confirm`; this node exists as its own step (per the required
    `fuzzy_match -> LLM_confirm -> confidence_score` pipeline shape) and
    is where the final scores would be re-validated/clamped if a future
    policy needed to adjust them independently of the LLM call itself.
    """
    clamped = [
        rename.model_copy(update={"confidence": min(1.0, max(0.0, rename.confidence))})
        for rename in state["confirmed_renames"]
    ]
    return {"confirmed_renames": clamped}


def build_rename_resolution_subgraph(
    *, confirmation_port: RenameConfirmationPort
) -> CompiledStateGraph[RenameResolutionState, None, Any, Any]:
    """Build and compile the nested rename-resolution subgraph."""
    graph = StateGraph(RenameResolutionState)

    graph.add_node("fuzzy_match", _fuzzy_match)
    graph.add_node("llm_confirm", _make_llm_confirm_node(confirmation_port))
    graph.add_node("confidence_score", _confidence_score)

    graph.add_edge(START, "fuzzy_match")
    graph.add_edge("fuzzy_match", "llm_confirm")
    graph.add_edge("llm_confirm", "confidence_score")
    graph.add_edge("confidence_score", END)

    return graph.compile()
