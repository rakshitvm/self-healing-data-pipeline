"""Deterministic structural schema diff.

Pure function: given a `SchemaBaseline` and the current, actually-observed
columns, compute a `ColumnDiff`. Uses stdlib `difflib.SequenceMatcher`
for rename-hint similarity — no LLM, no network, no randomness. This is
the one and only place the raw structural facts are computed; every
downstream node (including the LLM rename-confirmation step) only ever
*confirms or refines* what this function already established.
"""

from difflib import SequenceMatcher

from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition, SchemaBaseline
from self_healing_pipeline.domain.value_objects.column_diff import ColumnDiff, RenameHint, TypeChange

_RENAME_HINT_FLOOR = 0.4


def compute_column_diff(
    baseline: SchemaBaseline, current: tuple[ColumnDefinition, ...]
) -> ColumnDiff:
    """Compute the deterministic structural diff between `baseline` and `current`."""
    baseline_by_name = {c.name: c for c in baseline.columns}
    current_by_name = {c.name: c for c in current}

    added_names = current_by_name.keys() - baseline_by_name.keys()
    removed_names = baseline_by_name.keys() - current_by_name.keys()
    common_names = baseline_by_name.keys() & current_by_name.keys()

    added = tuple(current_by_name[name] for name in sorted(added_names))
    removed = tuple(baseline_by_name[name] for name in sorted(removed_names))
    type_changed = tuple(
        TypeChange(
            column=name,
            from_type=baseline_by_name[name].type,
            to_type=current_by_name[name].type,
        )
        for name in sorted(common_names)
        if baseline_by_name[name].type != current_by_name[name].type
    )
    rename_hints = compute_rename_hints(
        removed=tuple(c.name for c in removed), added=tuple(c.name for c in added)
    )

    return ColumnDiff(
        added=added, removed=removed, type_changed=type_changed, rename_hints=rename_hints
    )


def compute_rename_hints(
    removed: tuple[str, ...], added: tuple[str, ...]
) -> tuple[RenameHint, ...]:
    """Greedy, deterministic one-to-one best-match pairing by name similarity.

    Public: reused by both `compute_column_diff` (for the reported diff)
    and the nested rename-resolution subgraph's `fuzzy_match` node (for
    resolution) — one implementation, no duplicated fuzzy-matching logic.
    """
    candidates: list[RenameHint] = []
    for removed_name in removed:
        for added_name in added:
            similarity = SequenceMatcher(None, removed_name, added_name).ratio()
            if similarity >= _RENAME_HINT_FLOOR:
                candidates.append(
                    RenameHint(
                        removed_column=removed_name,
                        added_column=added_name,
                        similarity=similarity,
                    )
                )
    candidates.sort(key=lambda hint: hint.similarity, reverse=True)

    used_removed: set[str] = set()
    used_added: set[str] = set()
    hints: list[RenameHint] = []
    for candidate in candidates:
        if candidate.removed_column in used_removed or candidate.added_column in used_added:
            continue
        hints.append(candidate)
        used_removed.add(candidate.removed_column)
        used_added.add(candidate.added_column)

    return tuple(hints)
