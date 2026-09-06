"""Deterministic structural schema diff.

Pure function: given a `SchemaBaseline` and the current, actually-observed
columns, compute a `ColumnDiff`. Uses stdlib `difflib.SequenceMatcher`
for rename-hint similarity — no LLM, no network, no randomness. This is
the one and only place the raw structural facts are computed; every
downstream node (including the LLM rename-confirmation step) only ever
*confirms or refines* what this function already established.
"""

from datetime import datetime
from difflib import SequenceMatcher

from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition, SchemaBaseline
from self_healing_pipeline.domain.value_objects.column_diff import ColumnDiff, RenameHint, TypeChange

_RENAME_HINT_FLOOR = 0.4


def _parses_as(value: str, logical_type: str) -> bool:
    """Does `value` (a raw string) plausibly hold data of `logical_type`?

    Deliberately conservative — used only to decide whether a "column
    name" might actually be a swallowed data value (see
    `looks_headerless`), so a false positive here is much worse than a
    false negative. `"string"` always matches (a real header label is
    itself a string), which is exactly why an all-string baseline can
    never be distinguished this way (see `looks_headerless`).
    """
    if logical_type == "string":
        return True
    if logical_type == "int64":
        try:
            int(value)
            return True
        except ValueError:
            return False
    if logical_type == "float64":
        try:
            float(value)
            return True
        except ValueError:
            return False
    if logical_type == "bool":
        return value.strip().lower() in {"true", "false"}
    if logical_type == "datetime":
        try:
            datetime.fromisoformat(value)
            return True
        except ValueError:
            return False
    return False  # unrecognized logical type: never a match


def looks_headerless(candidate_names: tuple[str, ...], baseline: SchemaBaseline) -> bool:
    """Does `candidate_names` look like a swallowed *data* row rather than
    a genuine header — i.e. is `file_path` actually headerless, with
    every row (including what a naive read treated as the header)
    already data, positionally matching `baseline`?

    Pure and deterministic — no LLM, no file I/O of its own.
    `candidate_names` is expected to be `current_schema`'s column
    *names* from a plain `header=0` read: for a genuinely headerless
    file, those names ARE the first data row's values (pandas swallowed
    them as the header), which is exactly what this function examines.

    `False` immediately if the column count doesn't match the baseline
    exactly, or if the baseline has no non-string column at all (a
    real header of text labels is indistinguishable from a data row of
    text values purely by parseability — a hard, honest scope boundary).
    Otherwise `True` only if *every* position's candidate value cleanly
    parses as that position's baseline type — any single position
    failing means unresolved, never a partial/guessed classification.
    """
    if len(candidate_names) != len(baseline.columns):
        return False
    if not any(column.type != "string" for column in baseline.columns):
        return False
    return all(
        _parses_as(name, column.type)
        for name, column in zip(candidate_names, baseline.columns, strict=True)
    )


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
