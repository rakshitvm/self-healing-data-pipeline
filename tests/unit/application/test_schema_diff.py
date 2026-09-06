"""Focused unit tests for the deterministic `compute_column_diff` engine.

No LLM, no fakes standing in for reasoning — this is pure structural
comparison, so every assertion here is exact and deterministic.
"""

from self_healing_pipeline.application.orchestration.schema_diff import (
    compute_column_diff,
    compute_rename_hints,
    looks_headerless,
)
from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition, SchemaBaseline

BASELINE = SchemaBaseline(
    table="customers",
    version=1,
    columns=(
        ColumnDefinition(name="id", type="int64", nullable=False),
        ColumnDefinition(name="customer_name", type="string", nullable=False),
        ColumnDefinition(name="age", type="int64", nullable=True),
    ),
)


def test_diff_detects_added_column() -> None:
    current = BASELINE.columns + (ColumnDefinition(name="country", type="string"),)

    diff = compute_column_diff(BASELINE, current)

    assert [c.name for c in diff.added] == ["country"]
    assert diff.removed == ()
    assert diff.type_changed == ()
    assert diff.has_changes is True


def test_diff_detects_removed_column() -> None:
    current = tuple(c for c in BASELINE.columns if c.name != "age")

    diff = compute_column_diff(BASELINE, current)

    assert [c.name for c in diff.removed] == ["age"]
    assert diff.added == ()
    assert diff.has_changes is True


def test_diff_detects_type_change() -> None:
    current = tuple(
        c.model_copy(update={"type": "string"}) if c.name == "age" else c for c in BASELINE.columns
    )

    diff = compute_column_diff(BASELINE, current)

    assert len(diff.type_changed) == 1
    assert diff.type_changed[0].column == "age"
    assert diff.type_changed[0].from_type == "int64"
    assert diff.type_changed[0].to_type == "string"
    assert diff.has_changes is True


def test_diff_reports_no_changes_for_identical_schema() -> None:
    diff = compute_column_diff(BASELINE, BASELINE.columns)

    assert diff.has_changes is False
    assert diff.added == () and diff.removed == () and diff.type_changed == ()


def test_diff_produces_rename_hint_for_similar_removed_and_added_names() -> None:
    current = tuple(
        ColumnDefinition(name="name", type="string") if c.name == "customer_name" else c
        for c in BASELINE.columns
    )

    diff = compute_column_diff(BASELINE, current)

    assert "customer_name" in [c.name for c in diff.removed]
    assert "name" in [c.name for c in diff.added]
    assert len(diff.rename_hints) == 1
    assert diff.rename_hints[0].removed_column == "customer_name"
    assert diff.rename_hints[0].added_column == "name"
    assert diff.rename_hints[0].similarity > 0.4


def test_rename_hints_are_greedy_one_to_one() -> None:
    hints = compute_rename_hints(removed=("customer_name", "age"), added=("name", "years"))

    used_removed = {h.removed_column for h in hints}
    used_added = {h.added_column for h in hints}
    assert len(used_removed) == len(hints)
    assert len(used_added) == len(hints)


def test_unrelated_column_names_produce_no_rename_hint() -> None:
    hints = compute_rename_hints(removed=("customer_name",), added=("zzz_totally_unrelated_xyz",))
    assert hints == ()


# --- looks_headerless ------------------------------------------------------


def test_looks_headerless_true_for_genuine_headerless_data() -> None:
    """A real data row, positionally matching the baseline's types
    exactly, correctly looks headerless."""
    assert looks_headerless(("1", "Alice", "30"), BASELINE) is True


def test_looks_headerless_false_for_a_real_header() -> None:
    """The false-positive guard: a real header's own labels must never
    be mistaken for a swallowed data row."""
    assert looks_headerless(("id", "customer_name", "age"), BASELINE) is False


def test_looks_headerless_false_for_column_count_mismatch() -> None:
    assert looks_headerless(("1", "Alice"), BASELINE) is False
    assert looks_headerless(("1", "Alice", "30", "extra"), BASELINE) is False


def test_looks_headerless_false_when_baseline_is_all_string() -> None:
    """An all-string baseline is fundamentally undetectable this way — a
    real header of text labels is indistinguishable from a data row of
    text values by parseability alone. Always False, regardless of the
    candidate values, never a guess."""
    all_string_baseline = SchemaBaseline(
        table="t",
        version=1,
        columns=(
            ColumnDefinition(name="a", type="string"),
            ColumnDefinition(name="b", type="string"),
        ),
    )

    assert looks_headerless(("1", "2"), all_string_baseline) is False
    assert looks_headerless(("a", "b"), all_string_baseline) is False


def test_looks_headerless_false_when_one_position_fails_to_parse() -> None:
    """Never a partial/guessed classification — one bad position is
    enough to refuse the whole classification, same policy as
    MIXED_DELIMITER's own resolvers."""
    # position 0 ("abc") does not parse as int64, even though position 2
    # ("30") does — must not partially match.
    assert looks_headerless(("abc", "Alice", "30"), BASELINE) is False
