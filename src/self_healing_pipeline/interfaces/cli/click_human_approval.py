"""Click-backed `HumanApprovalPort` — the only place Click touches approval.

Confined to `interfaces/cli`: the domain and application layers depend
only on `HumanApprovalPort` (see `human_approval_port.py`), never on
Click. Swapping this for a UI/API-backed approval mechanism later means
writing one new class here — nothing in the workflow changes.
"""

import click

from self_healing_pipeline.domain.interfaces.services.human_approval_port import ApprovalRequest
from self_healing_pipeline.domain.value_objects.schema_repair_operations import OperationType


def _format_diff(request: ApprovalRequest) -> list[str]:
    """Symbols describe the repair's effect on the data (matching the
    project's own example format), not raw structural diff direction:
    `+` = added back with a default (baseline has it, current is
    missing it), `-` = dropped (current has it, baseline doesn't)."""
    lines = []
    for change in request.diff.type_changed:
        lines.append(f"  ~ {change.column}: {change.from_type} -> {change.to_type}")
    for column in request.diff.removed:
        lines.append(f"  + {column.name}: {column.type}")
    for column in request.diff.added:
        lines.append(f"  - {column.name}")
    return lines


def _format_operation(index: int, op: OperationType, **kwargs: object) -> str:
    if op is OperationType.RENAME:
        return f"  {index}. rename {kwargs['column']} -> {kwargs['target_column']}"
    if op is OperationType.CAST:
        return f"  {index}. cast {kwargs['column']} -> {kwargs['target_type']}"
    if op is OperationType.DROP:
        return f"  {index}. drop {kwargs['column']}"
    return f"  {index}. add {kwargs['column']} = {kwargs['default']!r}"


class ClickHumanApprovalPort:
    """Concrete `HumanApprovalPort` prompting interactively via Click.

    Only `"y"`/`"yes"` (case-insensitive) proceeds to apply — any other
    response (including empty input) rejects, matching `click.confirm`'s
    own `default=False` semantics.
    """

    def request_approval(self, request: ApprovalRequest) -> bool:
        click.echo("")
        click.echo(f"Schema drift detected for table: {request.table}")
        click.echo("")
        click.echo("Diff:")
        for line in _format_diff(request):
            click.echo(line)
        click.echo("")
        click.echo("Proposed repair order:")
        for index, op in enumerate(request.prescription.operations, start=1):
            click.echo(
                _format_operation(
                    index,
                    op.op,
                    column=op.column,
                    target_column=op.target_column,
                    target_type=op.target_type,
                    default=op.default,
                )
            )
        click.echo("")
        click.echo(f"Confidence: {request.confidence:.2f}")
        if request.escalated:
            click.echo("⚠ ESCALATED: confidence is below the configured threshold.")
        click.echo("")

        return click.confirm("Approve repair?", default=False)
