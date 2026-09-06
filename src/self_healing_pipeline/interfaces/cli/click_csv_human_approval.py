"""Click-backed `CsvHumanApprovalPort` — the only place Click touches
multi-failure CSV repair approval.

Confined to `interfaces/cli`, mirroring `click_human_approval.py`'s
Tier 2 pattern exactly: the domain and application layers depend only on
`CsvHumanApprovalPort`, never on Click.
"""

import click

from self_healing_pipeline.domain.interfaces.services.csv_approval_port import CsvApprovalRequest
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass


class ClickCsvHumanApprovalPort:
    """Concrete `CsvHumanApprovalPort` prompting interactively via Click.

    Only `"y"`/`"yes"` (case-insensitive) proceeds to apply — any other
    response (including empty input) rejects, matching `click.confirm`'s
    own `default=False` semantics.
    """

    def request_approval(self, request: CsvApprovalRequest) -> bool:
        click.echo("")
        click.echo(f"Multiple simultaneous failures detected for: {request.file_path}")
        click.echo("")
        click.echo("Detected failures:")
        for failure_class in sorted(f.value for f in request.failure_classes):
            click.echo(f"  - {failure_class}")
        click.echo("")
        click.echo(f"Proposed combined prescription: {request.prescription.model_dump_json()}")

        if request.mixed_delimiter_rows:
            click.echo("")
            click.echo(f"Expected delimiter: {request.prescription.delimiter}")
            for row in request.mixed_delimiter_rows:
                click.echo(f"Affected row: {row.row_number}")
                click.echo(f"Observed delimiter: {row.observed_delimiter}")
                click.echo("Proposed repair:")
                click.echo(f"  {row.original_text} -> {row.repaired_text}")

        if FailureClass.NO_HEADER in request.failure_classes:
            click.echo("")
            click.echo(
                "No header row detected; every row will be treated as data "
                "(columns get default integer names)."
            )

        if FailureClass.INVISIBLE_CHARACTERS in request.failure_classes:
            click.echo("")
            click.echo(
                "Invisible/BOM characters detected in the header row; they "
                "will be stripped from the repaired output's column names."
            )

        click.echo("")

        return click.confirm("Approve combined repair?", default=False)
