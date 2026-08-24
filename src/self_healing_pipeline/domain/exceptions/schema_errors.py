"""Tier 2 schema-drift failure exception.

Follows `csv_errors.py`'s own stated extension pattern exactly (see that
module's docstring: "Future Tier 2/3 failure modes are added the same
way: a new subclass of `PipelineError`... with its own `FAILURE_CLASS`.
No existing class, and no router, needs to change.").

`SchemaDriftError` does not itself assert that drift is present — it
asks Tier 2 to check `table_name`'s current schema against its baseline
(`configs/schema_baselines/<table_name>.json`, unchanged). Whether drift
actually exists, and its shape (rename/add/remove/type-change), is
determined entirely by the existing, unmodified `schema_repair_workflow`
once this error is routed to its handler — never by this class or by
`ErrorRouter` itself.
"""

from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass


class SchemaDriftError(PipelineError):
    """Raised to ask Tier 2 to check `table_name` for schema drift.

    Unlike `CsvRepairError` subclasses (one per confirmed failure
    dimension), there is only one Tier 2 error type: which repair
    operations (if any) are needed is a data-driven outcome of the
    existing `compute_column_diff`/`schema_repair_workflow`, not
    something this exception's type could usefully encode.
    """

    def __init__(
        self,
        message: str,
        *,
        table_name: str,
        source: str | None = None,
        file_path: str | None = None,
    ) -> None:
        super().__init__(
            message,
            failure_class=FailureClass.SCHEMA_DRIFT,
            source=source,
            table_name=table_name,
            file_path=file_path,
        )
