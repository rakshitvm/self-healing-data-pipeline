"""Root domain exception for the Self-Healing Data Pipeline.

`PipelineError` is the base of the pipeline's error hierarchy. It carries
the structured context an error router needs to make routing decisions,
independent of any concrete repair agent, orchestration framework
(e.g. LangGraph), or infrastructure concern (database, MLflow, Azure
OpenAI, pandas, Spark, FastAPI). Routing is inheritance-aware: a router
may dispatch on the concrete exception type, on any intermediate base
class, or on the `failure_class` value carried by every instance.
"""

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass


class PipelineError(Exception):
    """Base class for every domain-level pipeline failure.

    Carries the structured context a router needs: a human-readable
    message, the `FailureClass` dimension the failure belongs to, and the
    pipeline coordinates (source, table, file) it occurred against.
    """

    def __init__(
        self,
        message: str,
        *,
        failure_class: FailureClass,
        source: str | None = None,
        table_name: str | None = None,
        file_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.failure_class = failure_class
        self.source = source
        self.table_name = table_name
        self.file_path = file_path

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(message={self.message!r}, "
            f"failure_class={self.failure_class!r}, source={self.source!r}, "
            f"table_name={self.table_name!r}, file_path={self.file_path!r})"
        )
