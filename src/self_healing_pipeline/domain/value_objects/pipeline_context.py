"""Pipeline context value object.

Carries the correlation and provenance information describing where a
pipeline failure originated (ADF, Databricks, source file/table). It is
an immutable value object passed by value between layers and agents; it
holds no infrastructure dependencies.
"""

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass


class PipelineContext(BaseModel):
    """Immutable correlation context describing a pipeline failure."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    episode_id: UUID = Field(default_factory=uuid4)
    adf_run_id: UUID | None = None
    databricks_job_id: int | None = None
    source: str
    table_name: str | None = None
    file_path: str | None = None
    failure_class: FailureClass
    trace_id: UUID | None = None
