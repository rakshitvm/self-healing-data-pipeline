"""Tier 1 CSV self-healing pipeline — CLI entry point.

This module is the composition root: the one place in the codebase that
constructs infrastructure directly. Everything it builds is injected into
the next component via dependency injection — no domain or application
code ever constructs infrastructure itself.

Production pipeline wired here:

    ErrorRouter
        -> LangGraphCsvRepairAgent (RepairAgent adapter)
            -> run_audited_csv_repair(...)
                -> csv_repair_workflow (the canonical LangGraph StateGraph,
                   traced automatically via MLflow LangChain/OpenAI autolog —
                   see `enable_tracing`; zero changes to the graph itself)
                -> RepairAuditStore (PostgreSQL)
                -> RepairRunTracker (MLflow Runs, best-effort)
                -> RepairTraceTracer (MLflow Traces, best-effort)

`build_error_router` takes every port as an explicit argument (pure
dependency injection, easily tested with fakes). `build_production_error_router`
is the only function that reads real configuration and constructs real
infrastructure (Postgres connection, MLflow client, and the LLM provider
selected by `LLM_PROVIDER` — `groq` for temporary development, or
`azure_openai`, the specification's intended production provider).

Usage:
    python -m self_healing_pipeline.interfaces.cli.main repair <file.csv>
"""

import sys
from pathlib import Path
from uuid import UUID, uuid4

import click
from pydantic import ValidationError

from self_healing_pipeline.infrastructure.logging.logger import configure_logging, get_logger

from self_healing_pipeline.application.orchestration.audited_schema_repair import (
    run_audited_schema_repair,
)
from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    build_csv_repair_workflow,
)
from self_healing_pipeline.application.orchestration.error_router import ErrorRouter
from self_healing_pipeline.application.orchestration.schema_repair_workflow import (
    build_initial_schema_repair_state,
    build_schema_repair_workflow,
    result_from_final_state,
)
from self_healing_pipeline.domain.exceptions.csv_errors import (
    CsvRepairError,
    EngineSelectionError,
    HeaderDetectionError,
    InvisibleCharactersError,
    MixedDelimiterError,
    NoHeaderError,
    SingleColumnMalformationError,
    WrongDelimiterError,
    WrongEncodingError,
)
from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.exceptions.schema_errors import SchemaDriftError
from self_healing_pipeline.domain.interfaces.repositories.repair_audit_store import (
    RepairAuditStore,
)
from self_healing_pipeline.domain.interfaces.services.csv_failure_detector import (
    CsvFailureDetector,
)
from self_healing_pipeline.domain.interfaces.services.csv_repair_executor import (
    CsvRepairExecutor,
)
from self_healing_pipeline.domain.interfaces.services.csv_approval_port import CsvHumanApprovalPort
from self_healing_pipeline.domain.interfaces.services.csv_repair_proposal_port import (
    CsvRepairProposalPort,
)
from self_healing_pipeline.domain.interfaces.services.repair_run_tracker import RepairRunTracker
from self_healing_pipeline.domain.interfaces.services.repair_trace_tracer import RepairTraceTracer
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.agents.langgraph_csv_repair_agent import (
    LangGraphCsvRepairAgent,
)
from self_healing_pipeline.infrastructure.agents.langgraph_schema_repair_agent import (
    LangGraphSchemaRepairAgent,
)
from self_healing_pipeline.infrastructure.cloud.azure_blob_uploader import AzureBlobSasUploader
from self_healing_pipeline.infrastructure.cloud.cloud_integration_service import (
    CloudIntegrationService,
)
from self_healing_pipeline.infrastructure.cloud.databricks_job_trigger import DatabricksJobTrigger
from self_healing_pipeline.infrastructure.config.cloud_settings import (
    load_azure_storage_settings,
    load_databricks_settings,
)
from self_healing_pipeline.infrastructure.config.settings import get_settings
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
)
from self_healing_pipeline.infrastructure.csv.mixed_delimiter_row_repair_executor import (
    MixedDelimiterCsvRepairExecutor,
)
from self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor import (
    PandasCsvRepairExecutor,
)
from self_healing_pipeline.infrastructure.llm.proposal_provider_factory import (
    build_proposal_provider,
)
from self_healing_pipeline.infrastructure.mlflow.mlflow_repair_run_tracker import (
    MlflowRepairRunTracker,
)
from self_healing_pipeline.infrastructure.mlflow.mlflow_repair_trace_tracer import (
    MlflowRepairTraceTracer,
)
from self_healing_pipeline.infrastructure.llm.rename_confirmation_provider_factory import (
    build_rename_confirmation_provider,
)
from self_healing_pipeline.infrastructure.mlflow.tracing_setup import enable_tracing, flush_traces
from self_healing_pipeline.infrastructure.persistence.postgres.repair_audit_store import (
    PostgresRepairAuditStore,
)
from self_healing_pipeline.infrastructure.schema.composite_migration_history_store import (
    CompositeMigrationHistoryStore,
)
from self_healing_pipeline.infrastructure.schema.json_migration_history_store import (
    JsonMigrationHistoryStore,
)
from self_healing_pipeline.infrastructure.schema.json_schema_baseline_store import (
    JsonSchemaBaselineStore,
)
from self_healing_pipeline.infrastructure.schema.pandas_schema_executor import (
    PandasSchemaExecutor,
)
from self_healing_pipeline.infrastructure.schema.pandas_schema_inspector import (
    PandasSchemaInspector,
)
from self_healing_pipeline.infrastructure.schema.postgres_schema_migration_store import (
    PostgresSchemaMigrationStore,
)
from self_healing_pipeline.interfaces.cli.click_csv_human_approval import (
    ClickCsvHumanApprovalPort,
)
from self_healing_pipeline.interfaces.cli.click_human_approval import ClickHumanApprovalPort

_FAILURE_CLASS_TO_ERROR: dict[FailureClass, type[CsvRepairError]] = {
    FailureClass.WRONG_DELIMITER: WrongDelimiterError,
    FailureClass.WRONG_ENCODING: WrongEncodingError,
    FailureClass.HEADER_DETECTION: HeaderDetectionError,
    FailureClass.ENGINE_SELECTION: EngineSelectionError,
    FailureClass.SINGLE_COLUMN_MALFORMATION: SingleColumnMalformationError,
    FailureClass.MIXED_DELIMITER: MixedDelimiterError,
    FailureClass.NO_HEADER: NoHeaderError,
    FailureClass.INVISIBLE_CHARACTERS: InvisibleCharactersError,
}


def build_error_router(
    *,
    detector: CsvFailureDetector,
    executor: CsvRepairExecutor,
    llm_port: CsvRepairProposalPort,
    audit_store: RepairAuditStore,
    run_tracker: RepairRunTracker | None = None,
    trace_tracer: RepairTraceTracer | None = None,
    approval_port: CsvHumanApprovalPort | None = None,
    mixed_delimiter_executor: CsvRepairExecutor | None = None,
) -> ErrorRouter:
    """Wire the real Tier 1 pipeline from injected ports.

    Pure dependency injection — no settings read, no infrastructure
    constructed, so this is usable with fakes in tests. `mixed_delimiter_executor`
    is the separate `CsvRepairExecutor` used only for `MIXED_DELIMITER`
    prescriptions (see `build_csv_repair_workflow`); omitting it is a
    no-op for every other failure class.
    """
    graph = build_csv_repair_workflow(
        detector=detector,
        executor=executor,
        llm_port=llm_port,
        approval_port=approval_port,
        mixed_delimiter_executor=mixed_delimiter_executor,
    )
    agent = LangGraphCsvRepairAgent(
        graph, audit_store=audit_store, run_tracker=run_tracker, trace_tracer=trace_tracer
    )

    router = ErrorRouter()
    router.register(CsvRepairError, agent)
    return router


def build_production_error_router() -> ErrorRouter:
    """Build the real, production `ErrorRouter` using live infrastructure.

    Reads `Settings` (Postgres, MLflow) and selects the LLM provider via
    `LLM_PROVIDER` (`groq` for temporary development, or `azure_openai`).
    This is the only function in the codebase that constructs live
    infrastructure for the CLI; everything else is injected.

    Also enables MLflow tracing (best-effort — `enable_tracing`'s return
    value is intentionally not checked here: tracing is optional
    observability, never a startup precondition).
    """
    settings = get_settings()
    enable_tracing(settings.mlflow)
    return build_error_router(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=build_proposal_provider(),
        mixed_delimiter_executor=MixedDelimiterCsvRepairExecutor(),
        audit_store=PostgresRepairAuditStore.from_settings(settings.database),
        run_tracker=MlflowRepairRunTracker.from_settings(settings.mlflow),
        trace_tracer=MlflowRepairTraceTracer(),
        approval_port=ClickCsvHumanApprovalPort(),
    )


@click.group()
def cli() -> None:
    """Self-Healing Data Pipeline — Tier 1 CLI."""
    configure_logging()


@cli.command()
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
def repair(file_path: str) -> None:
    """Detect and repair a CSV file through the real Tier 1 pipeline.

    Detection happens once, up front, purely to decide whether there is
    anything to route to `ErrorRouter` at all (mirroring how an upstream
    ingestion failure would first be noticed in production). If the file
    is healthy, `ErrorRouter` is never invoked, so the LLM provider, the
    audit store, and MLflow are never touched either — this is the "zero
    LLM calls on the healthy path" guarantee, demonstrated exactly as
    implemented rather than as a separate code path.
    """
    detector = LocalCsvFailureDetector()
    failure_class = detector.detect(file_path)

    if failure_class is None:
        click.echo(f"HEALTHY: {file_path} already parses correctly. No repair needed.")
        return

    error_cls = _FAILURE_CLASS_TO_ERROR[failure_class]
    error: PipelineError = error_cls(
        f"Detected {failure_class.value} in {file_path}", file_path=file_path
    )

    router = build_production_error_router()
    try:
        result = router.route(error)
    finally:
        # Bounded, best-effort: mitigates a risk (Ticket 013) where
        # MLflow's default trace flush can hang the process for a long
        # time if the tracking server is unreachable. The repair result
        # above is already computed and unaffected.
        flush_traces()

    click.echo(f"failure_class: {failure_class.value}")
    click.echo(f"success: {result.success}")
    click.echo(f"applied: {result.applied}")
    if result.source_path is not None:
        click.echo(f"source_path: {result.source_path}")
    if result.output_path is not None:
        click.echo(f"output_path: {result.output_path}")
    if result.prescription is not None:
        click.echo(f"prescription: {result.prescription.model_dump_json()}")
    click.echo(f"message: {result.message}")
    if result.validation_errors:
        click.echo("validation_errors:")
        for validation_error in result.validation_errors:
            click.echo(f"  - {validation_error}")

    if not result.success:
        sys.exit(1)


_SCHEMA_BASELINES_DIR = Path("configs/schema_baselines")
_SCHEMA_MIGRATIONS_DIR = Path("configs/schema_migrations")


def _build_production_migration_history_store() -> CompositeMigrationHistoryStore:
    """JSON is always the authoritative primary; the PostgreSQL mirror is
    best-effort — if Postgres is unreachable, schema repair still works,
    exactly as MLflow tracking degrades gracefully for Tier 1.
    """
    primary = JsonMigrationHistoryStore(root=_SCHEMA_MIGRATIONS_DIR)
    mirror: PostgresSchemaMigrationStore | None
    try:
        mirror = PostgresSchemaMigrationStore.from_settings(get_settings().database)
    except Exception:  # noqa: BLE001 - best-effort mirror, must never block schema repair
        mirror = None
    return CompositeMigrationHistoryStore(primary=primary, mirror=mirror)


@cli.command(name="schema-repair")
@click.argument("table")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--episode-id",
    "episode_id_option",
    default=None,
    help="Reuse an existing episode ID for continuity (skips re-proposing rejected operations).",
)
def schema_repair(table: str, file_path: str, episode_id_option: str | None) -> None:
    """Detect and repair schema drift for TABLE against FILE_PATH's current structure.

    A separate Tier 2 agent/workflow from `repair` (Tier 1 CSV repair) —
    same composition-root pattern, same MLflow/logging infrastructure,
    entirely independent pipeline. The LLM never authorizes application:
    every prescription, regardless of confidence, requires an explicit
    human "y"/"yes" before `apply` runs.
    """
    episode_id = UUID(episode_id_option) if episode_id_option else uuid4()

    settings = get_settings()
    enable_tracing(settings.mlflow)

    history_store = _build_production_migration_history_store()
    graph = build_schema_repair_workflow(
        baseline_store=JsonSchemaBaselineStore(root=_SCHEMA_BASELINES_DIR),
        inspector=PandasSchemaInspector(),
        history_store=history_store,
        confirmation_port=build_rename_confirmation_provider(),
        approval_port=ClickHumanApprovalPort(),
        executor=PandasSchemaExecutor(),
    )
    initial_state = build_initial_schema_repair_state(
        episode_id=episode_id, table=table, file_path=file_path
    )

    try:
        final_state = run_audited_schema_repair(
            graph, initial_state, history_store=history_store, trace_tracer=MlflowRepairTraceTracer()
        )
    finally:
        flush_traces()

    result = result_from_final_state(final_state)

    click.echo("")
    click.echo(f"episode_id: {episode_id}")
    click.echo(f"table: {table}")
    click.echo(f"status: {result.status.value}")
    click.echo(f"confidence: {result.confidence}")
    click.echo(f"human_approved: {result.human_approved}")
    if result.prescription is not None:
        click.echo(f"prescription: {result.prescription.model_dump_json()}")
    if result.output_path is not None:
        click.echo(f"output_path: {result.output_path}")
    click.echo(f"message: {result.message}")

    if result.status.value not in ("succeeded", "healthy"):
        sys.exit(1)


def _build_cloud_integration_service() -> tuple[CloudIntegrationService, list[str]]:
    """Best-effort construction of the optional cloud-integration service.

    Never raises: any piece of configuration that is missing or invalid
    is treated as "that half is not configured" — never a reason to fail
    `repair-and-process` itself. Returns the service plus human-readable
    notes about what was/wasn't configured, for the CLI to echo.
    """
    notes: list[str] = []

    uploader: AzureBlobSasUploader | None
    try:
        storage_settings = load_azure_storage_settings()
        uploader = AzureBlobSasUploader(
            container_url=storage_settings.container_url, sas_token=storage_settings.sas_token
        )
    except ValidationError:
        uploader = None
        notes.append(
            "Azure Storage is not configured "
            "(AZURE_STORAGE_CONTAINER_URL/AZURE_STORAGE_SAS_TOKEN) — cloud upload skipped."
        )

    trigger: DatabricksJobTrigger | None
    try:
        databricks_settings = load_databricks_settings()
        trigger = DatabricksJobTrigger(
            host=databricks_settings.host,
            token=databricks_settings.token,
            job_id=databricks_settings.job_id,
            param_name=databricks_settings.notebook_param_name,
        )
    except ValidationError:
        trigger = None
        notes.append(
            "Databricks is not configured "
            "(DATABRICKS_HOST/DATABRICKS_TOKEN/DATABRICKS_JOB_ID) — job trigger skipped."
        )

    return CloudIntegrationService(uploader=uploader, trigger=trigger), notes


def _build_production_schema_repair_agent() -> LangGraphSchemaRepairAgent:
    """Build the Tier 2 `RepairAgent`, reusing the same wiring
    `schema-repair` uses — baseline store, inspector, history store,
    confirmation/approval ports, executor.

    Called only from `repair_and_process`, only when `--table` was
    supplied and a Tier 2 check is about to run.
    """
    history_store = _build_production_migration_history_store()
    graph = build_schema_repair_workflow(
        baseline_store=JsonSchemaBaselineStore(root=_SCHEMA_BASELINES_DIR),
        inspector=PandasSchemaInspector(),
        history_store=history_store,
        confirmation_port=build_rename_confirmation_provider(),
        approval_port=ClickHumanApprovalPort(),
        executor=PandasSchemaExecutor(),
    )
    return LangGraphSchemaRepairAgent(
        graph, history_store=history_store, trace_tracer=MlflowRepairTraceTracer()
    )


def _run_cloud_integration_step(local_file_path: str) -> None:
    """Run the optional cloud-integration step against `local_file_path`
    and exit 2 if it was configured but failed.

    Callers only ever reach this once the repair/check itself has
    already succeeded (healthy, or genuinely repaired) — the only
    remaining failure mode here is the cloud step itself, so the
    exit-code contract is identical regardless of which case got here.
    """
    click.echo("")
    click.echo("--- cloud integration ---")
    service, notes = _build_cloud_integration_service()
    for note in notes:
        click.echo(note)

    outcome = service.process(local_file_path=local_file_path)
    click.echo(f"uploaded: {outcome.uploaded}")
    if outcome.cloud_path is not None:
        click.echo(f"cloud_path: {outcome.cloud_path}")
    if outcome.upload_error is not None:
        click.echo(f"upload_error: {outcome.upload_error}")
    click.echo(f"databricks_triggered: {outcome.triggered}")
    if outcome.databricks_run_id is not None:
        click.echo(f"databricks_run_id: {outcome.databricks_run_id}")
    if outcome.trigger_error is not None:
        click.echo(f"trigger_error: {outcome.trigger_error}")
    click.echo(f"cloud_message: {outcome.message}")

    if outcome.upload_error is not None or outcome.trigger_error is not None:
        sys.exit(2)


@cli.command(name="repair-and-process")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--table",
    "table",
    default=None,
    help=(
        "Table name to also check for Tier 2 schema drift, against "
        "configs/schema_baselines/<table>.json via the existing "
        "SchemaRepair workflow. Omit for Tier-1-only behavior, "
        "unchanged from before this option existed."
    ),
)
def repair_and_process(file_path: str, table: str | None) -> None:
    """Run the Tier 1 `repair` pipeline (or, if `--table` is given and no
    Tier 1 parse failure is found, the Tier 2 `schema-repair` workflow),
    then — provided that check didn't fail or get rejected — upload the
    resulting file to Azure Blob Storage and trigger a Databricks job.

    Both tiers reach the same `ErrorRouter`: Tier 1 via `CsvRepairError`
    -> `LangGraphCsvRepairAgent`, Tier 2 via `SchemaDriftError`, routing
    to whichever detection/repair pipeline already handles that error
    type. A Tier 1 parse failure is always handled first, regardless of
    `--table` — a file that doesn't parse isn't diffed against a schema
    baseline.

    A healthy file (no Tier 1 failure, and no drift or no `--table`)
    uploads the original `file_path`. A repaired file uploads
    `result.output_path`. `result.success is False` always stops before
    any cloud call (exit code 1). Exit code 2 means the repair succeeded
    but the cloud step failed.
    """
    detector = LocalCsvFailureDetector()
    failure_class = detector.detect(file_path)

    error: PipelineError | None
    if failure_class is not None:
        error_cls = _FAILURE_CLASS_TO_ERROR[failure_class]
        error = error_cls(f"Detected {failure_class.value} in {file_path}", file_path=file_path)
    elif table is not None:
        error = SchemaDriftError(
            f"Checking table {table!r} for schema drift in {file_path}",
            table_name=table,
            file_path=file_path,
        )
    else:
        error = None

    if error is None:
        click.echo(f"HEALTHY: {file_path} already parses correctly. No repair needed.")
        get_logger(agent="CsvRepairAgent", node="healthy", table=table).info(
            "repair_completed",
            success=True,
            failure_class=None,
            source_path=file_path,
            message="No CSV failure detected; file already parses correctly. No repair needed.",
        )
        _run_cloud_integration_step(file_path)
        return

    router = build_production_error_router()
    if isinstance(error, SchemaDriftError):
        router.register(SchemaDriftError, _build_production_schema_repair_agent())

    try:
        result = router.route(error)
    finally:
        flush_traces()

    click.echo(f"failure_class: {error.failure_class.value}")
    if table is not None:
        click.echo(f"table: {table}")
    click.echo(f"success: {result.success}")
    click.echo(f"applied: {result.applied}")
    if result.source_path is not None:
        click.echo(f"source_path: {result.source_path}")
    if result.output_path is not None:
        click.echo(f"output_path: {result.output_path}")
    if result.prescription is not None:
        click.echo(f"prescription: {result.prescription.model_dump_json()}")
    click.echo(f"message: {result.message}")
    if result.validation_errors:
        click.echo("validation_errors:")
        for validation_error in result.validation_errors:
            click.echo(f"  - {validation_error}")

    if not result.success:
        click.echo("")
        click.echo("cloud_integration: skipped (repair did not succeed)")
        sys.exit(1)

    local_file_path = result.output_path if result.output_path is not None else file_path
    _run_cloud_integration_step(local_file_path)


if __name__ == "__main__":
    cli()
