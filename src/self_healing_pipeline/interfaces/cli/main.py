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

from self_healing_pipeline.infrastructure.logging.logger import configure_logging

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
    SingleColumnMalformationError,
    WrongDelimiterError,
    WrongEncodingError,
)
from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.interfaces.repositories.repair_audit_store import (
    RepairAuditStore,
)
from self_healing_pipeline.domain.interfaces.services.csv_failure_detector import (
    CsvFailureDetector,
)
from self_healing_pipeline.domain.interfaces.services.csv_repair_executor import (
    CsvRepairExecutor,
)
from self_healing_pipeline.domain.interfaces.services.csv_repair_proposal_port import (
    CsvRepairProposalPort,
)
from self_healing_pipeline.domain.interfaces.services.repair_run_tracker import RepairRunTracker
from self_healing_pipeline.domain.interfaces.services.repair_trace_tracer import RepairTraceTracer
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.agents.langgraph_csv_repair_agent import (
    LangGraphCsvRepairAgent,
)
from self_healing_pipeline.infrastructure.config.settings import get_settings
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
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
from self_healing_pipeline.interfaces.cli.click_human_approval import ClickHumanApprovalPort

_FAILURE_CLASS_TO_ERROR: dict[FailureClass, type[CsvRepairError]] = {
    FailureClass.WRONG_DELIMITER: WrongDelimiterError,
    FailureClass.WRONG_ENCODING: WrongEncodingError,
    FailureClass.HEADER_DETECTION: HeaderDetectionError,
    FailureClass.ENGINE_SELECTION: EngineSelectionError,
    FailureClass.SINGLE_COLUMN_MALFORMATION: SingleColumnMalformationError,
}


def build_error_router(
    *,
    detector: CsvFailureDetector,
    executor: CsvRepairExecutor,
    llm_port: CsvRepairProposalPort,
    audit_store: RepairAuditStore,
    run_tracker: RepairRunTracker | None = None,
    trace_tracer: RepairTraceTracer | None = None,
) -> ErrorRouter:
    """Wire the real Tier 1 pipeline from injected ports.

    Pure dependency injection — no settings are read and no infrastructure
    is constructed here, which is exactly what makes this function usable
    with fakes in tests.
    """
    graph = build_csv_repair_workflow(detector=detector, executor=executor, llm_port=llm_port)
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
        audit_store=PostgresRepairAuditStore.from_settings(settings.database),
        run_tracker=MlflowRepairRunTracker.from_settings(settings.mlflow),
        trace_tracer=MlflowRepairTraceTracer(),
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
        # Bounded, best-effort: mitigates an empirically-verified risk
        # (Ticket 013) where MLflow's default trace flush can hang the
        # process for a long time if the tracking server is unreachable.
        # The repair result above is already computed and unaffected.
        flush_traces()

    click.echo(f"failure_class: {failure_class.value}")
    click.echo(f"success: {result.success}")
    click.echo(f"applied: {result.applied}")
    if result.prescription is not None:
        click.echo(f"prescription: {result.prescription.model_dump_json()}")
    click.echo(f"message: {result.message}")

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
    click.echo(f"message: {result.message}")

    if result.status.value not in ("succeeded", "healthy"):
        sys.exit(1)


if __name__ == "__main__":
    cli()
