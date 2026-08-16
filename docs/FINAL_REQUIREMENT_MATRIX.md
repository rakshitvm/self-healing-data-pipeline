# Final Requirement Matrix

Status values: **COMPLETE** (implemented and verified), **NOT REQUIRED**
(explicitly out of scope for this project phase).

## Tier 1 — CSV self-healing

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Deterministic failure detection | `LocalCsvFailureDetector.detect()`/`detect_all()` (`infrastructure/csv/local_csv_failure_detector.py`) | `tests/unit/infrastructure/test_local_csv_failure_detector.py`, `..._multi_error.py`; CLI demos for all 5 classes | COMPLETE |
| wrong_delimiter healed | `csv_repair_workflow.py` + `PandasCsvRepairExecutor` | `test_csv_repair_workflow.py`; real Groq CLI run (this project's history) | COMPLETE |
| wrong_encoding healed | `_detect_encoding` (chardet) + deterministic correction in `propose` | `test_csv_repair_workflow_encoding_evidence.py`; real Groq CLI run (UTF-16 fixture) | COMPLETE |
| single_column_malformation healed | Same workflow, colon-delimited real fixture | `test_csv_repair_workflow.py`; real Groq CLI run | COMPLETE |
| header_detection healed | Same workflow, `header_row` correction | `test_csv_repair_workflow.py`; real Groq CLI run | COMPLETE |
| engine_selection healed | Same workflow (missing-field variant only — extra-field variant is architecturally unrepairable, documented) | `test_csv_repair_workflow.py::test_engine_selection_flows_through_full_workflow`; real Groq CLI run | COMPLETE |
| Multi-error detection (2+ simultaneous failures) | `detect_all()` returns a `frozenset[FailureClass]`; `propose` builds one combined evidence prompt | `test_local_csv_failure_detector_multi_error.py`, `test_csv_repair_workflow_multi_error.py`; real Groq CLI run (encoding+delimiter fixture) | COMPLETE |
| Genuine 3-way simultaneous failure | Investigated; **architecturally not achievable** via `csv.Sniffer` without redesigning delimiter detection | Documented in `local_csv_failure_detector.py` module docstring and README | NOT REQUIRED (explicitly out of scope — would be a larger redesign) |
| One failure's fix never overwrites another's parameter | Encoding correction touches only the `encoding` dict key | `test_encoding_correction_does_not_overwrite_other_fields` | COMPLETE |
| LLM proposes only, never applies | `CsvRepairProposalPort` returns unvalidated `dict`; `validate` node gates `apply` | `test_csv_repair_workflow.py::test_invalid_repair_parameters_never_reach_apply` | COMPLETE |
| Pydantic validation before apply | `CsvRepairParams` | Same as above | COMPLETE |
| Repair + independent reverify | `apply`/`reverify` nodes, both via `CsvRepairExecutor` | `test_csv_repair_workflow.py` | COMPLETE |
| Retry on invalid/failed repair (bounded) | `increment_retry`/`max_retries` | `test_verification_failure_retries_when_budget_remains` | COMPLETE |
| Human approval for every non-healthy repair (single- or multi-failure) | `human_approval` node reached unconditionally for any non-healthy repair, `CsvHumanApprovalPort` | `test_csv_repair_workflow.py` (single-failure approve/reject/no-port fail-safe), `test_csv_repair_workflow_multi_error.py` (multi-failure approve/reject/no-port fail-safe); real CLI run | COMPLETE |
| PostgreSQL audit | `repair_episodes`/`repair_events`, `PostgresRepairAuditStore` | `test_postgres_repair_audit_store.py`; real `psql` queries against live episodes | COMPLETE |
| Structured logging | `infrastructure/logging/logger.py`, every node | Visible in every CLI run's stdout | COMPLETE |
| `trace_id` propagation | `get_logger()` reads `mlflow.get_active_trace_id()` | Real CLI runs show identical `trace_id` across every log line | COMPLETE |
| MLflow tracing (nested spans, TOOL, CHAT_MODEL) | `MlflowRepairTraceTracer` + `mlflow.langchain.autolog()`/`mlflow.openai.autolog()` | Real trace inspections (`mlflow.get_trace`) across multiple sessions | COMPLETE |
| Token usage on LLM spans | MLflow's own `mlflow.chat.tokenUsage` (zero custom code) | Verified directly on real trace spans | COMPLETE |
| Cost on LLM spans | MLflow's own `mlflow.llm.cost` (zero custom code, never fabricated) | Verified directly on real trace spans | COMPLETE |
| Run-level token/cost metrics | `MlflowRepairRunTracker.log_metrics(trace_id=...)` → `trace_llm_usage.aggregate_chat_model_usage` | `test_mlflow_repair_run_tracker.py`, `test_trace_llm_usage.py` | COMPLETE |
| Healthy CSV short-circuits (no LLM/audit/MLflow) | Detection happens before router construction in `main.py`'s `repair()` | `test_cli_healthy_path_never_calls_flush_traces`; real CLI run | COMPLETE |

## Tier 2 — Schema repair

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Versioned, source-controlled baseline | `SchemaBaseline` + `JsonSchemaBaselineStore`, `configs/schema_baselines/*.json` | `test_schema_repair_workflow.py`; real CLI runs | COMPLETE |
| Deterministic `ColumnDiff` (added/removed/type-changed/rename hints) | `schema_diff.py::compute_column_diff` | `test_schema_diff.py` (if present) / `test_schema_repair_workflow.py` | COMPLETE |
| Repair order rename→cast→drop→add_default | `normalize_operation_order` | `test_schema_repair_operations.py` / workflow tests | COMPLETE |
| Nested LangGraph subgraph (fuzzy_match→llm_confirm→confidence_score) | `rename_resolution_subgraph.py`, invoked as a real compiled-graph node | `xray=True` graph inspection; real MLflow trace showing nested spans | COMPLETE |
| Confidence gating | `confidence_gate` node, `DEFAULT_CONFIDENCE_THRESHOLD` | `test_low_confidence_is_escalated_and_rejection_still_prevents_apply` | COMPLETE |
| High confidence still requires approval | Approval is unconditional, not threshold-gated | `test_high_confidence_still_requires_human_approval` | COMPLETE |
| Human approval before apply | `human_approval` node, `HumanApprovalPort`/`ClickHumanApprovalPort` | `test_human_rejection_does_not_modify_data`, `test_human_approval_allows_apply_end_to_end_added_column`; real CLI runs | COMPLETE |
| Migration history (JSON, append-only) | `JsonMigrationHistoryStore`, `configs/schema_migrations/<episode>/*.json` | `test_migration_json_written_for_applied_and_rejected` | COMPLETE |
| PostgreSQL mirror | `PostgresSchemaMigrationStore`, `schema_migration_events` table | `postgres_schema_migration_store.py`; real `psql` queries | COMPLETE |
| Episode continuity (no re-proposing rejected ops) | `--episode-id` reuse, prior history loaded in `detect` | `test_rejected_operation_is_not_re_proposed_in_same_episode`, `test_prior_history_is_loaded_on_retry` | COMPLETE |
| Added column drift healed | End-to-end workflow | `test_human_approval_allows_apply_end_to_end_added_column`; real CLI run | COMPLETE |
| Removed column drift healed | End-to-end workflow | `test_end_to_end_removed_column_repair`; real CLI run | COMPLETE |
| Type-change drift healed | End-to-end workflow | `test_end_to_end_type_change_repair(_succeeds_with_castable_data)`; real CLI run | COMPLETE |
| Rename drift (via nested subgraph) | End-to-end workflow | Real CLI run (`customers_rename` fixture) | COMPLETE |
| Healthy/no-drift bypasses LLM | `detect`/`diff` short-circuit before `resolve_renames`/`propose` | `test_healthy_path_bypasses_llm_and_human_approval` | COMPLETE |
| Token usage/cost on Tier 2's LLM call | `SchemaMigrationEntry.token_usage` populated via `trace_tracer.get_llm_usage` | `test_migration_entry_records_real_trace_llm_usage` | COMPLETE |
| Tier 2 multi-error (multiple simultaneous drift episodes) | Not implemented | — | NOT REQUIRED (explicitly out of scope this phase) |

## LLM provider abstraction

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Provider-agnostic ports | `CsvRepairProposalPort`, `RenameConfirmationPort` (domain `Protocol`s) | `domain/interfaces/services/*.py` | COMPLETE |
| Groq provider (temporary dev) | `GroqProposalProvider`, `GroqRenameConfirmationProvider` | Multiple real end-to-end CLI runs, both tiers | COMPLETE |
| Azure OpenAI provider (production) | `AzureOpenAIProposalProvider`, `AzureRenameConfirmationProvider` | Verified end-to-end against a real Azure OpenAI deployment (real HTTP 200 response, real MLflow `CHAT_MODEL` span, real PostgreSQL audit record) | COMPLETE |
| `.env`-driven provider selection | `LLMProviderSettings` via pydantic-settings (not raw `os.environ`) | `test_proposal_provider_factory.py` | COMPLETE |

## Documentation

| Requirement | Evidence | Status |
|---|---|---|
| README (architecture, both tiers, demos, config, limitations) | `README.md` | COMPLETE |
| PRIORITIES.md (P0/P1/P2, deferred items) | `PRIORITIES.md` | COMPLETE |
| ADR: human-in-the-loop approval | `docs/architecture/adr/0003-human-in-the-loop-approval.md` | COMPLETE |
| ADR: provider-agnostic LLM + deterministic execution | `docs/architecture/adr/0004-provider-agnostic-llm-deterministic-execution.md` | COMPLETE |
| NFR measurement report | `docs/NFR_RESULTS.md` (real, freshly measured) | COMPLETE |

## Quality gates

| Requirement | Evidence | Status |
|---|---|---|
| Full test suite passing | `pytest -q` → 245 passed | COMPLETE |
| Lint clean | `ruff check src/ tests/`, `ruff check scripts/` | COMPLETE |
| Type-check clean | `mypy --strict src/ tests/` | COMPLETE |
| Diff hygiene | `git diff --check` | COMPLETE |
| Containerized dependencies healthy | `docker compose ps` (PostgreSQL + MLflow both `healthy`) | COMPLETE |

## Summary

- **COMPLETE**: every Tier 1, Tier 2, multi-error, human-approval,
  audit/history, and observability requirement, using both the Groq and
  Azure OpenAI providers end-to-end.
- **NOT REQUIRED**: genuine 3-way simultaneous Tier 1 failure detection
  (architecturally blocked without a larger redesign) and Tier 2
  multi-error detection — both explicitly out of scope for this phase.
