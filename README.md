# Self-Healing Data Pipeline

## Overview

A self-healing data pipeline that detects and repairs structural data-quality
problems using deterministic detection, an LLM that *proposes* (never
applies) a fix, Pydantic validation, and — for anything data-changing —
mandatory human approval before the fix is applied. Every decision is
recorded in PostgreSQL and/or an append-only JSON history, and every
repair is traced in MLflow.

Two independent capabilities are implemented:

- **Tier 1 — CSV parse-failure self-healing**: detects and repairs
  malformed CSV files (wrong delimiter, wrong encoding, single-column
  malformation, missing/misplaced header, inconsistent field counts),
  including files with more than one of these problems at once.
- **Tier 2 — Schema-drift repair**: detects drift between a
  source-controlled schema baseline and a table's current structure
  (added/removed/renamed/retyped columns) and proposes a repair, always
  gated behind human approval.

## Architecture

Clean/hexagonal architecture (`domain` → `application` → `infrastructure`
→ `interfaces`), dependency inversion throughout — the domain layer knows
nothing about pandas, LangGraph, MLflow, PostgreSQL, Click, or any LLM
provider; everything infrastructure-specific is injected via `typing.Protocol`
ports, wired together in one composition root
(`src/self_healing_pipeline/interfaces/cli/main.py`).

Both tiers are independent LangGraph `StateGraph` workflows:

```
Tier 1: src/self_healing_pipeline/application/orchestration/csv_repair_workflow.py
  sample -> diagnose -> propose -> validate -> human_approval -> apply -> reverify
  human_approval is reached for every non-healthy repair (single- or
  multi-failure) — a healthy file short-circuits before diagnose and
  never reaches it

Tier 2: src/self_healing_pipeline/application/orchestration/schema_repair_workflow.py
  detect -> diff -> resolve_renames (nested subgraph) -> propose -> validate
    -> confidence_gate -> human_approval -> apply -> verify
```

The Tier 2 `resolve_renames` node is a genuinely separate, independently
compiled `StateGraph` (`rename_resolution_subgraph.py`,
`fuzzy_match -> llm_confirm -> confidence_score`) invoked as a single node
of the parent graph — not simulated with a function call; MLflow's own
trace correctly shows its internal nodes nested under the parent.

### Tier 1 — CSV repair

- **Detection** (`infrastructure/csv/local_csv_failure_detector.py`,
  `LocalCsvFailureDetector`): fully deterministic, stdlib-only (`csv.Sniffer`,
  `csv.reader`) plus `chardet` for encoding. `detect_all()` returns every
  applicable failure as a set (backward-compatible `detect()` still
  returns the single highest-priority one). Empirically verified maximum
  simultaneous set: `{WRONG_ENCODING, WRONG_DELIMITER}` or
  `{WRONG_ENCODING, one structural class}` — `csv.Sniffer` cannot
  confidently report a delimiter under the same raggedness that would
  also trigger a structural failure, so a genuine three-way combination
  isn't achievable through this mechanism.
- **Proposal**: `CsvRepairProposalPort` (Groq or Azure OpenAI — see
  Configuration) proposes delimiter/encoding/header_row/engine. For a
  multi-failure episode, the sample is prefixed with a deterministic
  evidence line listing every detected failure so the LLM proposes one
  combined fix. For `WRONG_ENCODING`, a deterministic `chardet` pass
  additionally corrects only the `encoding` field after the LLM
  responds — never overwriting whatever the LLM proposed for the other
  fields.
- **Validation**: `CsvRepairParams` (Pydantic) — a proposal that doesn't
  validate never reaches apply; the workflow retries (bounded) or fails.
- **Human approval**: required before apply for every non-healthy
  repair, single-failure or multi-failure (`human_approval` node) —
  nothing auto-applies.
- **Apply/reverify**: `PandasCsvRepairExecutor` — same validated params
  applied via one `pd.read_csv(...)` call, then independently
  re-verified.

### Tier 2 — Schema repair

- **Baseline**: versioned, source-controlled JSON per table
  (`configs/schema_baselines/<table>.json`).
- **Diff**: deterministic (`application/orchestration/schema_diff.py`) —
  added/removed/type-changed columns, plus rename *hints* from stdlib
  `difflib` name-similarity. The LLM never computes the raw diff.
- **Rename resolution**: the nested subgraph confirms/rejects each rename
  hint via the LLM (`RenameConfirmationPort`) and blends fuzzy +
  LLM confidence into a final score.
- **Repair order**: operations are always normalized into
  `rename -> cast -> drop -> add_default`
  (`domain/value_objects/schema_repair_operations.py`,
  `normalize_operation_order`) regardless of what order they were
  proposed/assembled in.
- **Confidence gating**: below-threshold proposals are flagged
  `escalated` (extra logging/visibility) but — per an explicit
  requirement — **every** proposal, low or high confidence, still
  requires human approval; nothing auto-applies.
- **Human approval**: `HumanApprovalPort`, framework-agnostic (no Click
  in the domain/application layers) — `ClickHumanApprovalPort` is the
  CLI implementation; a rejection never applies and is recorded.
- **History**: every attempt (applied or rejected) is an append-only
  entry in `configs/schema_migrations/<episode_id>/*.json`, mirrored
  best-effort into PostgreSQL (`schema_migration_events` table). On a
  retry with the same `--episode-id`, prior history is loaded and a
  previously-rejected operation is never re-proposed in that episode.

## Human-in-the-loop approval

**Invariant: any non-healthy repair that would modify data requires
explicit human approval before apply.**

```
Healthy:
  detect -> healthy -> exit

Tier 1:
  diagnose -> propose -> validate -> human_approval
      approved -> apply -> reverify -> success
      rejected -> set_rejected, no mutation

Tier 2:
  detect/diff -> propose/resolve -> validate/confidence -> human_approval
      approved -> apply -> verify
      rejected -> no mutation
```

The LLM may analyze, propose, and assign confidence — it may never
authorize a data-changing repair. See
`docs/architecture/adr/0003-human-in-the-loop-approval.md`.

Example prompt:
```
Schema drift detected for table: customers

Diff:
  + country: string
  ~ age: string -> int64

Proposed repair order:
  1. cast age -> int64
  2. add country = 'UNKNOWN'

Confidence: 0.94

Approve repair? [y/N]:
```
Only `y`/`yes` proceeds to apply; anything else (including empty input)
rejects — the file/data is left unchanged, and the rejection is recorded.

## LLM provider abstraction

`CsvRepairProposalPort` / `RenameConfirmationPort` (domain Protocols) —
concrete implementations for Groq (`infrastructure/llm/groq_*.py`,
temporary development provider) and Azure OpenAI
(`infrastructure/llm/azure_*.py`, the intended production provider,
selected via `LLM_PROVIDER=azure_openai`). See ADR 0004 for why the LLM
proposes rather than mutates data directly, and why execution stays
100% deterministic regardless of provider.

**Azure OpenAI: integration exists in code but is NOT yet verified with
real credentials — PENDING, expected tomorrow.** Every real end-to-end
demo run to date has used Groq (`LLM_PROVIDER=groq`), which shares the
same `openai`-SDK-based call pattern MLflow autologs identically.

## Observability

- **Structured logging**: `infrastructure/logging/logger.py`,
  JSON via `structlog`. Every workflow node logs `agent`, `node`,
  `table`, `trace_id`, and a `node_completed`/`repair_completed` event.
- **`trace_id` propagation**: `get_logger()` best-effort reads the
  currently-active MLflow trace ID (`mlflow.get_active_trace_id()`) so
  every node's log line carries the same `trace_id` as the MLflow trace
  and the PostgreSQL/JSON audit record for that episode.
- **MLflow tracing**: `infrastructure/mlflow/mlflow_repair_trace_tracer.py`
  wraps each graph invocation in one outer trace; `mlflow.langchain.autolog()`
  + `mlflow.openai.autolog()` (enabled once, `tracing_setup.py`)
  auto-instrument every LangGraph node, the sampling `TOOL` span, and
  every raw LLM call as a `CHAT_MODEL` span with real prompt/response and
  (from the SDK's own `usage` object) real token counts — with zero
  custom instrumentation code.
- **Token/cost observability**: `infrastructure/mlflow/trace_llm_usage.py`
  sums the real, already-captured `mlflow.chat.tokenUsage`/`mlflow.llm.cost`
  across a trace's `CHAT_MODEL` spans (never fabricated — cost is simply
  omitted if MLflow has no pricing entry for the model). Tier 1 logs the
  result as real MLflow Run metrics (`input_tokens`/`output_tokens`/
  `total_tokens`/`cost_usd`); Tier 2 stores it on the migration history
  entry (`SchemaMigrationEntry.token_usage`).
- **PostgreSQL audit**: `repair_episodes`/`repair_events` (Tier 1, always
  authoritative — MLflow is best-effort observability on top) and
  `schema_migration_events` (Tier 2, best-effort mirror of the JSON
  history, which is authoritative for Tier 2).
- **Elasticsearch + Kibana (centralized log search)**: indexes the
  application's *existing* structured JSON logs — nothing new is
  logged, and `infrastructure/logging/logger.py` is unmodified. This is
  purely additive operational-log search/filtering on top of the same
  logs that already go to stdout and `logs/self_healing_pipeline.log`.
  It does not replace or overlap with MLflow (LLM tracing, spans,
  prompts/responses, token/cost) or PostgreSQL (repair audit/episodes),
  which remain the systems of record for those concerns unchanged. See
  **Log search (Elasticsearch + Kibana)** below.

## Docker setup

```bash
docker compose up -d          # PostgreSQL, self-hosted MLflow, Elasticsearch, Kibana, Filebeat
docker compose ps             # all five should show healthy
PYTHONPATH=src python scripts/provision_kibana.py   # recreate the Kibana dashboard/saved search (idempotent)
```
MLflow UI: http://localhost:5000. PostgreSQL: `localhost:5432` (see
`.env.example` for credentials). Elasticsearch:
http://localhost:9200. Kibana: http://localhost:5601.

## Log search (Elasticsearch + Kibana)

Centralized search/filtering over the application's own structured JSON
logs — a separate concern from MLflow (LLM tracing) and PostgreSQL
(repair audit), which this does not replace or duplicate.

**Why Filebeat, not direct-to-Elasticsearch logging**: the application
already writes structured JSON to `logs/self_healing_pipeline.log`
(`infrastructure/logging/logger.py`, unmodified — stdout logging is
also preserved). The CLI runs on the host, not inside Docker, so the
simplest path that avoids adding an Elasticsearch client (and a new
runtime dependency/coupling) directly into the application's logging
path is to let **Filebeat** tail that existing file and ship it to
Elasticsearch. **Logstash is not used** — Filebeat decodes each
already-JSON line natively (`ndjson` parser, `keys_under_root: true`),
so no separate parsing pipeline is needed.

Two of the application's own JSON field names collide with fields
Filebeat/Elasticsearch reserve for their own metadata and are renamed
on ingest (`docker/filebeat/filebeat.yml`) — the on-disk log format
itself is untouched:
- `event` (a string, e.g. `"node_completed"`) → indexed as `app_event`
  (Elasticsearch's auto-generated template maps `event` as an object
  for Filebeat's own ECS `event.*` fields; a string there is rejected).
- `agent` (a string, e.g. `"CsvRepairAgent"`) → indexed as `app_agent`
  (Filebeat populates its own `agent.*` object with the shipping
  beat's identity, which otherwise silently overwrites the app's value).

**Searchable/filterable fields** (Kibana index pattern
`self-healing-pipeline-logs-*`): `trace_id`, `table`, `level`,
`failure_class`, `failure_classes`, `node`, `app_agent`, `app_event`,
`success`, `applied`, `timestamp` (original app timestamp) plus
Kibana's own `@timestamp`. Token/cost fields are **not** present in
these logs (they are only ever written to MLflow — see above) and are
intentionally not part of this dashboard.

**Kibana objects are version-controlled, not ad hoc.** They're exported
to `docker/kibana/saved_objects.ndjson` (index pattern
`self-healing-pipeline-logs`, 3 visualizations, the dashboard, and the
saved search — real Kibana `_export` output, not hand-written) and
recreated deterministically by `scripts/provision_kibana.py`, which
waits for Kibana to report healthy, then imports the file via the
saved-objects `_import` API with `overwrite=true` (stdlib-only, no new
dependency; safe/idempotent to re-run). This is what makes the
dashboard reproducible from a fresh `docker compose up -d` rather than
existing only in a live Kibana's `.kibana` index.

Provisioned objects: dashboard **"Self-Healing Pipeline — Operational
Log Dashboard"**, with panels for repair volume over time (distinct
`trace_id` count where `app_event:"repair_completed"`), failure-class
distribution (terms on `failure_class`), and outcome breakdown (terms
on `node`, filtered to `set_success`/`set_failure`/`set_rejected`);
plus a separate saved search, **"Repair episode trace explorer"**, for
finding one episode's full log sequence by `trace_id`. MTTR is not
included as an automated panel — the data (per-episode `trace_id` +
per-node `timestamp`) supports a human reading it off the trace
explorer, but a correct automated aggregation needs a scripted metric
this implementation does not ship, rather than risk shipping an
inaccurate one.

```bash
docker compose up -d
# generate at least one real log line first (healthy files produce none):
PYTHONPATH=src python -m self_healing_pipeline.interfaces.cli.main repair <a malformed csv>
curl http://localhost:9200/_cat/indices/self-healing-pipeline-logs-*?v
```
Kibana: http://localhost:5601 → **Dashboard** → "Self-Healing Pipeline —
Operational Log Dashboard", or **Discover** → "Repair episode trace
explorer" and filter by `trace_id`.

## Configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `azure_openai` (production) or `groq` (temporary dev) |
| `AZURE_OPENAI_API_KEY`/`_ENDPOINT`/`_API_VERSION`/`_DEPLOYMENT_NAME` | Azure OpenAI (PENDING credentials) |
| `GROQ_API_KEY`/`GROQ_MODEL` | Groq (dev provider, currently used for all real demos) |
| `DATABASE_URL`, `DB_*` | PostgreSQL connection |
| `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME` | MLflow |
| `LOG_LEVEL`, `LOG_FORMAT` | Structured logging |
| `ELASTICSEARCH_PORT`, `KIBANA_PORT` | Optional `docker-compose.yml` port overrides (default 9200/5601); not read by the application itself |

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # fill in real values
docker compose up -d
```

### Tier 1 — healthy CSV
```bash
printf 'id,name,value\n1,alpha,10\n2,beta,20\n' > /tmp/healthy.csv
PYTHONPATH=src python -m self_healing_pipeline.interfaces.cli.main repair /tmp/healthy.csv
# -> "HEALTHY: ... already parses correctly." — no LLM call, no audit episode, no MLflow run
```

### Tier 1 — each failure class
```bash
# wrong_delimiter
printf 'id;name;value\n1;alpha;10\n2;beta;20\n' > /tmp/f1.csv
# wrong_encoding (Latin-1 bytes)
python3 -c "open('/tmp/f2.csv','wb').write('id,name\n1,café\n2,naïve\n'.encode('latin-1'))"
# header_detection (junk title line)
printf 'Report\nid,name,value\n1,a,10\n2,b,20\n' > /tmp/f3.csv
# engine_selection (missing field — genuinely repairable variant)
printf 'id,name,value\n1,a,10\n2,b\n3,c,30\n' > /tmp/f4.csv
# single_column_malformation
printf 'id:name:age\n1:Alice:30\n2:Bob:25\n' > /tmp/f5.csv

PYTHONPATH=src python -m self_healing_pipeline.interfaces.cli.main repair /tmp/f1.csv
```

### Tier 1 — multi-error
```bash
python3 -c "open('/tmp/multi.csv','wb').write('id;name;age\n1;café;30\n2;naïve;25\n'.encode('latin-1'))"
PYTHONPATH=src python -m self_healing_pipeline.interfaces.cli.main repair /tmp/multi.csv
# detects {wrong_encoding, wrong_delimiter} -> combined proposal -> human approval prompt -> apply -> reverify
```

### Tier 2 — schema repair
```bash
# baseline already exists at configs/schema_baselines/customers_added.json
printf 'id,name,age,country\n1,Alice,30,India\n' > /tmp/customers.csv
PYTHONPATH=src python -m self_healing_pipeline.interfaces.cli.main schema-repair customers_added /tmp/customers.csv
```

### Inspecting results
- **Logs**: stdout, JSON lines — `grep trace_id` to follow one episode across every node.
- **PostgreSQL**: `docker exec self-healing-pipeline-postgres psql -U postgres -d self_healing_pipeline -c "SELECT * FROM repair_episodes ORDER BY started_at DESC LIMIT 5;"`
- **MLflow**: open http://localhost:5000, or `mlflow.get_trace(trace_id)` from the printed `trace_id`.
- **Schema migration history**: `configs/schema_migrations/<episode_id>/*.json`.

## Tests and quality checks

```bash
PYTHONPATH=src pytest -q
ruff check src/ tests/
ruff check scripts/
mypy --strict src/ tests/
```

## Known limitations

- **Azure OpenAI is not yet verified** — code path exists and mirrors the
  already-verified Groq path exactly (same `openai` SDK usage, same
  MLflow autolog behavior), but has not been run against real Azure
  credentials. PENDING.
- A genuine three-way simultaneous Tier 1 failure (encoding + delimiter +
  a structural class) is not achievable with the current
  `csv.Sniffer`-based detector — architecturally explained in
  `local_csv_failure_detector.py`'s module docstring.
- `ENGINE_SELECTION`'s "extra field" (ragged row with *more* fields than
  the header) trigger is not repairable by any `CsvRepairParams` — this
  is a schema-shape limitation (no error-recovery field), not a bug; the
  "missing field" variant is genuinely repairable and is what the CLI
  demo above uses.
- A known, intermittent MLflow trace-tag race can occasionally cause one
  integration test to fail under full-suite load (self-heals on rerun;
  already mitigated with bounded retries, see
  `mlflow_repair_trace_tracer.py`).

## Intentionally deferred

- Multi-error **schema** detection (Tier 2 currently handles one drift
  episode's diff at a time; Tier 1 multi-error CSV detection is
  implemented — see above).
- Tier 3/Tier 4 features (not scoped for this project phase).
- A UI/API approval mechanism (the approval ports are already
  framework-agnostic Protocols specifically so a CLI prompt can be
  swapped later without touching the workflow).

## Project structure

See [docs/architecture/overview.md](docs/architecture/overview.md) and
`docs/architecture/adr/` for architecture decision records.

## License

TBD
