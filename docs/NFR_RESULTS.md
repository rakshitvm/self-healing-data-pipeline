# Non-Functional Requirements — Measurement Results

All numbers below are real, freshly measured on this machine during this
session — none are invented or estimated. Reproduce with:

```bash
docker compose up -d
PYTHONPATH=src .venv/bin/python scripts/measure_nfr.py
PYTHONPATH=src pytest -q
docker compose ps
```

## Environment

- Machine: local development container (Linux, this repository's `.venv`)
- Python: 3.12
- `LLM_PROVIDER=groq` (measurement B makes a real network call to Groq)
- MLflow tracing enabled (`enable_tracing`), PostgreSQL + MLflow both
  running via `docker compose`

## A. Healthy-path overhead

**What this measures, precisely**: an **end-to-end comparison of the
whole CLI process against a bare in-process parser** — it is **not** an
isolated measurement of MLflow/instrumentation overhead in isolation,
and the numbers below must not be read as "tracing costs 726%." The two
sides are not doing comparable amounts of work: the baseline is a single
Python statement inside an already-running interpreter, while the
"instrumented" side is a brand-new `python -m ...` subprocess that has
to start the interpreter, import pandas/LangGraph/MLflow/pydantic-settings/
click/etc., load configuration, and only then run detection — before any
CSV parsing happens at all. The comparison is real and reproducible, but
it conflates process/import startup cost with actual detection work; it
does **not** isolate what MLflow tracing specifically costs.

**Methodology**: (1) baseline — read a real ~100MB CSV with stdlib
`csv.reader`, counting rows, in-process, no other code involved; (2)
full CLI — the complete `repair` CLI command (fresh Python subprocess
per invocation) against the same file, which is healthy, so **LLM calls
= 0** (verified: the CLI command short-circuits at the detector call,
before `build_production_error_router()` is ever constructed — see
`interfaces/cli/main.py`'s `repair()`). Multiple iterations via
`scripts/measure_nfr.py`.

**Latest measured run:**

| | Baseline (`csv.reader`, in-process) | Full CLI (`repair`, healthy, fresh subprocess) |
|---|---|---|
| Mean | 1.7299s | 14.2866s |

**Reported difference: 725.9%** — this is the end-to-end CLI-process-vs.
bare-parser gap described above, not an isolated tracing/MLflow overhead
figure. No breakdown attributing this gap specifically to MLflow vs.
other imports (pandas, LangGraph, pydantic-settings, click) was measured
in this run, so no such split is claimed here.

**Interpretation**: the large relative gap is consistent with Python
**process startup and import cost** (a fresh subprocess re-imports
pandas/LangGraph/MLflow/etc. every invocation) dominating over the
actual per-byte CSV scan, which is of the same order of magnitude as the
baseline. This is a real characteristic of the CLI's current
one-process-per-invocation design, not a claim about what any single
component (including MLflow) contributes in isolation. A long-lived
process (e.g. a service that imports once and handles many files) would
not repay this import cost on every file — that scenario was not
measured here.

## B. Repair latency (end-to-end)

**Methodology**: full CLI invocation (`repair`) against the existing
`wrong_delimiter` fixture, real Groq call included — explicitly
end-to-end latency (diagnosis → proposal → validation → apply →
reverify), not isolated component timing.

**Latest measured run:**

| | Value |
|---|---|
| Mean | 7.9579s |

**Limitation, stated explicitly per the requirement**: this includes
real network latency to Groq (the currently-configured provider) and is
explicitly an end-to-end figure, not this codebase's own processing time
in isolation. Azure OpenAI's latency profile is unverified (PENDING
credentials) and may differ.

**Note on repeatability**: `scripts/measure_nfr.py` has been run more
than once during this project; absolute values differ run to run (this
section reports the latest run's means) due to normal machine load and
real network variance to Groq — see "Limitations of this measurement"
below. Only mean values are reported here because per-iteration/stdev/p95
figures were not captured for this specific latest run; do not treat
earlier per-iteration breakdowns in this project's history as still
current.

## C. Reliability

**Primary evidence: the existing automated test suite**, run to
completion immediately before this report was written:

```
PYTHONPATH=src pytest -q
222 passed in 32.14s
```

- Total: 222
- Passed: 222
- Failed: 0
- Flaky/pre-existing known issue: `tests/integration/test_mlflow_tracing_smoke.py::test_real_repair_invocation_creates_a_trace_with_expected_id_and_tags` has intermittently failed under full-suite load in earlier sessions this project (an MLflow async trace-export/tag race, already mitigated with bounded retries in `mlflow_repair_trace_tracer.py`) — it did **not** fail in this run, but is documented here rather than silently omitted, since it is a known, real, occasionally-reproducible condition, not a hypothetical one.

## D. Containerization

```
$ docker compose ps
NAME                              STATUS
self-healing-pipeline-mlflow      Up About an hour (healthy)
self-healing-pipeline-postgres    Up About an hour (healthy)
```

Both required services report `healthy` via Docker's own healthcheck.

## Limitations of this measurement

- Single machine, single run per measurement — not a statistically
  rigorous benchmark suite (explicitly out of scope per the time
  constraint for this phase); numbers are indicative, not SLA-grade.
- Measurement B depends on real third-party (Groq) network conditions
  at the time of the run and will vary run to run.
- The 100MB fixture is synthetic (generated by `scripts/measure_nfr.py`),
  not a real production file — chosen only to give a reproducible,
  sizeable input for the overhead comparison.
- Azure OpenAI latency/cost characteristics are not measured here —
  PENDING credentials.
