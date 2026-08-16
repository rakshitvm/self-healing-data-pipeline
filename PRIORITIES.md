# Project Priorities

## P0 — Core self-healing functionality (Tier 1)

- Deterministic, local failure detection (`LocalCsvFailureDetector`) —
  no LLM involved in deciding *whether* or *what* is wrong.
- LLM structured proposal (`CsvRepairProposalPort`, Groq/Azure OpenAI) —
  the LLM proposes, never applies.
- Pydantic validation (`CsvRepairParams`) — an invalid proposal never
  reaches apply.
- Repair + reverify (`PandasCsvRepairExecutor`) — independent
  post-apply verification, not just "no exception raised."
- PostgreSQL audit (`repair_episodes`/`repair_events`) — authoritative
  record of every attempt.
- Observability — structured logs with `trace_id` propagation, MLflow
  tracing (LangGraph/tool/LLM spans), MLflow Run-level token/cost
  metrics.
- Human approval for anything data-changing, including the simplest
  single-failure case (see P1 — extended further to also cover
  multi-failure Tier 1).

**Status: COMPLETE** (both Azure OpenAI and Groq provider paths are
implemented and have been verified end-to-end against real credentials).

## P1 — Tier 2 schema repair + hardening

- Tier 2 schema-drift detection, versioned baselines, deterministic
  `ColumnDiff`.
- Rename resolution via a genuine nested LangGraph subgraph
  (`fuzzy_match -> llm_confirm -> confidence_score`).
- Confidence gating — below-threshold proposals flagged `escalated`;
  human approval required regardless of confidence (explicit
  requirement — high confidence does not bypass approval).
- Append-only migration history (source-controlled JSON, PostgreSQL
  mirror), episode continuity (rejected operations not re-proposed in
  the same episode).
- Multi-error Tier 1 CSV detection — a single file's multiple
  independent failures (e.g. wrong encoding + wrong delimiter) detected
  and repaired together, gated behind human approval.
- Additional hardening: encoding-aware CSV sampling, deterministic
  repair-order enforcement (`rename -> cast -> drop -> add_default`),
  MLflow trace-tag retry (mitigates a known async-export race).

**Status: COMPLETE**

## P2 — Stretch (time-permitting)

- UI/API-based human approval (deferred — approval ports are already
  framework-agnostic Protocols, ready for this without workflow
  changes).
- Tier 2 multi-error (multiple simultaneous schema-drift episodes) —
  deferred; only Tier 1 multi-error was in scope.
- Tier 3/Tier 4 features — out of scope for this project phase.

**Status: NOT STARTED (intentionally deferred, not blocking)**

## Explicitly deferred, and why

| Deferred item | Why |
|---|---|
| Tier 2 multi-error detection | Out of scope for this phase — Tier 1 multi-error was the explicit requirement |
| Genuine 3-way Tier 1 simultaneous failure (encoding+delimiter+structural) | Not achievable without changing the delimiter-detection algorithm itself (`csv.Sniffer` requires the same consistency a structural failure would violate) — would be a larger redesign than justified for this phase |
| UI/API approval mechanism | CLI approval satisfies the human-in-the-loop requirement now; ports were deliberately kept framework-agnostic so this can be added later without touching the workflow |
| Tier 3/Tier 4 | Not requested for this project phase |
