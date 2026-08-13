# ADR 0003: Human-in-the-loop approval before risky repairs

## Context

Tier 1 CSV repair and Tier 2 schema repair both use an LLM to propose a
fix for a detected problem. A proposal, once validated, could in
principle be applied automatically — the workflow already validates it
structurally (`CsvRepairParams`/`SchemaRepairOperation`) before it's
ever eligible to apply. But structural validity does not mean the
proposal is *correct*: an LLM can validly propose a wrong encoding, a
wrong rename pairing, or a wrong default value, and Tier 2 in particular
executes data-*mutating* operations (rename/cast/drop/add_default)
directly against the file.

An explicit product requirement states: the LLM may analyze, propose,
recommend, and assign confidence — it may never be the thing that
authorizes a data-changing repair, regardless of how confident it (or a
deterministic confidence score) claims to be.

## Decision

- **Tier 2**: every repair prescription, unconditionally, passes through
  a `human_approval` node before `apply` — including high-confidence
  ones. `confidence_gate` still computes and logs a real confidence
  score and flags low-confidence proposals as `escalated` for extra
  visibility, but this only affects *logging*, not whether approval is
  requested — approval is requested either way.
- **Tier 1**: a single-failure repair auto-applies exactly as it always
  has (unchanged, low-risk, single well-understood parameter change).
  A genuine **multi-failure** episode (more than one independent Tier 1
  dimension repaired at once via one combined prescription) routes
  through a new `human_approval` node before `apply`, mirroring Tier 2's
  gate.
- Rejection is a first-class terminal outcome (`RepairEpisodeStatus.REJECTED`
  / `SchemaRepairStatus.REJECTED`): `apply` is never called, the
  file/data is provably unchanged (asserted directly in tests, not just
  inferred), and the rejection — including the full proposal and
  confidence that was rejected — is recorded in the audit/history trail
  exactly like an applied repair would be.
- The approval mechanism is a framework-agnostic `Protocol`
  (`HumanApprovalPort` for Tier 2, `CsvHumanApprovalPort` for Tier 1) —
  the domain/application layers depend only on the Protocol; a
  `ClickHumanApprovalPort`/`ClickCsvHumanApprovalPort` in
  `interfaces/cli/` is the only place Click is imported for this
  purpose, so a UI/API-based approval mechanism can be substituted later
  with zero change to the workflow.
- Omitting the approval port entirely (or reaching the gate without one
  wired in) fails **safe**: it's treated as an automatic rejection, never
  a silent auto-apply.

## Alternatives considered

- **Auto-apply above a confidence threshold.** Rejected — explicitly
  contradicts the requirement that high confidence must not bypass
  approval; also, confidence here is a blend of deterministic
  similarity and one LLM's self-reported number, not a guarantee.
- **Auto-apply everything, audit-only (no gate).** Rejected — Tier 2
  operations mutate files directly; an unattended wrong rename or wrong
  default is a real data-integrity risk with no recovery path in this
  architecture (no automatic rollback).
- **A single, shared approval Protocol for both tiers.** Considered, but
  Tier 1's prescription (`CsvRepairParams`) and Tier 2's
  (`ColumnDiff`/`SchemaRepairPrescription`) are structurally unrelated
  domain models; forcing them through one shared request type would
  have coupled the two tiers' data models together for no real benefit.
  Two small, parallel Protocols were judged cleaner than one contorted
  one.

## Consequences

- Every multi-failure Tier 1 repair and every Tier 2 repair requires an
  interactive step — there is currently no fully unattended path for
  these cases (single-failure Tier 1 remains unattended, as before).
- The audit/history trail is richer: a rejected proposal is fully
  recorded (diff, prescription, confidence), not discarded, which is
  valuable for understanding *why* something needed a human and what
  was decided.
- Adding a non-CLI approval channel later (UI, Slack, API) requires only
  a new `HumanApprovalPort`/`CsvHumanApprovalPort` implementation — no
  change to either workflow.
