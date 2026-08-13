# ADR 0004: Provider-agnostic LLM abstraction with deterministic repair execution

## Context

Both tiers need an LLM for the one task that's genuinely a judgment
call (proposing repair parameters; confirming a rename candidate). The
production LLM provider is Azure OpenAI; Groq is a temporary
development-only stand-in used until Azure credentials are available.
Everything else in a repair — detecting the problem, validating a
proposal, applying it, and re-verifying the result — is either already
deterministic or *should* be, since it directly affects real data and
must be exactly reproducible and auditable.

## Decision

- **The LLM only proposes; it never executes.** `CsvRepairProposalPort.propose()`
  and `RenameConfirmationPort.confirm()` both return a plain, unvalidated
  payload (`dict[str, Any]` / a small `RenameConfirmation` value object)
  — never a side effect, never a direct call into pandas or the
  filesystem. Both ports are `typing.Protocol`s in the domain layer with
  zero import of `openai`, `mlflow`, or any provider-specific type.
- **Provider selection is a small, isolated factory**
  (`infrastructure/llm/proposal_provider_factory.py`,
  `rename_confirmation_provider_factory.py`), reading `LLM_PROVIDER` via
  the same `pydantic-settings` mechanism as every other setting (not raw
  `os.environ`, so `.env` is honored without requiring the shell to
  separately export it — a real bug this project hit and fixed). Groq
  and Azure OpenAI implementations
  (`groq_proposal_provider.py`/`azure_openai_proposal_provider.py`, and
  the rename-confirmation equivalents) are structurally identical:
  both use the same `openai` Python SDK (`OpenAI` vs `AzureOpenAI`
  clients) with the same prompt-building and same defensive
  "never raise, return an empty/rejecting result" contract — so MLflow's
  `mlflow.openai.autolog()` instruments both identically, and switching
  providers requires no change anywhere else in the system.
- **Deterministic facts are computed deterministically, never left to
  the LLM.** The raw structural diff (`compute_column_diff`), rename
  *candidate* generation (`compute_rename_hints`, stdlib `difflib`), and
  — since it's directly recoverable from the file's actual bytes —
  `WRONG_ENCODING`'s correct encoding (via `chardet`, run before the LLM
  call for prompt evidence, and again after to deterministically correct
  just that one field of the LLM's response) are all pure functions with
  no LLM involvement. The LLM is only asked questions that genuinely
  require judgment: *is this really a rename*, and *what should the
  repair parameters be given the evidence*.
- **Every proposal is validated before it can do anything.**
  `CsvRepairParams`/`SchemaRepairOperation` (Pydantic) reject a
  structurally invalid proposal outright — it never reaches `apply`, and
  the workflow retries (bounded) or fails cleanly.
- **Execution order is enforced by code, never by the LLM.**
  `normalize_operation_order` always sorts Tier 2 operations into
  `rename -> cast -> drop -> add_default`, regardless of what order the
  LLM/diff assembled them in.

## Why the LLM proposes rather than directly mutates data

An LLM call is non-deterministic (even at `temperature=0`, provider
behavior can vary run to run) and unauditable in isolation — "trust the
model's judgment" is not an acceptable basis for changing production
data or an ingested file. Keeping the LLM strictly on the "propose"
side of a hard validate/apply boundary means every actually-executed
change is fully deterministic, reproducible from its recorded
prescription, and independently verifiable (`reverify`/`verify`) without
needing to re-ask the model anything.

## Alternatives considered

- **Let the LLM call the executor/pandas directly (agentic tool-calling
  all the way through).** Rejected — removes the validate boundary
  entirely; a malformed or malicious-looking LLM response could directly
  manipulate data with no structural check in between.
- **A single combined LLM+execution provider per vendor.** Rejected —
  would duplicate deterministic logic (diff, encoding detection, order
  normalization) per provider and make provider swaps touch far more
  code than the factory + two small provider classes this design needs.
- **Have the LLM compute the diff/rename candidates itself.** Rejected
  — these are cheap, exact, reproducible computations; asking an LLM to
  approximate a `set` difference or a string-similarity ratio would be
  strictly worse (slower, non-deterministic, unauditable) for zero
  benefit, and would violate "prefer deterministic logic wherever a fact
  can be established deterministically."

## Consequences

- Adding a third LLM provider requires only one new class implementing
  the existing Protocol plus a factory branch — no workflow change.
- Every applied repair is fully explainable from its stored prescription
  alone, without needing the original LLM call to be reproducible.
- The system's correctness does not depend on LLM determinism or
  provider-specific behavior — only its *proposal quality* does, which
  is exactly the piece validation, confidence gating, and human approval
  exist to guard.
