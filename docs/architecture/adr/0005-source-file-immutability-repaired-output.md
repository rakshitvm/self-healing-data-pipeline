# ADR 0005: Tier 1 source-file immutability — repaired output written separately

## Context

Tier 1's `apply` step (`PandasCsvRepairExecutor.execute`) originally read
the source CSV with the validated, human-approved repair parameters and
then wrote the corrected result **back to the same path** — the same
approach Tier 2's `PandasSchemaExecutor.execute` still uses today (see
"Consequences" below). This is a reasonable-sounding default, but it has
a real operational cost: the moment `apply` runs, the original ingested
file is gone. If the repair itself is subtly wrong — a plausible
delimiter guess that happens to be incorrect for a few rows, an encoding
correction that mangles a handful of characters — there is no way to
recover the original bytes to try again, compare against, or hand off
to a human for closer inspection. A live demo surfaced this directly: an
approved repair's own audit message read "Repaired and rewrote
`<source path>`," which is an accurate description of what the code did,
but not what the project actually wants for a file whose provenance may
matter downstream.

## Decision

- **Tier 1's `apply` never opens the source file for writing.** It reads
  the source with the validated `CsvRepairParams` and, only if that
  produces a sane multi-column result, writes a **separate** file at
  `<source directory>/repaired/<source filename>` — creating the
  `repaired/` directory if it does not yet exist.
- The write is atomic: a temp file is created in the same target
  directory and swapped into place with `os.replace`, so a failure
  partway through a write can never leave a corrupted or partial output
  file, and can never touch the source under any circumstance (the
  source was never opened for writing in the first place).
- If the computed output path would ever resolve to the same path as
  the source (a defense-in-depth check, not something reachable through
  the normal `<dir>/repaired/<filename>` construction), `execute()`
  refuses to write anything and reports failure rather than risk
  overwriting the source.
- `reverify` independently re-reads the **repaired output** file — never
  the source, and never by re-applying the prescription again.
- `RepairResult` and the PostgreSQL audit payload (`apply`/`reverify`
  events) both record `source_path` and `output_path` explicitly, so the
  audit trail never implies the source was rewritten when it wasn't.
- The CLI (`repair` command) prints `source_path` and `output_path`
  alongside the existing `success`/`applied`/`prescription` fields.

## Alternatives considered

- **Keep rewriting the source in place, rely on the PostgreSQL audit
  trail for recovery.** Rejected — the audit trail records *what
  happened*, not the original bytes; it cannot reconstruct a source file
  that was overwritten, and "the data is technically recoverable from a
  backup" is not the same guarantee as "the source was never touched."
- **Write the repaired output next to the source with a suffix (e.g.
  `<name>.repaired.csv`) instead of a `repaired/` subdirectory.**
  Considered, but a subdirectory keeps a directory listing of the source
  location uncluttered when repeated repairs are run, and makes "is this
  file a repair output" a structural property (its parent directory)
  rather than a filename convention a caller could collide with.
- **Version every repaired output (timestamped filenames) so repeated
  repairs of the same source don't overwrite each other's output.**
  Rejected for now — out of scope for the requirement that triggered
  this change (source immutability), and would add versioning/retention
  semantics that weren't asked for. Repeated repairs of the same source
  currently overwrite the previous `repaired/<filename>` (last-write-
  wins); this is recorded as a known limitation (README §25) rather
  than silently accepted as a non-issue.

## Consequences

- The source file is provably unchanged after any Tier 1 repair attempt
  — approved, rejected, or failed — asserted directly in tests
  (`test_pandas_csv_repair_executor.py`,
  `test_csv_repair_workflow.py`), not just inferred from code review.
- A caller (or a human reviewing a repair) can always diff the repaired
  output against the original source, since both still exist.
- **This decision has not been extended to Tier 2.**
  `PandasSchemaExecutor.execute` still mutates its target file in place,
  the same way Tier 1 used to. This is a genuine, currently-accepted
  asymmetry between the two tiers (see README §12, §25) — applying the
  same pattern to Tier 2 is recorded as a logical next improvement
  (README §26), not implemented here, since doing so was outside the
  scope of the change that produced this ADR.
- Repeated Tier 1 repairs of the same source file share one
  `repaired/<filename>` output path; no output versioning exists.
