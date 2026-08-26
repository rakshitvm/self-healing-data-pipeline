Self-Healing Data Pipeline — Test Sample Files
================================================

10 CSV files (10 data rows each) covering every Tier 1 and Tier 2 error
type. Every file was verified against the real detector/diff logic
before being added here.

Run each one with the CLI:

  PYTHONPATH=src python -m self_healing_pipeline.interfaces.cli.main repair test_samples/<file>

For the Tier 2 files, use schema-repair or repair-and-process with
--table customers_nochange:

  PYTHONPATH=src python -m self_healing_pipeline.interfaces.cli.main schema-repair customers_nochange test_samples/<file>

TIER 1 (CSV parsing errors)
----------------------------
tier1_wrong_delimiter.csv              -> wrong_delimiter
tier1_wrong_encoding.csv               -> wrong_encoding (real latin-1 bytes, not UTF-8)
tier1_header_detection.csv             -> header_detection (junk title line before the real header)
tier1_engine_selection.csv             -> engine_selection (one row has an extra field)
tier1_single_column_malformation.csv   -> single_column_malformation (space-separated, no delimiter)

TIER 2 (schema drift vs. the "customers_nochange" baseline: id, name, age)
------------------------------------------------------------------------
tier2_added_column.csv       -> ADDED: country
tier2_removed_column.csv     -> REMOVED: age
tier2_renamed_column.csv     -> REMOVED: name / ADDED: full_name (rename hint, similarity 0.62)
tier2_type_changed.csv       -> TYPE CHANGED: age (int64 -> float64)
tier2_no_drift_healthy.csv   -> no drift — confirms Tier 2 doesn't falsely flag a healthy file

Note: the baseline used here is "customers_nochange" (id, name, age) —
one of the original baselines already committed to the repo under
configs/schema_baselines/. Always pass --table customers_nochange for
the Tier 2 files above.
