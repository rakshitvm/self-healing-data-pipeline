"""Optional, additive cloud-integration adapters (Azure Blob Storage
upload + Databricks job trigger) — see `cloud_integration_service.py`.

Confined to this package: no domain model, `ErrorRouter`,
`CsvRepairAgent`, or `application/orchestration` module imports anything
here. Nothing in the existing `repair`/`schema-repair` commands or the
LangGraph workflows depends on this package at all — it is only ever
constructed by the new, separate `repair-and-process` CLI command.
"""
