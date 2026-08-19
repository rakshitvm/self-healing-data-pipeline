"""Concrete `CloudIntegrationPort`: upload a repaired file to Azure Blob
Storage, then (only if the upload succeeded) trigger a Databricks job
with the resulting blob URL.

Composes `AzureBlobSasUploader` and `DatabricksJobTrigger` — this class
adds no HTTP logic of its own, only sequencing and the "never raise"
contract `CloudIntegrationPort` requires. It never opens the local file
for writing and never deletes it; a failure at either step simply stops
the sequence and reports why, leaving the local repaired file untouched.
"""

from __future__ import annotations

from self_healing_pipeline.domain.interfaces.services.cloud_integration_port import (
    CloudIntegrationOutcome,
)
from self_healing_pipeline.infrastructure.cloud.azure_blob_uploader import AzureBlobSasUploader
from self_healing_pipeline.infrastructure.cloud.databricks_job_trigger import DatabricksJobTrigger


class CloudIntegrationService:
    """Concrete `CloudIntegrationPort`. Construct with `uploader=None`
    and/or `trigger=None` to skip either half — `process()` still never
    raises and still reports an honest outcome either way."""

    def __init__(
        self,
        *,
        uploader: AzureBlobSasUploader | None,
        trigger: DatabricksJobTrigger | None,
    ) -> None:
        self._uploader = uploader
        self._trigger = trigger

    def process(self, *, local_file_path: str) -> CloudIntegrationOutcome:
        if self._uploader is None:
            return CloudIntegrationOutcome(
                uploaded=False,
                message="Azure Storage is not configured — cloud upload skipped.",
            )

        upload_outcome = self._uploader.upload(local_file_path)
        if not upload_outcome.success:
            return CloudIntegrationOutcome(
                uploaded=False,
                upload_error=upload_outcome.error,
                message="Azure Blob upload failed; the local repaired file is unaffected.",
            )

        if self._trigger is None:
            return CloudIntegrationOutcome(
                uploaded=True,
                cloud_path=upload_outcome.blob_url,
                message="Uploaded to Azure Blob Storage. Databricks is not configured — job trigger skipped.",
            )

        trigger_outcome = self._trigger.trigger(cloud_path=upload_outcome.blob_url or "")
        if not trigger_outcome.success:
            return CloudIntegrationOutcome(
                uploaded=True,
                cloud_path=upload_outcome.blob_url,
                trigger_error=trigger_outcome.error,
                message="Uploaded to Azure Blob Storage, but the Databricks job trigger failed.",
            )

        return CloudIntegrationOutcome(
            uploaded=True,
            cloud_path=upload_outcome.blob_url,
            triggered=True,
            databricks_run_id=trigger_outcome.run_id,
            message="Uploaded to Azure Blob Storage and triggered the Databricks job.",
        )
