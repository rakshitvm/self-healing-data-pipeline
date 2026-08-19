"""Azure Blob Storage upload via a container-scoped SAS token.

Deliberately stdlib-only (`urllib.request`) — no new dependency
(`azure-storage-blob`) is required. A SAS-token-authorized "Put Blob"
request needs no request signing at all: the SAS token itself, appended
to the URL as a query string, *is* the authorization — see Azure's own
REST API documentation for "Put Blob"
(https://learn.microsoft.com/rest/api/storageservices/put-blob). This
keeps the whole adapter to one small, auditable HTTP call.

This module never raises: every failure mode (missing file, network
error, non-2xx response) is caught and reported on the returned
`UploadOutcome`, matching this project's existing best-effort
observability contracts (`TrackingOutcome`, `CsvExecutionOutcome`).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from pydantic import BaseModel, ConfigDict

# A stable, GA Azure Storage REST API version. SAS tokens generated with
# a specific `sv=` (signed version) query parameter take precedence over
# this header; it is sent defensively for SAS tokens that omit `sv`.
_AZURE_STORAGE_API_VERSION = "2021-08-06"


class UploadOutcome(BaseModel):
    """Result of one best-effort blob upload attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    success: bool
    blob_url: str | None = None
    error: str | None = None


class AzureBlobSasUploader:
    """Uploads a single local file to Azure Blob Storage using a
    container-scoped SAS URL. Read-only with respect to the local file —
    it is only ever opened for reading.
    """

    def __init__(self, *, container_url: str, sas_token: str) -> None:
        """`container_url` is the plain container endpoint, e.g.
        `https://<account>.blob.core.windows.net/<container>` — no query
        string. `sas_token` is the SAS query string (with or without a
        leading `?`); it is only ever used to build the upload request
        URL, never logged, never included in the `blob_url` this class
        reports back (that field is the credential-free blob URL, safe
        to pass on to a downstream job).
        """
        self._container_url = container_url.rstrip("/")
        self._sas_token = sas_token[1:] if sas_token.startswith("?") else sas_token

    def upload(self, local_file_path: str, *, blob_name: str | None = None) -> UploadOutcome:
        """Upload `local_file_path` as a block blob. `blob_name` defaults
        to the local file's own name (no directory structure implied)."""
        source = Path(local_file_path)
        try:
            data = source.read_bytes()
        except OSError as exc:
            return UploadOutcome(success=False, error=f"Failed to read {local_file_path!r}: {exc}")

        name = blob_name or source.name
        public_blob_url = f"{self._container_url}/{name}"
        request_url = f"{public_blob_url}?{self._sas_token}"

        request = urllib.request.Request(
            request_url,
            data=data,
            method="PUT",
            headers={
                "x-ms-blob-type": "BlockBlob",
                "x-ms-version": _AZURE_STORAGE_API_VERSION,
                "Content-Type": "text/csv",
                "Content-Length": str(len(data)),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if 200 <= response.status < 300:
                    return UploadOutcome(success=True, blob_url=public_blob_url)
                return UploadOutcome(
                    success=False, error=f"Unexpected status {response.status} from Azure Blob Storage"
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:500]
            return UploadOutcome(success=False, error=f"HTTP {exc.code} from Azure Blob Storage: {body}")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return UploadOutcome(success=False, error=f"{type(exc).__name__}: {exc}")
