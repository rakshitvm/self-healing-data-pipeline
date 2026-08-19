"""Tests for the optional cloud-integration adapters (Azure Blob upload +
Databricks job trigger + the `CloudIntegrationService` that composes them).

`AzureBlobSasUploader` and `DatabricksJobTrigger` are exercised against a
local fake HTTP server (the same `http.server.HTTPServer` background-thread
pattern used by `tests/integration/test_mlflow_tracing_smoke.py`'s
`fake_openai_server` fixture) so the actual HTTP request construction is
genuinely tested, not merely mocked. No real Azure/Databricks credentials
or network access are required or used.
"""

import http.server
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from self_healing_pipeline.infrastructure.cloud.azure_blob_uploader import AzureBlobSasUploader
from self_healing_pipeline.infrastructure.cloud.cloud_integration_service import (
    CloudIntegrationService,
)
from self_healing_pipeline.infrastructure.cloud.databricks_job_trigger import DatabricksJobTrigger


class _RecordingHandler(http.server.BaseHTTPRequestHandler):
    """Records every request it receives and replies according to
    `status_code`/`response_body`, both set on the class before the server
    starts (simplest way to configure behavior per-test with a stdlib
    `HTTPServer`, matching this file's other fake-server tests)."""

    received: list[dict[str, Any]] = []
    status_code: int = 200
    response_body: bytes = b"{}"

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        type(self).received.append(
            {
                "method": self.command,
                "path": self.path,
                # header names are case-insensitive over HTTP (RFC 7230 §3.2);
                # urllib.request re-cases what it sends (e.g. "X-ms-blob-type"),
                # so compare case-insensitively rather than assuming exact casing.
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
            }
        )
        self.send_response(type(self).status_code)
        self.send_header("Content-Length", str(len(type(self).response_body)))
        self.end_headers()
        self.wfile.write(type(self).response_body)

    def do_PUT(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def log_message(self, fmt: str, *args: object) -> None:
        pass


@pytest.fixture
def fake_server() -> Iterator[tuple[str, type[_RecordingHandler]]]:
    _RecordingHandler.received = []
    _RecordingHandler.status_code = 200
    _RecordingHandler.response_body = b"{}"
    server = http.server.HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", _RecordingHandler
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestAzureBlobSasUploader:
    def test_uploads_file_contents_with_sas_token_in_query_string(
        self, tmp_path: Path, fake_server: tuple[str, type[_RecordingHandler]]
    ) -> None:
        """The actual PUT still goes to the plain (SAS-authenticated)
        endpoint unchanged; only the *returned* `blob_url` is now a
        `wasbs://` URI — see `TestWasbsCloudPath` for that in isolation."""
        base_url, handler = fake_server
        netloc = urlsplit(base_url).netloc
        file_path = _write(tmp_path, "repaired.csv", "id,name\n1,alpha\n")
        uploader = AzureBlobSasUploader(
            container_url=f"{base_url}/mycontainer", sas_token="sv=2021&sig=fake-signature"
        )

        outcome = uploader.upload(file_path)

        assert outcome.success is True
        assert outcome.blob_url == f"wasbs://mycontainer@{netloc}/repaired.csv"
        assert "sig=fake-signature" not in (outcome.blob_url or "")  # credential-free result
        assert len(handler.received) == 1
        request = handler.received[0]
        assert request["method"] == "PUT"
        assert request["path"] == "/mycontainer/repaired.csv?sv=2021&sig=fake-signature"
        assert request["headers"]["x-ms-blob-type"] == "BlockBlob"
        assert request["body"] == b"id,name\n1,alpha\n"

    def test_strips_leading_question_mark_from_sas_token(
        self, tmp_path: Path, fake_server: tuple[str, type[_RecordingHandler]]
    ) -> None:
        base_url, handler = fake_server
        file_path = _write(tmp_path, "repaired.csv", "id\n1\n")
        uploader = AzureBlobSasUploader(container_url=f"{base_url}/c", sas_token="?sv=2021&sig=x")

        outcome = uploader.upload(file_path)

        assert outcome.success is True
        assert handler.received[0]["path"] == "/c/repaired.csv?sv=2021&sig=x"

    def test_non_2xx_response_is_reported_as_failure_not_raised(
        self, tmp_path: Path, fake_server: tuple[str, type[_RecordingHandler]]
    ) -> None:
        base_url, handler = fake_server
        handler.status_code = 403
        handler.response_body = b"AuthenticationFailed"
        file_path = _write(tmp_path, "repaired.csv", "id\n1\n")
        uploader = AzureBlobSasUploader(container_url=f"{base_url}/c", sas_token="sv=x")

        outcome = uploader.upload(file_path)

        assert outcome.success is False
        assert outcome.blob_url is None
        assert outcome.error is not None and "403" in outcome.error

    def test_missing_local_file_is_reported_as_failure_not_raised(self, tmp_path: Path) -> None:
        uploader = AzureBlobSasUploader(
            container_url="http://127.0.0.1:1/c", sas_token="sv=x"
        )

        outcome = uploader.upload(str(tmp_path / "does_not_exist.csv"))

        assert outcome.success is False
        assert outcome.error is not None

    def test_unreachable_host_is_reported_as_failure_not_raised(self, tmp_path: Path) -> None:
        file_path = _write(tmp_path, "repaired.csv", "id\n1\n")
        # port 1 is a reserved, non-listening port
        uploader = AzureBlobSasUploader(container_url="http://127.0.0.1:1/c", sas_token="sv=x")

        outcome = uploader.upload(file_path)

        assert outcome.success is False
        assert outcome.error is not None


class TestWasbsCloudPath:
    """Pure tests of the HTTPS container URL -> `wasbs://` cloud-path
    transform, isolated from the network — see Cell 4 of
    `databricks_medallion_pipeline` for why Spark on Databricks needs this
    scheme rather than a plain HTTPS URL."""

    def test_converts_realistic_https_container_url_to_wasbs(self) -> None:
        uploader = AzureBlobSasUploader(
            container_url="https://myaccount.blob.core.windows.net/mycontainer",
            sas_token="sv=2021-08-06&sig=fake-signature",
        )

        cloud_path = uploader._wasbs_url("repaired.csv")

        assert cloud_path == "wasbs://mycontainer@myaccount.blob.core.windows.net/repaired.csv"
        assert "sig=" not in cloud_path
        assert "sv=" not in cloud_path

    def test_handles_trailing_slash_on_container_url(self) -> None:
        uploader = AzureBlobSasUploader(
            container_url="https://myaccount.blob.core.windows.net/mycontainer/",
            sas_token="sv=x",
        )

        cloud_path = uploader._wasbs_url("repaired.csv")

        assert cloud_path == "wasbs://mycontainer@myaccount.blob.core.windows.net/repaired.csv"

    def test_preserves_sovereign_cloud_storage_suffix(self) -> None:
        """The authority is reused verbatim from `container_url` rather
        than a hardcoded "blob.core.windows.net", so this also works for
        e.g. Azure Government storage accounts."""
        uploader = AzureBlobSasUploader(
            container_url="https://myaccount.blob.core.usgovcloudapi.net/mycontainer",
            sas_token="sv=x",
        )

        cloud_path = uploader._wasbs_url("repaired.csv")

        assert cloud_path == "wasbs://mycontainer@myaccount.blob.core.usgovcloudapi.net/repaired.csv"


class TestDatabricksSettingsNotebookParamNameDefault:
    """The Databricks job's parameter is named `input_path` (per the
    Databricks Job configuration) — the app must default to that name
    while still allowing an explicit override."""

    def test_default_notebook_param_name_is_input_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from self_healing_pipeline.infrastructure.config.cloud_settings import DatabricksSettings

        monkeypatch.setenv("DATABRICKS_HOST", "https://example.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "fake-token")
        monkeypatch.setenv("DATABRICKS_JOB_ID", "1")
        monkeypatch.delenv("DATABRICKS_NOTEBOOK_PARAM_NAME", raising=False)

        settings = DatabricksSettings()  # type: ignore[call-arg]

        assert settings.notebook_param_name == "input_path"

    def test_notebook_param_name_env_override_still_works(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from self_healing_pipeline.infrastructure.config.cloud_settings import DatabricksSettings

        monkeypatch.setenv("DATABRICKS_HOST", "https://example.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "fake-token")
        monkeypatch.setenv("DATABRICKS_JOB_ID", "1")
        monkeypatch.setenv("DATABRICKS_NOTEBOOK_PARAM_NAME", "custom_param")

        settings = DatabricksSettings()  # type: ignore[call-arg]

        assert settings.notebook_param_name == "custom_param"


class TestDatabricksJobTrigger:
    def test_triggers_run_now_with_configured_param_name(
        self, fake_server: tuple[str, type[_RecordingHandler]]
    ) -> None:
        base_url, handler = fake_server
        handler.response_body = json.dumps({"run_id": 12345}).encode()
        trigger = DatabricksJobTrigger(
            host=base_url, token="fake-token", job_id=99, param_name="repaired_csv_path"
        )

        outcome = trigger.trigger(cloud_path="https://example.blob.core.windows.net/c/x.csv")

        assert outcome.success is True
        assert outcome.run_id == 12345
        assert len(handler.received) == 1
        request = handler.received[0]
        assert request["method"] == "POST"
        assert request["path"] == "/api/2.1/jobs/run-now"
        assert request["headers"]["authorization"] == "Bearer fake-token"
        body = json.loads(request["body"])
        assert body == {
            "job_id": 99,
            "notebook_params": {
                "repaired_csv_path": "https://example.blob.core.windows.net/c/x.csv"
            },
        }

    def test_strips_trailing_slash_from_host(
        self, fake_server: tuple[str, type[_RecordingHandler]]
    ) -> None:
        base_url, handler = fake_server
        handler.response_body = json.dumps({"run_id": 1}).encode()
        trigger = DatabricksJobTrigger(
            host=f"{base_url}/", token="t", job_id=1, param_name="p"
        )

        outcome = trigger.trigger(cloud_path="x")

        assert outcome.success is True
        assert handler.received[0]["path"] == "/api/2.1/jobs/run-now"

    def test_non_2xx_response_is_reported_as_failure_not_raised(
        self, fake_server: tuple[str, type[_RecordingHandler]]
    ) -> None:
        base_url, handler = fake_server
        handler.status_code = 401
        handler.response_body = b'{"error_code": "PERMISSION_DENIED"}'
        trigger = DatabricksJobTrigger(host=base_url, token="bad", job_id=1, param_name="p")

        outcome = trigger.trigger(cloud_path="x")

        assert outcome.success is False
        assert outcome.run_id is None
        assert outcome.error is not None and "401" in outcome.error

    def test_unreachable_host_is_reported_as_failure_not_raised(self) -> None:
        trigger = DatabricksJobTrigger(host="http://127.0.0.1:1", token="t", job_id=1, param_name="p")

        outcome = trigger.trigger(cloud_path="x")

        assert outcome.success is False
        assert outcome.error is not None


class _StubUploader:
    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome

    def upload(self, local_file_path: str, *, blob_name: str | None = None) -> Any:
        return self._outcome


class _StubTrigger:
    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome

    def trigger(self, *, cloud_path: str) -> Any:
        return self._outcome


class TestCloudIntegrationService:
    def test_uploader_none_skips_upload_and_trigger(self) -> None:
        service = CloudIntegrationService(uploader=None, trigger=None)

        outcome = service.process(local_file_path="/tmp/x.csv")

        assert outcome.uploaded is False
        assert outcome.triggered is False
        assert outcome.upload_error is None  # not configured, not a failure

    def test_upload_failure_skips_trigger_and_leaves_local_file_alone(self, tmp_path: Path) -> None:
        from self_healing_pipeline.infrastructure.cloud.azure_blob_uploader import UploadOutcome

        file_path = _write(tmp_path, "repaired.csv", "id\n1\n")
        service = CloudIntegrationService(
            uploader=_StubUploader(UploadOutcome(success=False, error="boom")),  # type: ignore[arg-type]
            trigger=_StubTrigger(None),  # type: ignore[arg-type]
        )

        outcome = service.process(local_file_path=file_path)

        assert outcome.uploaded is False
        assert outcome.upload_error == "boom"
        assert outcome.triggered is False
        assert Path(file_path).read_text(encoding="utf-8") == "id\n1\n"  # untouched

    def test_trigger_none_reports_uploaded_but_not_triggered(self, tmp_path: Path) -> None:
        from self_healing_pipeline.infrastructure.cloud.azure_blob_uploader import UploadOutcome

        file_path = _write(tmp_path, "repaired.csv", "id\n1\n")
        service = CloudIntegrationService(
            uploader=_StubUploader(  # type: ignore[arg-type]
                UploadOutcome(success=True, blob_url="https://example/c/repaired.csv")
            ),
            trigger=None,
        )

        outcome = service.process(local_file_path=file_path)

        assert outcome.uploaded is True
        assert outcome.cloud_path == "https://example/c/repaired.csv"
        assert outcome.triggered is False

    def test_trigger_failure_still_reports_upload_success(self, tmp_path: Path) -> None:
        from self_healing_pipeline.infrastructure.cloud.azure_blob_uploader import UploadOutcome
        from self_healing_pipeline.infrastructure.cloud.databricks_job_trigger import TriggerOutcome

        file_path = _write(tmp_path, "repaired.csv", "id\n1\n")
        service = CloudIntegrationService(
            uploader=_StubUploader(  # type: ignore[arg-type]
                UploadOutcome(success=True, blob_url="https://example/c/repaired.csv")
            ),
            trigger=_StubTrigger(TriggerOutcome(success=False, error="job not found")),  # type: ignore[arg-type]
        )

        outcome = service.process(local_file_path=file_path)

        assert outcome.uploaded is True
        assert outcome.cloud_path == "https://example/c/repaired.csv"
        assert outcome.triggered is False
        assert outcome.trigger_error == "job not found"

    def test_full_success(self, tmp_path: Path) -> None:
        from self_healing_pipeline.infrastructure.cloud.azure_blob_uploader import UploadOutcome
        from self_healing_pipeline.infrastructure.cloud.databricks_job_trigger import TriggerOutcome

        file_path = _write(tmp_path, "repaired.csv", "id\n1\n")
        service = CloudIntegrationService(
            uploader=_StubUploader(  # type: ignore[arg-type]
                UploadOutcome(success=True, blob_url="https://example/c/repaired.csv")
            ),
            trigger=_StubTrigger(TriggerOutcome(success=True, run_id=42)),  # type: ignore[arg-type]
        )

        outcome = service.process(local_file_path=file_path)

        assert outcome.uploaded is True
        assert outcome.triggered is True
        assert outcome.databricks_run_id == 42
