"""Databricks Jobs API ("run now") trigger.

Deliberately stdlib-only (`urllib.request`) — no `databricks-sdk`
dependency is required for a single authenticated POST. See Databricks'
own REST API reference for "Trigger a new job run"
(`POST /api/2.1/jobs/run-now`):
https://docs.databricks.com/api/workspace/jobs/runnow

This module never raises: every failure mode (missing config, network
error, non-2xx response, unparsable body) is caught and reported on the
returned `TriggerOutcome`, matching this project's existing best-effort
observability contracts (`TrackingOutcome`, `CsvExecutionOutcome`). It
never touches the local filesystem — it only ever sends `cloud_path`
(a plain string) as a job parameter.

The job-parameter *key* your Databricks job actually expects depends on
how that job/notebook was authored — this project has no way to know
that in advance, so it is configurable (`param_name`) rather than
guessed; see `DatabricksSettings` in `infrastructure/config/cloud_settings.py`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from pydantic import BaseModel, ConfigDict


class TriggerOutcome(BaseModel):
    """Result of one best-effort "trigger a Databricks job run" attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    success: bool
    run_id: int | None = None
    error: str | None = None


class DatabricksJobTrigger:
    """Triggers a `run-now` invocation of one pre-existing Databricks job."""

    def __init__(self, *, host: str, token: str, job_id: int, param_name: str) -> None:
        """`host` is the workspace URL (e.g.
        `https://adb-xxxx.azuredatabricks.net`, no trailing slash
        required — it is stripped if present). `token` is a Databricks
        personal access token, only ever used as a bearer credential on
        this one request, never logged. `job_id` must already exist in
        the target workspace — this class never creates or discovers
        jobs. `param_name` is the notebook-parameter key the target
        job's notebook task is written to read (see module docstring).
        """
        self._host = host.rstrip("/")
        self._token = token
        self._job_id = job_id
        self._param_name = param_name

    def trigger(self, *, cloud_path: str) -> TriggerOutcome:
        """Call `run-now` for the configured job, passing `cloud_path` as
        the single notebook parameter named `param_name`."""
        url = f"{self._host}/api/2.1/jobs/run-now"
        payload = json.dumps(
            {"job_id": self._job_id, "notebook_params": {self._param_name: cloud_path}}
        ).encode()

        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read())
                if 200 <= response.status < 300:
                    run_id = body.get("run_id")
                    return TriggerOutcome(success=True, run_id=run_id)
                return TriggerOutcome(
                    success=False, error=f"Unexpected status {response.status} from Databricks"
                )
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")[:500]
            return TriggerOutcome(success=False, error=f"HTTP {exc.code} from Databricks: {body_text}")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return TriggerOutcome(success=False, error=f"{type(exc).__name__}: {exc}")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return TriggerOutcome(success=False, error=f"Could not parse Databricks response: {exc}")
