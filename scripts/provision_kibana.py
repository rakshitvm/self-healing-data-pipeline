"""Provision the Kibana saved objects for the Elasticsearch/Kibana log
search stack (index pattern, operational dashboard, and the repair
episode trace explorer saved search) from the checked-in export at
docker/kibana/saved_objects.ndjson.

This is the reproducibility fix for those objects having originally been
created ad hoc via the Kibana API: the exact same objects (same IDs) are
now version-controlled and can be recreated deterministically on any
fresh `docker compose up -d` environment. Uses `overwrite=true`, so
re-running this script is always safe/idempotent.

Only the stdlib is used — no new dependency is added for this.

Usage:
    docker compose up -d
    PYTHONPATH=src python scripts/provision_kibana.py
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

KIBANA_URL = "http://localhost:5601"
NDJSON_PATH = Path(__file__).resolve().parent.parent / "docker" / "kibana" / "saved_objects.ndjson"
MAX_WAIT_SECONDS = 120
POLL_INTERVAL_SECONDS = 3


def _wait_for_kibana() -> bool:
    """Poll Kibana's own status endpoint until it reports available, or
    give up after MAX_WAIT_SECONDS. Bounded and deterministic — never
    hangs indefinitely."""
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{KIBANA_URL}/api/status", timeout=5) as resp:
                status = json.loads(resp.read())
                # Kibana 7.x reports overall health as status.overall.state
                # ("green"/"yellow"/"red"), not the 8.x "level" field.
                state = status.get("status", {}).get("overall", {}).get("state")
                if state == "green":
                    return True
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass
        time.sleep(POLL_INTERVAL_SECONDS)
    return False


def _import_saved_objects() -> tuple[bool, str]:
    """POST the checked-in NDJSON export to Kibana's saved-objects import
    API as a real multipart/form-data upload (the API requires a file
    upload, not a raw JSON body)."""
    body = NDJSON_PATH.read_bytes()
    boundary = "----kibana-provision-boundary"
    parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="saved_objects.ndjson"\r\n'
        f"Content-Type: application/ndjson\r\n\r\n".encode(),
        body,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    payload = b"".join(parts)

    req = urllib.request.Request(
        f"{KIBANA_URL}/api/saved_objects/_import?overwrite=true",
        data=payload,
        method="POST",
        headers={
            "kbn-xsrf": "true",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"

    if not result.get("success"):
        return False, json.dumps(result)
    imported = ", ".join(f"{o['type']}/{o['id']}" for o in result.get("successResults", []))
    return True, f"imported {result.get('successCount')} object(s): {imported}"


def main() -> int:
    if not NDJSON_PATH.is_file():
        print(f"Saved objects export not found: {NDJSON_PATH}", file=sys.stderr)
        return 1

    print(f"Waiting for Kibana at {KIBANA_URL} (up to {MAX_WAIT_SECONDS}s)...")
    if not _wait_for_kibana():
        print("Kibana did not become available in time.", file=sys.stderr)
        return 1

    print("Kibana is available. Importing saved objects...")
    ok, message = _import_saved_objects()
    print(message)
    if not ok:
        return 1

    print(f"Dashboard ready: {KIBANA_URL}/app/dashboards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
