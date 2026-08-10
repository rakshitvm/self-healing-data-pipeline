"""Real MLflow integration/smoke tests (Ticket 013).

Runs against the actual local MLflow server brought up by Ticket 012
(docker-compose, http://localhost:5000). Automatically SKIPPED (not
failed) if that server is unreachable, so the suite stays robust in
environments where Docker hasn't been started — but in this repository's
development environment the server is expected to be up, and these
tests genuinely exercise real, persisted MLflow traces (not mocks).

No real Azure OpenAI or Groq credentials are required or used: the main
trace-structure tests use a fake `CsvRepairProposalPort` (the same
pattern used by every other test in this project). One dedicated test
additionally drives a real `openai.OpenAI` SDK call against a local fake
HTTP server standing in for an LLM endpoint — the same call pattern
`AzureOpenAIProposalProvider`/`GroqProposalProvider` use internally
(neither is modified) — to prove MLflow's LLM-span capture (prompt,
response, token usage) genuinely works, without needing real external
credentials.
"""

import http.server
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import mlflow
import pytest
from mlflow.entities import Trace

from self_healing_pipeline.application.orchestration.audited_csv_repair import (
    run_audited_csv_repair,
)
from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    build_csv_repair_workflow,
    build_initial_state,
)
from self_healing_pipeline.domain.entities.repair_episode import RepairEpisode
from self_healing_pipeline.domain.entities.repair_event import RepairEvent
from self_healing_pipeline.domain.interfaces.services.csv_repair_executor import (
    CsvExecutionOutcome,
)
from self_healing_pipeline.domain.value_objects.csv_repair_params import CsvRepairParams
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.config.settings import MLflowSettings
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
)
from self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor import (
    PandasCsvRepairExecutor,
)
from self_healing_pipeline.infrastructure.mlflow.mlflow_repair_trace_tracer import (
    MlflowRepairTraceTracer,
)
from self_healing_pipeline.infrastructure.mlflow.tracing_setup import enable_tracing

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "ticket013-integration-tests"
VALID_PROPOSAL = {"delimiter": ";", "encoding": "utf-8", "header_row": 0, "engine": "python"}
WRONG_DELIMITER_CSV = "id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n"


def _mlflow_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 5000), timeout=1.0):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _mlflow_reachable(),
    reason="real MLflow server not reachable at http://localhost:5000 (Ticket 012 docker-compose)",
)


@pytest.fixture(autouse=True)
def _configure_tracing() -> None:
    settings = MLflowSettings(  # type: ignore[call-arg]
        MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI,
        MLFLOW_EXPERIMENT_NAME=EXPERIMENT_NAME,
    )
    assert enable_tracing(settings) is True


class _FakeProposalPort:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        return dict(self._payload)


class _FailOnceThenSucceedProposalPort:
    """Returns an invalid proposal once, then a valid one — forces a retry."""

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {"delimiter": "too-long-invalid"}
        return dict(VALID_PROPOSAL)


class _RaisingExecutor:
    """Fake `CsvRepairExecutor` whose `execute` raises — forces an exception span."""

    def execute(self, file_path: str, params: CsvRepairParams) -> CsvExecutionOutcome:
        raise RuntimeError("simulated executor crash for exception-span testing")


class _InMemoryAuditStore:
    def __init__(self) -> None:
        self.episodes: list[RepairEpisode] = []
        self.events: list[RepairEvent] = []
        self.completed: list[RepairEpisode] = []

    def start_episode(self, episode: RepairEpisode) -> None:
        self.episodes.append(episode)

    def record_event(self, event: RepairEvent) -> None:
        self.events.append(event)

    def complete_episode(self, episode: RepairEpisode) -> None:
        self.completed.append(episode)


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _trace_id_from(event: RepairEvent) -> str:
    """Extract `payload["mlflow_trace_id"]`, asserting both are present."""
    assert event.payload is not None
    trace_id = event.payload["mlflow_trace_id"]
    assert isinstance(trace_id, str)
    return trace_id


def _fetch_trace(trace_id: str, *, min_spans: int = 1, retries: int = 20, delay: float = 0.5) -> Trace:
    """Poll until the trace is fully exported (async), not merely present.

    `mlflow.get_trace` can return a trace object before its spans have
    finished being flushed/attached — especially under load (e.g. the
    full test suite running many traces through the same async export
    queue at once), so a bare "is not None" check is not sufficient.
    Waits for a terminal `trace.info.state` (OK/ERROR, not IN_PROGRESS)
    and at least `min_spans` spans before returning.
    """
    last_exc: Exception | None = None
    last_trace: Trace | None = None
    for _ in range(retries):
        try:
            trace = cast(Trace | None, mlflow.get_trace(trace_id, flush=True))
            if trace is not None:
                last_trace = trace
                if trace.info.state in ("OK", "ERROR") and len(trace.data.spans) >= min_spans:
                    return trace
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        time.sleep(delay)
    if last_trace is not None:
        raise AssertionError(
            f"trace {trace_id} did not fully export after polling: "
            f"state={last_trace.info.state}, spans={len(last_trace.data.spans)}"
        )
    raise AssertionError(f"trace {trace_id} not available after polling: {last_exc}")


def test_real_repair_invocation_creates_a_trace_with_expected_id_and_tags(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )
    store = _InMemoryAuditStore()

    final_state = run_audited_csv_repair(
        graph,
        build_initial_state(file_path),
        audit_store=store,
        trace_tracer=MlflowRepairTraceTracer(),
    )

    assert final_state["status"].value == "succeeded"
    trace_id = _trace_id_from(store.events[0])
    assert trace_id is not None and trace_id.startswith("tr-")

    trace = _fetch_trace(trace_id)
    assert trace.info.state == "OK"
    assert trace.info.tags.get("episode_id") == str(final_state["episode_id"])
    assert trace.info.tags.get("failure_class") == "wrong_delimiter"


def test_real_trace_has_expected_spans_with_correct_parent_child_relationships(
    tmp_path: Path,
) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )
    store = _InMemoryAuditStore()
    run_audited_csv_repair(
        graph,
        build_initial_state(file_path),
        audit_store=store,
        trace_tracer=MlflowRepairTraceTracer(),
    )
    trace_id = _trace_id_from(store.events[0])
    trace = _fetch_trace(trace_id)

    by_name: dict[str, list[Any]] = {}
    for span in trace.data.spans:
        by_name.setdefault(span.name, []).append(span)

    root = by_name["repair_invocation"][0]
    assert root.parent_id is None  # genuinely the top-level span, not just named that way

    langgraph_span = by_name["LangGraph"][0]
    assert langgraph_span.parent_id == root.span_id

    for node_name in ("diagnose", "propose", "validate", "apply", "reverify"):
        assert node_name in by_name, f"missing span for node {node_name!r}"
        assert by_name[node_name][0].parent_id == langgraph_span.span_id

    sample_tool_span = by_name["sample_tool"][0]
    tool_span = by_name["sample_csv_file"][0]
    assert sample_tool_span.parent_id == langgraph_span.span_id
    assert tool_span.parent_id == sample_tool_span.span_id
    assert tool_span.span_type == "TOOL"


def test_real_tool_span_carries_real_input_and_output(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )
    store = _InMemoryAuditStore()
    run_audited_csv_repair(
        graph,
        build_initial_state(file_path),
        audit_store=store,
        trace_tracer=MlflowRepairTraceTracer(),
    )
    trace_id = _trace_id_from(store.events[0])
    trace = _fetch_trace(trace_id)

    tool_span = next(s for s in trace.data.spans if s.name == "sample_csv_file")
    assert tool_span.inputs is not None
    assert tool_span.inputs["file_path"] == file_path
    assert tool_span.inputs["max_lines"] == 5
    assert "id;name;value" in str(tool_span.outputs)


def test_real_trace_shows_retries_as_distinguishable_sibling_spans(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    flaky_llm = _FailOnceThenSucceedProposalPort()
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(), executor=PandasCsvRepairExecutor(), llm_port=flaky_llm
    )
    store = _InMemoryAuditStore()

    final_state = run_audited_csv_repair(
        graph,
        build_initial_state(file_path, max_retries=2),
        audit_store=store,
        trace_tracer=MlflowRepairTraceTracer(),
    )

    assert final_state["status"].value == "succeeded"
    assert flaky_llm.calls == 2  # one failed attempt, then one successful attempt
    trace_id = _trace_id_from(store.events[0])
    trace = _fetch_trace(trace_id)

    propose_spans = sorted(
        (s for s in trace.data.spans if s.name == "propose"), key=lambda s: s.start_time_ns
    )
    validate_spans = sorted(
        (s for s in trace.data.spans if s.name == "validate"), key=lambda s: s.start_time_ns
    )
    assert len(propose_spans) == 2, "expected two distinct propose spans (initial + 1 retry)"
    assert len(validate_spans) == 2
    assert propose_spans[0].span_id != propose_spans[1].span_id
    assert propose_spans[0].start_time_ns < validate_spans[0].start_time_ns < propose_spans[1].start_time_ns


def test_real_trace_records_exception_on_node_failure(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=_RaisingExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )
    store = _InMemoryAuditStore()

    with pytest.raises(RuntimeError, match="simulated executor crash"):
        run_audited_csv_repair(
            graph,
            build_initial_state(file_path),
            audit_store=store,
            trace_tracer=MlflowRepairTraceTracer(),
        )

    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id is not None
    trace = _fetch_trace(trace_id)

    assert trace.info.state == "ERROR"
    apply_span = next(s for s in trace.data.spans if s.name == "apply")
    assert apply_span.status.status_code.value == "ERROR"
    exception_events = [e for e in apply_span.events if e.name == "exception"]
    assert len(exception_events) == 1
    assert exception_events[0].attributes["exception.type"] == "RuntimeError"
    assert "simulated executor crash" in str(exception_events[0].attributes["exception.message"])


class _FakeOpenAIHandler(http.server.BaseHTTPRequestHandler):
    """Mimics an OpenAI-compatible chat completions endpoint. No real
    Azure/Groq credentials involved — this is a local, in-process server."""

    def do_POST(self) -> None:  # noqa: N802 - required name by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain the request body; content not inspected
        response = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(VALID_PROPOSAL)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59},
        }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:  # silence default request logging
        pass


@pytest.fixture
def fake_openai_server() -> Iterator[str]:
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5)


class _OpenAiSdkBackedProposalPort:
    """Test-only `CsvRepairProposalPort` using the real `openai` SDK
    against a local fake server — mirrors exactly what
    `AzureOpenAIProposalProvider`/`GroqProposalProvider` do internally
    (neither is modified), to exercise MLflow's real LLM-span capture
    without needing real external credentials."""

    def __init__(self, base_url: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key="fake-not-a-real-key", base_url=base_url)

    def propose(self, *, failure_class: FailureClass, sample: str, file_path: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model="fake-model",
            messages=[
                {"role": "system", "content": "You are a CSV repair parameter proposer."},
                {"role": "user", "content": f"Failure: {failure_class.value}\nSample: {sample}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        return dict(json.loads(content)) if content else {}


def test_real_llm_span_carries_prompt_response_and_token_usage(
    tmp_path: Path, fake_openai_server: str
) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_OpenAiSdkBackedProposalPort(fake_openai_server),
    )
    store = _InMemoryAuditStore()

    final_state = run_audited_csv_repair(
        graph,
        build_initial_state(file_path),
        audit_store=store,
        trace_tracer=MlflowRepairTraceTracer(),
    )
    assert final_state["status"].value == "succeeded"

    trace_id = _trace_id_from(store.events[0])
    trace = _fetch_trace(trace_id)

    llm_spans = [s for s in trace.data.spans if s.span_type == "CHAT_MODEL"]
    assert len(llm_spans) == 1, "expected exactly one LLM span for the one propose call"
    llm_span = llm_spans[0]

    propose_span = next(s for s in trace.data.spans if s.name == "propose")
    assert llm_span.parent_id == propose_span.span_id  # nested under propose, not top-level

    assert "wrong_delimiter" in str(llm_span.inputs)  # real prompt captured
    assert "delimiter" in str(llm_span.outputs)  # real response captured
    assert llm_span.attributes.get("mlflow.chat.tokenUsage") is not None  # real usage captured


def test_mlflow_unavailable_does_not_block_repair_and_is_bounded(tmp_path: Path) -> None:
    """MLflow's tracing is client-buffered: `mlflow.start_span` and
    `get_last_active_trace_id` succeed purely locally, generating a real
    trace_id, even when the tracking server is unreachable — only the
    background async *export* of that trace fails (silently, logged as a
    warning). So `mlflow_trace_id` is still populated here; what this
    test actually verifies is the property that matters: the repair
    completes correctly and the process is never blocked, at any stage
    (`enable_tracing` setup, the traced invocation itself, and the
    explicit bounded flush)."""
    unreachable_settings = MLflowSettings(  # type: ignore[call-arg]
        MLFLOW_TRACKING_URI="http://localhost:59999",
        MLFLOW_EXPERIMENT_NAME="ticket013-unreachable",
    )
    setup_started = time.perf_counter()
    setup_ok = enable_tracing(unreachable_settings)
    setup_elapsed = time.perf_counter() - setup_started
    assert setup_ok is False  # set_experiment against an unreachable host fails fast, not hangs
    assert setup_elapsed < 15.0, f"enable_tracing should fail fast, took {setup_elapsed}s"

    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_FakeProposalPort(VALID_PROPOSAL),
    )
    store = _InMemoryAuditStore()

    invoke_started = time.perf_counter()
    final_state = run_audited_csv_repair(
        graph,
        build_initial_state(file_path),
        audit_store=store,
        trace_tracer=MlflowRepairTraceTracer(),
    )
    invoke_elapsed = time.perf_counter() - invoke_started

    assert final_state["status"].value == "succeeded"
    assert invoke_elapsed < 15.0, f"repair should complete quickly even with MLflow unreachable, took {invoke_elapsed}s"

    from self_healing_pipeline.infrastructure.mlflow.tracing_setup import flush_traces

    flush_started = time.perf_counter()
    flush_traces(timeout_seconds=5.0)
    flush_elapsed = time.perf_counter() - flush_started
    assert flush_elapsed < 8.0, f"flush_traces must respect its bound, took {flush_elapsed}s"

    # restore tracing to the real server for any other tests sharing this process
    real_settings = MLflowSettings(  # type: ignore[call-arg]
        MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME=EXPERIMENT_NAME
    )
    enable_tracing(real_settings)
