"""MLflow tracing setup: autolog registration and bounded flush-on-exit.

Confined to infrastructure. `enable_tracing` turns on MLflow's built-in
LangChain + OpenAI autologging (empirically verified against MLflow
3.15.1, Ticket 013) so every LangGraph node becomes a correctly nested
child span, the sample tool gets a TOOL span, and the raw `openai` SDK
call inside `propose` becomes a CHAT_MODEL span carrying real
prompt/response/token-usage — all with zero code changes to
`csv_repair_workflow.py`, `AzureOpenAIProposalProvider`, or
`GroqProposalProvider`.

Two SEPARATE, empirically-discovered hang risks are mitigated here, not
just one:

1. `mlflow.set_experiment(...)` (called by `enable_tracing` itself) can
   hang for minutes against an unreachable tracking server. This is NOT
   just a matter of `MLFLOW_HTTP_REQUEST_TIMEOUT` (default 120s) — MLflow
   retries failed HTTP calls up to `MLFLOW_HTTP_REQUEST_MAX_RETRIES`
   times (default 7) with exponential backoff
   (`MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR`, default 2), so the *cumulative*
   wait across retries dominates even a short per-request timeout.
   Verified: with the defaults, this hangs past 20s; with a short timeout
   AND retries disabled, an unreachable server fails in ~0.01s. Both env
   vars are set (via `setdefault`, so an operator's own configuration is
   never overridden) the first time `enable_tracing` runs.
2. The default atexit trace-flush can *also* hang if the tracking server
   is unreachable at process-exit time (governed by the same retry
   settings above, now bounded by mitigation 1 — but `flush_traces`
   layers on an explicit, independent bound regardless, using a daemon
   thread with a hard join timeout, so process shutdown is never at the
   mercy of MLflow's internals even if something else changes upstream).
"""

import os
import threading

from self_healing_pipeline.infrastructure.config.settings import MLflowSettings


def enable_tracing(settings: MLflowSettings) -> bool:
    """Enable MLflow tracing (autolog) against `settings`. Best-effort.

    Safe to call multiple times (MLflow's own autolog functions are
    idempotent). Returns whether tracing is enabled; a `False` return
    must not be treated as fatal by callers — the LangGraph workflow
    runs identically either way, just without traces. Bounded: does not
    hang even if `settings.tracking_uri` is unreachable (see module
    docstring) — sets `MLFLOW_HTTP_REQUEST_TIMEOUT`/
    `MLFLOW_HTTP_REQUEST_MAX_RETRIES` (via `setdefault`, so an operator's
    own configuration is never overridden) before making any MLflow call,
    trading retry resilience for a bounded worst case: acceptable for a
    best-effort observability path that is never the authoritative
    result.
    """
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    try:
        import mlflow
        import mlflow.langchain
        import mlflow.openai

        mlflow.set_tracking_uri(settings.tracking_uri)
        mlflow.set_experiment(settings.experiment_name)
        mlflow.langchain.autolog(log_traces=True, silent=True)
        mlflow.openai.autolog(log_traces=True, silent=True)
    except Exception:  # noqa: BLE001 - tracing setup must never block startup
        return False
    return True


def flush_traces(timeout_seconds: float = 5.0) -> None:
    """Best-effort, BOUNDED flush of pending async trace exports.

    Runs the real flush in a daemon thread and never waits past
    `timeout_seconds`, regardless of MLflow's reachability. An
    unfinished flush is abandoned (the daemon thread dies with the
    process) rather than blocking shutdown.
    """

    def _do_flush() -> None:
        try:
            import mlflow

            mlflow.flush_trace_async_logging(terminate=True)
        except Exception:  # noqa: BLE001
            pass

    thread = threading.Thread(target=_do_flush, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
