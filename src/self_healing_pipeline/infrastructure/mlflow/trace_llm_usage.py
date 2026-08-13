"""Run-level LLM token/cost aggregation from an already-recorded trace.

Confirmed empirically (not assumed) against real Groq calls in both Tier
1 and Tier 2: `mlflow.openai.autolog()` — already enabled globally by
`enable_tracing`, with zero code in this module — automatically attaches
real `mlflow.chat.tokenUsage` (prompt/completion/total tokens, from the
SDK response's own `usage` object) and, when MLflow has pricing data for
the model, `mlflow.llm.cost` to every CHAT_MODEL span it instruments,
for any raw `openai.OpenAI`/`AzureOpenAI` client call — which is exactly
what `GroqProposalProvider`, `AzureOpenAIProposalProvider`,
`GroqRenameConfirmationProvider`, and `AzureRenameConfirmationProvider`
all use internally. This module does not compute, estimate, or invent
any token count or price — it only reads and sums values MLflow itself
already captured, so a caller can log one run-level rollup instead of
only ever having per-span numbers. If a model has no MLflow pricing
entry, `total_cost_usd` is simply omitted — never fabricated.
"""

from typing import Any


def aggregate_chat_model_usage(trace_id: str) -> dict[str, Any] | None:
    """Best-effort sum of every CHAT_MODEL span's real token usage/cost.

    Returns `None` if the trace can't be read, has no CHAT_MODEL spans,
    or none of them carry usage data (e.g. tracing is disabled) — never
    raises, matching every other best-effort MLflow touchpoint in this
    codebase.
    """
    try:
        import mlflow

        trace = mlflow.get_trace(trace_id, flush=True)
    except Exception:  # noqa: BLE001 - best-effort: must never block the caller
        return None
    if trace is None:
        return None

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    total_cost: float | None = None
    found_usage = False

    for span in trace.data.spans:
        if span.span_type != "CHAT_MODEL":
            continue
        usage = span.attributes.get("mlflow.chat.tokenUsage")
        if isinstance(usage, dict):
            found_usage = True
            prompt_tokens += int(usage.get("input_tokens") or 0)
            completion_tokens += int(usage.get("output_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
        cost = span.attributes.get("mlflow.llm.cost")
        if isinstance(cost, dict) and cost.get("total_cost") is not None:
            total_cost = (total_cost or 0.0) + float(cost["total_cost"])

    if not found_usage:
        return None

    result: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if total_cost is not None:
        result["total_cost_usd"] = total_cost
    return result
