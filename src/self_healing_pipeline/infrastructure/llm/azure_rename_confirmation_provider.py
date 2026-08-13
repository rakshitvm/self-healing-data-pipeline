"""Azure OpenAI implementation of `RenameConfirmationPort`.

The only LLM call in Tier 2 schema repair. Mirrors
`AzureOpenAIProposalProvider`'s raw-SDK-call shape exactly (same client
construction, same `response_format={"type": "json_object"}` pattern) so
MLflow's existing `mlflow.openai.autolog()` (already enabled once, at the
composition root, for Tier 1) captures this call's CHAT_MODEL span —
prompt, response, and real token usage — with zero additional tracing
code. Unlike `CsvRepairProposalPort`, this port's return value
deliberately surfaces `response.usage`, so real (not invented) token
counts can also be recorded on the migration history entry.
"""

import json
from typing import Any

from openai import AzureOpenAI

from self_healing_pipeline.domain.interfaces.services.rename_confirmation_port import (
    RenameConfirmation,
)
from self_healing_pipeline.domain.value_objects.column_diff import RenameHint
from self_healing_pipeline.infrastructure.config.settings import AzureOpenAISettings

_SYSTEM_PROMPT = (
    "You are a deterministic schema-drift rename confirmer for a data "
    "pipeline. You are given a table name and a candidate rename pairing "
    "a column that disappeared from the baseline schema with a column "
    "that newly appeared in the current schema, along with a deterministic "
    "name-similarity score. Decide whether this genuinely looks like the "
    "same column renamed (versus an unrelated column removed and a "
    "different, unrelated column added). Respond with ONLY a JSON object "
    'with exactly these keys: "confirmed" (true or false) and '
    '"confidence" (a number from 0.0 to 1.0, your own confidence in the '
    "confirmed/rejected judgment). Return JSON only, with no surrounding text."
)


def _build_user_prompt(*, table: str, hint: RenameHint) -> str:
    return (
        f"Table: {table}\n"
        f"Removed column (in baseline, not in current): {hint.removed_column!r}\n"
        f"Added column (in current, not in baseline): {hint.added_column!r}\n"
        f"Deterministic name similarity: {hint.similarity:.2f}\n\n"
        "Is this genuinely a rename? Respond as the required JSON object."
    )


class AzureRenameConfirmationProvider:
    """Concrete `RenameConfirmationPort` backed by Azure OpenAI."""

    def __init__(self, client: AzureOpenAI, deployment: str) -> None:
        self._client = client
        self._deployment = deployment

    @classmethod
    def from_settings(
        cls, settings: AzureOpenAISettings
    ) -> "AzureRenameConfirmationProvider":
        client = AzureOpenAI(
            azure_endpoint=settings.endpoint,
            api_key=settings.api_key,
            api_version=settings.api_version,
            azure_deployment=settings.deployment_name,
        )
        return cls(client=client, deployment=settings.deployment_name)

    def confirm(self, *, table: str, hint: RenameHint) -> RenameConfirmation:
        """Ask the LLM to confirm/reject `hint`.

        Best-effort at the *judgment* level only: any request/parsing
        failure is treated as a non-confirmation (confidence 0.0), never
        raised — a broken LLM call must never crash schema repair, it
        should simply fail to confirm a rename (falling back to
        drop+add_default for those columns, exactly like an explicit
        rejection would).
        """
        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(table=table, hint=hint)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content
            parsed: dict[str, Any] = json.loads(content) if content is not None else {}
            usage = response.usage
        except Exception:  # noqa: BLE001 - never let a broken LLM call crash schema repair
            return RenameConfirmation(confirmed=False, llm_confidence=0.0)

        confirmed = bool(parsed.get("confirmed", False))
        confidence = float(parsed.get("confidence", 0.0))
        confidence = min(1.0, max(0.0, confidence))

        return RenameConfirmation(
            confirmed=confirmed,
            llm_confidence=confidence,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=usage.completion_tokens if usage is not None else None,
        )
