"""Groq implementation of `RenameConfirmationPort` — TEMPORARY development provider.

Mirrors `GroqProposalProvider`'s rationale exactly: Groq exposes an
OpenAI-compatible API, so this reuses the plain `openai.OpenAI` client
pointed at Groq's base URL. `AzureRenameConfirmationProvider` remains the
production implementation; this module leaves it untouched.
"""

import json
from typing import Any

from openai import OpenAI

from self_healing_pipeline.domain.interfaces.services.rename_confirmation_port import (
    RenameConfirmation,
)
from self_healing_pipeline.domain.value_objects.column_diff import RenameHint
from self_healing_pipeline.infrastructure.config.settings import GroqSettings

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


class GroqRenameConfirmationProvider:
    """Concrete `RenameConfirmationPort` backed by Groq's OpenAI-compatible API."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_settings(cls, settings: GroqSettings) -> "GroqRenameConfirmationProvider":
        client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        return cls(client=client, model=settings.model)

    def confirm(self, *, table: str, hint: RenameHint) -> RenameConfirmation:
        """Same best-effort contract as `AzureRenameConfirmationProvider.confirm`."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
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
