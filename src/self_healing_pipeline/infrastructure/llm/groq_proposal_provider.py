"""Groq implementation of `CsvRepairProposalPort` — TEMPORARY development provider.

Groq is used only because the team has not yet provided Azure OpenAI
credentials; `AzureOpenAIProposalProvider` remains the production
implementation the project specification actually requires, and this
module leaves it completely untouched. Groq exposes an OpenAI-compatible
API (https://api.groq.com/openai/v1, JSON Object Mode confirmed
supported on all Groq models per Groq's own docs), so this provider
reuses the already-installed `openai` SDK's plain `OpenAI` client — not
`AzureOpenAI` — pointed at Groq's base URL. No new dependency is needed.

The prompt below is intentionally near-identical to
`AzureOpenAIProposalProvider`'s (same task, same required JSON shape,
same request options) — duplicated rather than imported, since
`AzureOpenAIProposalProvider` must not be modified to expose it.
"""

import json
from typing import Any

from openai import OpenAI

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.config.settings import GroqSettings

_SYSTEM_PROMPT = (
    "You are a deterministic CSV repair parameter proposer for a data "
    "pipeline. Given a failure classification and a small sample of a "
    "malformed CSV file, respond with ONLY a JSON object describing the "
    "parameters needed to read the file correctly. The JSON object must "
    'have exactly these keys: "delimiter" (a single character), '
    '"encoding" (a valid Python text encoding name), "header_row" '
    '(an integer row index, or null), and "engine" (one of "python", '
    '"c", "pyarrow"). Return JSON only, with no surrounding text.'
)


def _build_user_prompt(*, failure_class: FailureClass, sample: str, file_path: str) -> str:
    return (
        f"Failure class: {failure_class.value}\n"
        f"File path: {file_path}\n"
        "CSV sample (first lines, as read with a naive default parse):\n"
        f"{sample}\n\n"
        "Propose the repair parameters as a JSON object."
    )


class GroqProposalProvider:
    """Concrete `CsvRepairProposalPort` backed by Groq's OpenAI-compatible API.

    TEMPORARY development-only provider — see module docstring. Behaves
    identically to `AzureOpenAIProposalProvider` on failure: any request
    or parsing error returns an empty dict rather than raising, so the
    workflow's existing retry/fail routing handles it exactly like any
    other invalid proposal.
    """

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_settings(cls, settings: GroqSettings) -> "GroqProposalProvider":
        """Build a provider with a real `OpenAI` client pointed at Groq."""
        client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        return cls(client=client, model=settings.model)

    def propose(
        self, *, failure_class: FailureClass, sample: str, file_path: str
    ) -> dict[str, Any]:
        """Propose a raw (unvalidated) candidate for `CsvRepairParams`.

        Returns an empty dict — never raises — if the request fails or the
        response cannot be parsed as a JSON object.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _build_user_prompt(
                            failure_class=failure_class, sample=sample, file_path=file_path
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content
            if content is None:
                return {}
            parsed = json.loads(content)
        except Exception:  # noqa: BLE001 - any failure here becomes an invalid proposal, not a crash
            return {}

        return parsed if isinstance(parsed, dict) else {}
