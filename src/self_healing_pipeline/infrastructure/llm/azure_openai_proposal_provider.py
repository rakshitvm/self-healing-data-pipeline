"""Azure OpenAI implementation of `CsvRepairProposalPort`.

The Azure/OpenAI SDK is confined to this module — no domain model,
`ErrorRouter`, `CsvRepairAgent`, `CsvFailureDetector`, `CsvRepairExecutor`,
or `application/orchestration` module imports it. This provider only
*proposes*: it never calls pandas, never writes to the file system, and
never applies a repair. Its return value is a raw, unvalidated payload —
`CsvRepairParams` validation happens downstream, in the existing
workflow's `validate` node (unchanged). If the model's response cannot be
parsed as a JSON object (or the request itself fails), `propose` returns
an empty dict rather than raising, so the workflow's existing retry/fail
routing handles it exactly like any other invalid proposal.
"""

import json
from typing import Any

from openai import AzureOpenAI

from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.config.settings import AzureOpenAISettings

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


class AzureOpenAIProposalProvider:
    """Concrete `CsvRepairProposalPort` backed by Azure OpenAI."""

    def __init__(self, client: AzureOpenAI, deployment: str) -> None:
        self._client = client
        self._deployment = deployment

    @classmethod
    def from_settings(cls, settings: AzureOpenAISettings) -> "AzureOpenAIProposalProvider":
        """Build a provider with a real `AzureOpenAI` client from `settings`."""
        client = AzureOpenAI(
            azure_endpoint=settings.endpoint,
            api_key=settings.api_key,
            api_version=settings.api_version,
            azure_deployment=settings.deployment_name,
        )
        return cls(client=client, deployment=settings.deployment_name)

    def propose(
        self, *, failure_class: FailureClass, sample: str, file_path: str
    ) -> dict[str, Any]:
        """Propose a raw (unvalidated) candidate for `CsvRepairParams`.

        Returns an empty dict — never raises — if the request fails or the
        response cannot be parsed as a JSON object; the workflow's
        `validate` node then correctly treats that as an invalid proposal.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
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
        except Exception:  
            return {}

        return parsed if isinstance(parsed, dict) else {}
