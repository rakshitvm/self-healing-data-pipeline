"""Provider selection for `CsvRepairProposalPort`.

The smallest possible selection mechanism: reads `LLM_PROVIDER` directly
from the environment (not from `Settings`, since Groq's and Azure's
settings must each remain independently optional — see `GroqSettings`)
and returns the matching concrete provider. LangGraph and every other
caller depend only on `CsvRepairProposalPort`; nothing is coupled to
either concrete provider — this factory is the one place that chooses.
"""

import os

from self_healing_pipeline.domain.interfaces.services.csv_repair_proposal_port import (
    CsvRepairProposalPort,
)
from self_healing_pipeline.infrastructure.config.settings import (
    load_azure_openai_settings,
    load_groq_settings,
)
from self_healing_pipeline.infrastructure.llm.azure_openai_proposal_provider import (
    AzureOpenAIProposalProvider,
)
from self_healing_pipeline.infrastructure.llm.groq_proposal_provider import GroqProposalProvider

DEFAULT_LLM_PROVIDER = "azure_openai"
_GROQ = "groq"
_AZURE_OPENAI = "azure_openai"


def build_proposal_provider(provider_name: str | None = None) -> CsvRepairProposalPort:
    """Build the `CsvRepairProposalPort` implementation selected by `LLM_PROVIDER`.

    `provider_name` overrides the environment for testing; production
    callers should omit it and rely on the `LLM_PROVIDER` env var
    (defaulting to `"azure_openai"`, the specification's intended
    production provider).
    """
    raw_selection = provider_name if provider_name is not None else os.environ.get(
        "LLM_PROVIDER", DEFAULT_LLM_PROVIDER
    )
    selected = raw_selection.strip().lower()

    if selected == _GROQ:
        return GroqProposalProvider.from_settings(load_groq_settings())
    if selected == _AZURE_OPENAI:
        return AzureOpenAIProposalProvider.from_settings(load_azure_openai_settings())

    raise ValueError(
        f"Unknown LLM_PROVIDER={selected!r}; expected {_GROQ!r} or {_AZURE_OPENAI!r}."
    )
