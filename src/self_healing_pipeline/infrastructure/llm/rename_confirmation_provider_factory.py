"""Provider selection for `RenameConfirmationPort`.

Mirrors `proposal_provider_factory.py` exactly, including reusing the
same `LLMProviderSettings` (`LLM_PROVIDER` in `.env`) — schema repair and
CSV repair select their LLM provider the same way, from the same setting.
"""

from self_healing_pipeline.domain.interfaces.services.rename_confirmation_port import (
    RenameConfirmationPort,
)
from self_healing_pipeline.infrastructure.config.settings import (
    load_azure_openai_settings,
    load_groq_settings,
    load_llm_provider_settings,
)
from self_healing_pipeline.infrastructure.llm.azure_rename_confirmation_provider import (
    AzureRenameConfirmationProvider,
)
from self_healing_pipeline.infrastructure.llm.groq_rename_confirmation_provider import (
    GroqRenameConfirmationProvider,
)

DEFAULT_LLM_PROVIDER = "azure_openai"
_GROQ = "groq"
_AZURE_OPENAI = "azure_openai"


def build_rename_confirmation_provider(
    provider_name: str | None = None,
) -> RenameConfirmationPort:
    """Build the `RenameConfirmationPort` implementation selected by `LLM_PROVIDER`."""
    raw_selection = (
        provider_name if provider_name is not None else load_llm_provider_settings().provider
    )
    selected = raw_selection.strip().lower()

    if selected == _GROQ:
        return GroqRenameConfirmationProvider.from_settings(load_groq_settings())
    if selected == _AZURE_OPENAI:
        return AzureRenameConfirmationProvider.from_settings(load_azure_openai_settings())

    raise ValueError(
        f"Unknown LLM_PROVIDER={selected!r}; expected {_GROQ!r} or {_AZURE_OPENAI!r}."
    )
