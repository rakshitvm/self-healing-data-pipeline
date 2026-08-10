"""Focused unit tests for `build_proposal_provider` (LLM_PROVIDER selection).

No real network calls: client construction is patched for both providers,
and only syntactically-valid placeholder credentials are ever used.
"""

from unittest.mock import patch

import pytest

from self_healing_pipeline.infrastructure.llm.azure_openai_proposal_provider import (
    AzureOpenAIProposalProvider,
)
from self_healing_pipeline.infrastructure.llm.groq_proposal_provider import GroqProposalProvider
from self_healing_pipeline.infrastructure.llm.proposal_provider_factory import (
    build_proposal_provider,
)

_GROQ_ENV = {"GROQ_API_KEY": "unused", "GROQ_MODEL": "llama-3.3-70b-versatile"}
_AZURE_ENV = {
    "AZURE_OPENAI_API_KEY": "unused",
    "AZURE_OPENAI_ENDPOINT": "https://unused.openai.azure.com",
    "AZURE_OPENAI_API_VERSION": "2024-02-01",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "unused",
}


def test_provider_selection_chooses_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _GROQ_ENV.items():
        monkeypatch.setenv(key, value)

    with patch("self_healing_pipeline.infrastructure.llm.groq_proposal_provider.OpenAI"):
        provider = build_proposal_provider("groq")

    assert isinstance(provider, GroqProposalProvider)
    assert not isinstance(provider, AzureOpenAIProposalProvider)


def test_provider_selection_chooses_azure_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _AZURE_ENV.items():
        monkeypatch.setenv(key, value)

    with patch("self_healing_pipeline.infrastructure.llm.azure_openai_proposal_provider.AzureOpenAI"):
        provider = build_proposal_provider("azure_openai")

    assert isinstance(provider, AzureOpenAIProposalProvider)
    assert not isinstance(provider, GroqProposalProvider)


def test_provider_selection_reads_llm_provider_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _GROQ_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    with patch("self_healing_pipeline.infrastructure.llm.groq_proposal_provider.OpenAI"):
        provider = build_proposal_provider()

    assert isinstance(provider, GroqProposalProvider)


def test_no_azure_client_is_constructed_when_groq_is_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in _GROQ_ENV.items():
        monkeypatch.setenv(key, value)

    with (
        patch("self_healing_pipeline.infrastructure.llm.groq_proposal_provider.OpenAI"),
        patch(
            "self_healing_pipeline.infrastructure.llm.azure_openai_proposal_provider.AzureOpenAI"
        ) as mock_azure_cls,
    ):
        build_proposal_provider("groq")

    mock_azure_cls.assert_not_called()


def test_unknown_provider_raises_a_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_proposal_provider("not-a-real-provider")
