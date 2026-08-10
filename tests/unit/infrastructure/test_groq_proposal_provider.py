"""Focused unit tests for `GroqProposalProvider`.

Every test uses a `MagicMock` in place of the real `openai.OpenAI`
client. No network call is made and no real Groq API key is required.
"""

import json
from unittest.mock import MagicMock

from self_healing_pipeline.domain.interfaces.services.csv_repair_proposal_port import (
    CsvRepairProposalPort,
)
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.infrastructure.llm.groq_proposal_provider import GroqProposalProvider

VALID_PROPOSAL = {"delimiter": ";", "encoding": "utf-8", "header_row": 0, "engine": "python"}


def _mock_client_returning(content: str | None) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    client.chat.completions.create.return_value = response
    return client


def test_groq_provider_satisfies_csv_repair_proposal_port() -> None:
    provider = GroqProposalProvider(client=MagicMock(), model="llama-3.3-70b-versatile")

    assert isinstance(provider, CsvRepairProposalPort)


def test_correct_openai_compatible_request_construction() -> None:
    client = _mock_client_returning(json.dumps(VALID_PROPOSAL))
    provider = GroqProposalProvider(client=client, model="llama-3.3-70b-versatile")

    provider.propose(
        failure_class=FailureClass.WRONG_DELIMITER,
        sample="id;name\n1;alpha\n",
        file_path="/data/sales.csv",
    )

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"
    assert call_kwargs["response_format"] == {"type": "json_object"}
    roles = [m["role"] for m in call_kwargs["messages"]]
    assert roles == ["system", "user"]
    user_content = call_kwargs["messages"][1]["content"]
    assert "wrong_delimiter" in user_content
    assert "/data/sales.csv" in user_content
    assert "id;name" in user_content


def test_successful_structured_proposal_response() -> None:
    client = _mock_client_returning(json.dumps(VALID_PROPOSAL))
    provider = GroqProposalProvider(client=client, model="llama-3.3-70b-versatile")

    result = provider.propose(
        failure_class=FailureClass.WRONG_DELIMITER, sample="id;name\n1;a\n", file_path="/tmp/x.csv"
    )

    assert result == VALID_PROPOSAL


def test_malformed_response_is_returned_as_empty_proposal_not_an_exception() -> None:
    client = _mock_client_returning("this is not valid json {{{")
    provider = GroqProposalProvider(client=client, model="llama-3.3-70b-versatile")

    result = provider.propose(
        failure_class=FailureClass.WRONG_ENCODING, sample="x", file_path="/tmp/y.csv"
    )

    assert result == {}


def test_missing_content_is_returned_as_empty_proposal() -> None:
    client = _mock_client_returning(None)
    provider = GroqProposalProvider(client=client, model="llama-3.3-70b-versatile")

    result = provider.propose(
        failure_class=FailureClass.WRONG_ENCODING, sample="x", file_path="/tmp/y.csv"
    )

    assert result == {}


def test_client_exception_is_returned_as_empty_proposal_not_raised() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("simulated API failure")
    provider = GroqProposalProvider(client=client, model="llama-3.3-70b-versatile")

    result = provider.propose(
        failure_class=FailureClass.WRONG_ENCODING, sample="x", file_path="/tmp/y.csv"
    )

    assert result == {}


def test_from_settings_builds_plain_openai_client_pointed_at_groq() -> None:
    from unittest.mock import patch

    from self_healing_pipeline.infrastructure.config.settings import GroqSettings

    settings = GroqSettings(  # type: ignore[call-arg]
        GROQ_API_KEY="unused-in-this-test",
        GROQ_MODEL="llama-3.3-70b-versatile",
    )

    with patch(
        "self_healing_pipeline.infrastructure.llm.groq_proposal_provider.OpenAI"
    ) as mock_openai_cls:
        provider = GroqProposalProvider.from_settings(settings)

    mock_openai_cls.assert_called_once_with(
        api_key="unused-in-this-test", base_url="https://api.groq.com/openai/v1"
    )
    assert provider._model == "llama-3.3-70b-versatile"  # noqa: SLF001 - white-box wiring check
