"""Focused unit tests for `AzureOpenAIProposalProvider`.

Every test uses a `MagicMock` in place of the real `openai.AzureOpenAI`
client. No network call is made and no real Azure credentials are
required or read.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from self_healing_pipeline.application.orchestration.csv_repair_workflow import (
    build_csv_repair_workflow,
    build_initial_state,
)
from self_healing_pipeline.domain.interfaces.services.csv_repair_proposal_port import (
    CsvRepairProposalPort,
)
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.pipeline_status import RepairEpisodeStatus
from self_healing_pipeline.infrastructure.csv.local_csv_failure_detector import (
    LocalCsvFailureDetector,
)
from self_healing_pipeline.infrastructure.csv.pandas_csv_repair_executor import (
    PandasCsvRepairExecutor,
)
from self_healing_pipeline.infrastructure.llm.azure_openai_proposal_provider import (
    AzureOpenAIProposalProvider,
)

VALID_PROPOSAL = {"delimiter": ";", "encoding": "utf-8", "header_row": 0, "engine": "python"}
WRONG_DELIMITER_CSV = "id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n"


def _mock_client_returning(content: str | None) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    client.chat.completions.create.return_value = response
    return client


def _write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_azure_provider_satisfies_csv_repair_proposal_port() -> None:
    provider = AzureOpenAIProposalProvider(client=MagicMock(), deployment="gpt-4o-mini")

    assert isinstance(provider, CsvRepairProposalPort)


def test_correct_azure_request_construction() -> None:
    client = _mock_client_returning(json.dumps(VALID_PROPOSAL))
    provider = AzureOpenAIProposalProvider(client=client, deployment="my-deployment")

    provider.propose(
        failure_class=FailureClass.WRONG_DELIMITER,
        sample="id;name\n1;alpha\n",
        file_path="/data/sales.csv",
    )

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "my-deployment"
    assert call_kwargs["response_format"] == {"type": "json_object"}
    roles = [m["role"] for m in call_kwargs["messages"]]
    assert roles == ["system", "user"]
    user_content = call_kwargs["messages"][1]["content"]
    assert "wrong_delimiter" in user_content
    assert "/data/sales.csv" in user_content
    assert "id;name" in user_content


def test_successful_structured_proposal_response() -> None:
    client = _mock_client_returning(json.dumps(VALID_PROPOSAL))
    provider = AzureOpenAIProposalProvider(client=client, deployment="gpt-4o-mini")

    result = provider.propose(
        failure_class=FailureClass.WRONG_DELIMITER, sample="id;name\n1;a\n", file_path="/tmp/x.csv"
    )

    assert result == VALID_PROPOSAL


def test_malformed_response_is_returned_as_empty_proposal_not_an_exception() -> None:
    client = _mock_client_returning("this is not valid json {{{")
    provider = AzureOpenAIProposalProvider(client=client, deployment="gpt-4o-mini")

    result = provider.propose(
        failure_class=FailureClass.WRONG_ENCODING, sample="x", file_path="/tmp/y.csv"
    )

    assert result == {}


def test_missing_content_is_returned_as_empty_proposal() -> None:
    client = _mock_client_returning(None)
    provider = AzureOpenAIProposalProvider(client=client, deployment="gpt-4o-mini")

    result = provider.propose(
        failure_class=FailureClass.WRONG_ENCODING, sample="x", file_path="/tmp/y.csv"
    )

    assert result == {}


def test_client_exception_is_returned_as_empty_proposal_not_raised() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("simulated API failure")
    provider = AzureOpenAIProposalProvider(client=client, deployment="gpt-4o-mini")

    result = provider.propose(
        failure_class=FailureClass.WRONG_ENCODING, sample="x", file_path="/tmp/y.csv"
    )

    assert result == {}


def test_provider_never_applies_a_repair_or_touches_the_file(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    before = Path(file_path).read_bytes()
    client = _mock_client_returning(json.dumps(VALID_PROPOSAL))
    provider = AzureOpenAIProposalProvider(client=client, deployment="gpt-4o-mini")

    result = provider.propose(
        failure_class=FailureClass.WRONG_DELIMITER, sample=WRONG_DELIMITER_CSV, file_path=file_path
    )

    assert isinstance(result, dict)
    assert not hasattr(provider, "apply")
    assert not hasattr(provider, "execute")
    assert Path(file_path).read_bytes() == before


def test_csv_repair_params_validation_still_happens_before_apply_valid(tmp_path: Path) -> None:
    """A valid Azure proposal flows all the way through validate -> apply -> verify."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    client = _mock_client_returning(json.dumps(VALID_PROPOSAL))
    provider = AzureOpenAIProposalProvider(client=client, deployment="gpt-4o-mini")
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(), executor=PandasCsvRepairExecutor(), llm_port=provider
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["validated_params"] is not None
    assert result["repair_result"] is not None and result["repair_result"].success is True
    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    assert client.chat.completions.create.call_count == 1


def test_csv_repair_params_validation_still_happens_before_apply_invalid(tmp_path: Path) -> None:
    """An invalid Azure proposal (bad delimiter) never reaches apply."""
    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    invalid_payload = {"delimiter": "too-long", "encoding": "utf-8"}
    client = _mock_client_returning(json.dumps(invalid_payload))
    provider = AzureOpenAIProposalProvider(client=client, deployment="gpt-4o-mini")
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(), executor=PandasCsvRepairExecutor(), llm_port=provider
    )

    result = graph.invoke(build_initial_state(file_path, max_retries=0))

    assert result["validated_params"] is None
    assert result["validation_errors"]
    assert result["repair_result"] is None
    assert result["status"] == RepairEpisodeStatus.FAILED


def test_healthy_csv_path_makes_zero_azure_calls(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "healthy.csv", "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n"
    )
    client = _mock_client_returning(json.dumps(VALID_PROPOSAL))
    provider = AzureOpenAIProposalProvider(client=client, deployment="gpt-4o-mini")
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(), executor=PandasCsvRepairExecutor(), llm_port=provider
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
    client.chat.completions.create.assert_not_called()


def test_non_azure_fake_provider_can_still_be_injected_into_the_workflow(tmp_path: Path) -> None:
    """A plain, non-Azure object satisfying the Port still works (Protocol
    substitutability is preserved after adding the Azure implementation)."""

    class _TrivialFakeProvider:
        def propose(
            self, *, failure_class: FailureClass, sample: str, file_path: str
        ) -> dict[str, object]:
            return VALID_PROPOSAL

    file_path = _write(tmp_path, "wrong_delimiter.csv", WRONG_DELIMITER_CSV)
    graph = build_csv_repair_workflow(
        detector=LocalCsvFailureDetector(),
        executor=PandasCsvRepairExecutor(),
        llm_port=_TrivialFakeProvider(),
    )

    result = graph.invoke(build_initial_state(file_path))

    assert result["status"] == RepairEpisodeStatus.SUCCEEDED
