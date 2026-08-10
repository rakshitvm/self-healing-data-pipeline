"""Tier 1 CSV repair agent.

Concrete `RepairAgent` for the five Tier 1 CSV failure dimensions. It
holds no routing logic (`ErrorRouter` owns dispatch) and no CSV
manipulation code of its own (a `CsvRepairExecutor` owns that) — its only
job is: given a `PipelineError`, select the Tier 1 default
`CsvRepairParams` prescription for its `failure_class`, execute that
prescription through the injected `CsvRepairExecutor`, and translate the
outcome into a `RepairResult`.

The per-dimension default prescriptions below are a Tier 1 placeholder:
later tiers replace them with an LLM-derived prescription (Azure OpenAI)
without changing this agent's shape, since `CsvRepairParams` is already
the structure any LLM-derived prescription must fit.
"""

from typing import ClassVar

from self_healing_pipeline.domain.exceptions.domain_exceptions import PipelineError
from self_healing_pipeline.domain.interfaces.services.csv_repair_executor import (
    CsvRepairExecutor,
)
from self_healing_pipeline.domain.value_objects.csv_repair_params import (
    CsvEngine,
    CsvRepairParams,
)
from self_healing_pipeline.domain.value_objects.failure_class import FailureClass
from self_healing_pipeline.domain.value_objects.repair_result import RepairResult


class CsvRepairAgent:
    """Tier 1 concrete `RepairAgent` for CSV ingestion failures."""

    _DEFAULT_PRESCRIPTIONS: ClassVar[dict[FailureClass, CsvRepairParams]] = {
        FailureClass.WRONG_DELIMITER: CsvRepairParams(
            delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON
        ),
        FailureClass.WRONG_ENCODING: CsvRepairParams(
            delimiter=",", encoding="latin-1", header_row=0, engine=CsvEngine.PYTHON
        ),
        FailureClass.HEADER_DETECTION: CsvRepairParams(
            delimiter=",", encoding="utf-8", header_row=1, engine=CsvEngine.PYTHON
        ),
        FailureClass.ENGINE_SELECTION: CsvRepairParams(
            delimiter=",", encoding="utf-8", header_row=0, engine=CsvEngine.PYARROW
        ),
        FailureClass.SINGLE_COLUMN_MALFORMATION: CsvRepairParams(
            delimiter=";", encoding="utf-8", header_row=0, engine=CsvEngine.PYTHON
        ),
    }

    def __init__(self, executor: CsvRepairExecutor) -> None:
        self._executor = executor

    def handle(self, error: PipelineError) -> RepairResult:
        """Repair the CSV failure described by `error`.

        Looks up the Tier 1 default prescription for `error.failure_class`,
        executes it via the injected `CsvRepairExecutor`, and returns the
        outcome as a `RepairResult`. If `error.failure_class` is not one of
        the Tier 1 CSV dimensions this agent supports, `error` is re-raised
        as-is — reusing the existing `PipelineError` hierarchy rather than
        introducing a new one. If `error.file_path` is missing, execution is
        skipped and an unsuccessful `RepairResult` is returned.
        """
        prescription = self._DEFAULT_PRESCRIPTIONS.get(error.failure_class)
        if prescription is None:
            raise error

        if error.file_path is None:
            return RepairResult(
                success=False,
                applied=False,
                prescription=prescription,
                validation_errors=["missing_file_path"],
                message=f"Cannot repair {error.failure_class.value}: no file_path was provided.",
            )

        outcome = self._executor.execute(error.file_path, prescription)

        return RepairResult(
            success=outcome.success,
            applied=outcome.success,
            confidence=outcome.confidence,
            prescription=prescription,
            validation_errors=outcome.validation_errors,
            message=outcome.message,
        )
