"""Pandas-backed current-schema inspection.

The only place allowed to import pandas for schema inspection purposes,
mirroring `PandasCsvRepairExecutor`'s isolation of pandas to a single
infrastructure adapter.
"""

import pandas as pd

from self_healing_pipeline.domain.entities.schema_definition import ColumnDefinition

_PANDAS_DTYPE_TO_LOGICAL_TYPE: dict[str, str] = {
    "int64": "int64",
    "int32": "int64",
    "float64": "float64",
    "float32": "float64",
    "bool": "bool",
    "object": "string",
    "string": "string",
    "datetime64[ns]": "datetime",
}


def _logical_type(dtype: str) -> str:
    return _PANDAS_DTYPE_TO_LOGICAL_TYPE.get(dtype, "string")


class PandasSchemaInspector:
    """Concrete `CurrentSchemaInspector` backed by pandas dtype inference."""

    def inspect(self, file_path: str) -> tuple[ColumnDefinition, ...]:
        frame = pd.read_csv(file_path)
        columns = []
        for name in frame.columns:
            series = frame[name]
            columns.append(
                ColumnDefinition(
                    name=str(name),
                    type=_logical_type(str(series.dtype)),
                    nullable=bool(series.isna().any()),
                )
            )
        return tuple(columns)
