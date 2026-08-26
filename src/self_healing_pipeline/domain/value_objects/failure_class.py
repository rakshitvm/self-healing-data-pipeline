"""Failure classification value object.

`FailureClass` is the Tier 1 vocabulary of CSV ingestion failure
dimensions a repair episode can be tagged with. It is a pure
classification value and carries no repair behavior. New failure
dimensions may be appended over time without coupling this taxonomy to
any concrete agent or handler implementation.
"""

from enum import Enum


class FailureClass(str, Enum):
    """Category of failure that triggered a repair episode."""

    WRONG_DELIMITER = "wrong_delimiter"
    WRONG_ENCODING = "wrong_encoding"
    HEADER_DETECTION = "header_detection"
    ENGINE_SELECTION = "engine_selection"
    SINGLE_COLUMN_MALFORMATION = "single_column_malformation"
    MIXED_DELIMITER = "mixed_delimiter"
    SCHEMA_DRIFT = "schema_drift"
    UNKNOWN = "unknown"
