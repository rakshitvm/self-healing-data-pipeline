"""Structured application logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from self_healing_pipeline.infrastructure.config.settings import LoggingSettings


def configure_logging(settings: LoggingSettings | None = None) -> None:
    """Configure structured logging to stdout and a file."""
    if settings is None:
        from self_healing_pipeline.infrastructure.config.settings import (
            _load_logging_settings,
        )

        settings = _load_logging_settings()

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "self_healing_pipeline.log"

    level = getattr(logging, settings.log_level)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=processors,
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


def _active_mlflow_trace_id() -> str | None:
    """Best-effort lookup of the currently active MLflow trace ID.

    Read-only and side-effect-free: creates no spans, never raises. If
    MLflow is unavailable, unconfigured, or no trace is currently active,
    returns `None` so callers fall back to unset `trace_id` exactly as
    before.
    """
    try:
        import mlflow

        trace_id: str | None = mlflow.get_active_trace_id()
    except Exception:  # noqa: BLE001 - observability must never break the app
        return None
    return trace_id


def get_logger(
    *,
    agent: str | None = None,
    node: str | None = None,
    table: str | None = None,
    trace_id: str | None = None,
) -> structlog.stdlib.BoundLogger:
    """Return a logger with the required Tier 1 correlation fields bound.

    If `trace_id` is not explicitly supplied, best-effort fills it from
    the currently active MLflow trace (if any) — see
    `_active_mlflow_trace_id`.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger()

    if trace_id is None:
        trace_id = _active_mlflow_trace_id()

    return logger.bind(
        agent=agent,
        node=node,
        table=table,
        trace_id=trace_id,
    )
