"""
Structured Logging Module.

Provides JSON-formatted or text-formatted structured logging with context variables
for request tracing (Request ID, location_id, model_name).
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Context variable for distributed request tracking across async execution
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id_ctx", default=None)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured cloud-ready logging (CloudWatch, ELK, Datadog)."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # Include request_id if available in context
        req_id = request_id_ctx.get()
        if req_id:
            log_payload["request_id"] = req_id

        # Attach custom extra fields if provided
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_payload.update(record.extra_data)

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload)


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure root and application loggers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if log_format.lower() == "json":
        handler.setFormatter(JSONFormatter())
    else:
        text_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        handler.setFormatter(logging.Formatter(text_format))

    root_logger.addHandler(handler)

    # Silence overly verbose external loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Retrieve logger instance with standardized name."""
    return logging.getLogger(name)
