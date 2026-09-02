# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Structured (JSON) logging setup shared by every service.

Plain-text logs are fine on a laptop but do not survive being shipped to a
log aggregator (ELK, Loki, CloudWatch) with per-field search -- so every
service logs one JSON object per line instead. See
docs/decisions/0005-edge-vs-cloud.md for why this matters more once
inference workers run on distributed edge devices rather than one host.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from src.config.constants import LogFields


class JsonFormatter(logging.Formatter):
    """Renders each `LogRecord` as a single-line JSON object.

    Time: O(k) per record, where k is the number of structured fields
    attached via `extra=` -- always small and bounded, never input-sized.
    Space: O(k) for the serialized dict.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in vars(record).items():
            if key in _RESERVED_RECORD_FIELDS:
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


_RESERVED_RECORD_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


def configure_logging(service_name: str, level: str = "INFO") -> logging.Logger:
    """Configure and return the root logger for `service_name`.

    Idempotent: safe to call more than once (e.g. in tests) without
    duplicating handlers.
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(level.upper())
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    camera_id: str | None = None,
    frame_id: str | None = None,
    track_id: str | None = None,
    latency_ms: float | None = None,
    **extra: Any,
) -> None:
    """Log `message` with the standard structured fields attached.

    Thin convenience wrapper so call sites don't repeat `LogFields.*` keys;
    unset fields are omitted rather than logged as `null` noise.
    """
    fields = {
        LogFields.CAMERA_ID: camera_id,
        LogFields.FRAME_ID: frame_id,
        LogFields.TRACK_ID: track_id,
        LogFields.LATENCY_MS: latency_ms,
        **extra,
    }
    logger.log(level, message, extra={k: v for k, v in fields.items() if v is not None})
