# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/common/logging_utils.py."""

import json
import logging

from src.common.logging_utils import JsonFormatter, configure_logging, log_with_context


def _make_record(message: str, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=message, args=(), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_produces_valid_single_line_json():
    formatter = JsonFormatter()
    record = _make_record("hello", camera_id="cam-01")
    output = formatter.format(record)

    assert "\n" not in output
    parsed = json.loads(output)
    assert parsed["message"] == "hello"
    assert parsed["camera_id"] == "cam-01"
    assert parsed["level"] == "INFO"


def test_json_formatter_includes_exception_traceback():
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="failed", args=(), exc_info=sys.exc_info(),
        )
    output = json.loads(formatter.format(record))
    assert "ValueError: boom" in output["exception"]


def test_configure_logging_is_idempotent_across_repeated_calls():
    """Calling `configure_logging` twice for the same service must not

    duplicate handlers -- otherwise every log line would print N times
    after N calls (a real bug this test would have caught immediately).
    """
    logger_a = configure_logging("idempotent-test-service")
    logger_b = configure_logging("idempotent-test-service")
    assert logger_a is logger_b
    assert len(logger_a.handlers) == 1


def test_log_with_context_omits_unset_fields(caplog):
    logger = configure_logging("context-test-service")

    # `configure_logging` sets `propagate = False` on purpose (so a service's
    # own handler doesn't also fire through the root logger's handlers and
    # print every line twice) -- but pytest's `caplog` fixture only captures
    # records that reach the *root* logger, regardless of the `logger=`
    # passed to `at_level`. Without propagation, `caplog.records` would stay
    # empty and `caplog.records[-1]` below would raise `IndexError`. Flip it
    # on just for this test, and restore it so other tests aren't affected.
    original_propagate = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger="context-test-service"):
            log_with_context(logger, logging.INFO, "frame processed", camera_id="cam-02")
    finally:
        logger.propagate = original_propagate

    record = caplog.records[-1]
    assert record.camera_id == "cam-02"
    assert not hasattr(record, "track_id")
