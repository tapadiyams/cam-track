# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/config/settings.py."""

import pytest

from src.config.settings import Settings, get_settings


def test_settings_have_safe_local_dev_defaults(monkeypatch):
    """Every field must have a default so a bare `docker-compose up` works;

    this is a documented promise in the module docstring, not just a
    convenience -- regression here breaks the zero-config demo path.
    """
    for key in (
        "STREAM_BACKEND",
        "REDIS_URL",
        "KAFKA_BOOTSTRAP_SERVERS",
        "TIMESCALE_DSN",
        "DETECTOR_WEIGHTS_PATH",
        "CAMERA_CONFIG_PATH",
        "DASHBOARD_PORT",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.stream_backend == "redis"
    assert settings.dashboard_port == 8080


def test_get_settings_reads_environment_overrides(monkeypatch):
    monkeypatch.setenv("STREAM_BACKEND", "kafka")
    monkeypatch.setenv("MAX_BATCH_SIZE", "32")
    settings = get_settings()
    assert settings.stream_backend == "kafka"
    assert settings.max_batch_size == 32


def test_get_settings_rejects_invalid_types(monkeypatch):
    """A non-integer MAX_BATCH_SIZE must fail fast at startup, not silently

    coerce to something wrong or crash later inside the batcher.
    """
    monkeypatch.setenv("MAX_BATCH_SIZE", "not-a-number")
    with pytest.raises(Exception):
        get_settings()
