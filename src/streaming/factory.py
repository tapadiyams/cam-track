# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Selects a `MessageBroker` implementation from `settings.stream_backend`."""

from __future__ import annotations

from src.config.settings import Settings
from src.streaming.base import MessageBroker


def get_broker(settings: Settings) -> MessageBroker:
    """Construct the configured broker backend.

    Time/Space: O(1) -- delegates to the chosen backend's constructor.
    """
    if settings.stream_backend == "redis":
        from src.streaming.redis_streams import RedisStreamsBroker

        return RedisStreamsBroker(settings.redis_url)
    if settings.stream_backend == "kafka":
        from src.streaming.kafka_backend import KafkaBroker

        return KafkaBroker(settings.kafka_bootstrap_servers)
    raise ValueError(
        f"Unknown stream_backend {settings.stream_backend!r}; expected 'redis' or 'kafka'."
    )
