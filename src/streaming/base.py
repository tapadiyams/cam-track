# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Broker-agnostic producer/consumer interface.

Every backend (Redis Streams, Kafka) implements this ABC so ingestion,
inference, and storage code depends only on `MessageBroker`, never on a
specific client library. Swapping backends -- or running both side by side
during a migration -- means changing `settings.stream_backend`, not any
call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Message:
    """One consumed message: broker-assigned id plus the decoded payload."""

    message_id: str
    payload: dict[str, Any]


class MessageBroker(ABC):
    """Durable, ordered, at-least-once pub/sub over a named stream/topic.

    At-least-once, not exactly-once: consumers must be idempotent (storage
    writes key on `event_id`, which is safe to upsert twice). This matches
    what both Redis Streams consumer groups and Kafka consumer groups
    natively guarantee without a distributed transaction layer.
    """

    @abstractmethod
    def publish(self, stream: str, payload: dict[str, Any]) -> str:
        """Publish `payload` to `stream`. Returns the broker-assigned id."""

    @abstractmethod
    def consume(
        self, stream: str, group: str, consumer_name: str, count: int = 10
    ) -> list[Message]:
        """Read up to `count` pending messages for `group` from `stream`.

        Blocks briefly (bounded by
        `TimeoutConstants.STREAM_READ_BLOCK_MILLISECONDS`) if nothing is
        available rather than busy-polling; returns an empty list on
        timeout rather than raising.
        """

    @abstractmethod
    def ack(self, stream: str, group: str, message_id: str) -> None:
        """Acknowledge successful processing of `message_id`.

        Only after `ack` does the broker consider the message delivered;
        an unacked message is redelivered to another consumer in the group
        after its visibility timeout, which is what makes at-least-once
        delivery hold even if a worker crashes mid-processing.
        """

    @abstractmethod
    def ensure_group(self, stream: str, group: str) -> None:
        """Create `group` on `stream` if it does not already exist."""
