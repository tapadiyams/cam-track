# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Redis Streams implementation of `MessageBroker` -- the default backend.

Redis Streams gives us consumer groups (each message delivered to exactly
one worker in a group, with per-consumer pending-entry lists for crash
recovery) plus sub-millisecond publish latency, using infrastructure most
teams already run for caching. See
docs/decisions/0003-message-queue-choice.md for the full comparison against
Kafka.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from src.config.constants import RetryConstants, TimeoutConstants
from src.streaming.base import Message, MessageBroker

if TYPE_CHECKING:
    import redis


def _retry_backoff_seconds(attempt: int) -> float:
    """Exponential backoff, capped, for the `attempt`-th retry (0-indexed)."""
    delay = RetryConstants.BACKOFF_BASE_SECONDS * (RetryConstants.BACKOFF_MULTIPLIER**attempt)
    return min(delay, RetryConstants.BACKOFF_MAX_SECONDS)


class RedisStreamsBroker(MessageBroker):
    """`MessageBroker` backed by a single Redis instance's Streams."""

    def __init__(self, redis_url: str, client: "redis.Redis | None" = None) -> None:
        # `client` is injectable so tests can pass a fakeredis instance
        # instead of monkeypatching the `redis` module. The `redis` import
        # itself is deferred to here (and to the methods below) so this
        # module -- and the pure `_retry_backoff_seconds` helper above --
        # stay importable without the `redis` package installed, matching
        # the lazy-import pattern used for the optional Kafka backend.
        if client is not None:
            self._redis = client
        else:
            import redis as redis_module

            self._redis = redis_module.Redis.from_url(redis_url, decode_responses=True)
        self._known_groups: set[tuple[str, str]] = set()

    def publish(self, stream: str, payload: dict[str, Any]) -> str:
        # Time: O(1) amortized on the Redis side (append to the stream's
        # backing radix tree); Space: O(len(payload)) for the serialized
        # fields, held both client- and server-side.
        #
        # Retries only `ConnectionError` (broker unreachable) -- never a
        # partial/ambiguous failure -- because XADD is not idempotent by
        # itself; a retried publish after a genuine ambiguous timeout could
        # duplicate a message, which downstream consumers must already
        # tolerate (see the at-least-once note on `MessageBroker`).
        import redis as redis_module

        fields = {"data": json.dumps(payload, default=str)}
        last_error: redis_module.ConnectionError | None = None
        for attempt in range(RetryConstants.MAX_RETRIES):
            try:
                message_id: str = self._redis.xadd(stream, fields)
                return message_id
            except redis_module.ConnectionError as exc:
                last_error = exc
                time.sleep(_retry_backoff_seconds(attempt))
        assert last_error is not None
        raise last_error

    def consume(
        self, stream: str, group: str, consumer_name: str, count: int = 10
    ) -> list[Message]:
        self.ensure_group(stream, group)
        # XREADGROUP with id ">" returns only messages never delivered to
        # any consumer in this group -- redelivery of a crashed consumer's
        # pending entries is a separate, explicit reclaim path (not
        # implemented here; see the worker's XPENDING/XCLAIM TODO in
        # src/inference/worker.py for where that hook belongs).
        response = self._redis.xreadgroup(
            groupname=group,
            consumername=consumer_name,
            streams={stream: ">"},
            count=count,
            block=TimeoutConstants.STREAM_READ_BLOCK_MILLISECONDS,
        )
        if not response:
            return []

        messages: list[Message] = []
        for _stream_name, entries in response:
            for entry_id, fields in entries:
                payload = json.loads(fields["data"])
                messages.append(Message(message_id=entry_id, payload=payload))
        return messages

    def ack(self, stream: str, group: str, message_id: str) -> None:
        self._redis.xack(stream, group, message_id)

    def ensure_group(self, stream: str, group: str) -> None:
        # Time: O(1) after the first call per (stream, group) pair thanks
        # to the local cache; avoids a round trip to Redis on every
        # `consume()` call, which would otherwise happen once per poll.
        import redis as redis_module

        key = (stream, group)
        if key in self._known_groups:
            return
        try:
            self._redis.xgroup_create(name=stream, groupname=group, id="0", mkstream=True)
        except redis_module.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._known_groups.add(key)


__all__ = ["RedisStreamsBroker"]
