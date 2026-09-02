# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Kafka implementation of `MessageBroker` -- the alternate backend.

Kept behind the same interface as `RedisStreamsBroker` for teams that
already run Kafka, need cross-datacenter replication (MirrorMaker), or
expect the frame/event volume to outgrow a single Redis instance's memory.
See docs/decisions/0003-message-queue-choice.md for when to pick this over
the Redis default.

Requires the optional `confluent-kafka` dependency; imported lazily inside
`__init__` so the rest of the codebase does not pay for `librdkafka` unless
this backend is actually selected (`STREAM_BACKEND=kafka`).
"""

from __future__ import annotations

from typing import Any

from src.config.constants import TimeoutConstants
from src.streaming.base import Message, MessageBroker


class KafkaBroker(MessageBroker):
    """`MessageBroker` backed by Kafka topics and a consumer group."""

    def __init__(self, bootstrap_servers: str, group_id_default: str = "camtrack") -> None:
        try:
            from confluent_kafka import Consumer, Producer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "KafkaBroker requires the optional 'confluent-kafka' dependency. "
                "Install it with: pip install cam-track[kafka]"
            ) from exc

        self._bootstrap_servers = bootstrap_servers
        self._producer = Producer({"bootstrap.servers": bootstrap_servers})
        self._consumer_cls = Consumer
        self._consumers: dict[str, Any] = {}
        self._known_groups: set[tuple[str, str]] = set()
        self._group_id_default = group_id_default

    def _consumer_for(self, group: str) -> Any:
        if group not in self._consumers:
            self._consumers[group] = self._consumer_cls(
                {
                    "bootstrap.servers": self._bootstrap_servers,
                    "group.id": group,
                    "enable.auto.commit": False,  # we ack explicitly, matching Redis semantics
                    "auto.offset.reset": "earliest",
                }
            )
        return self._consumers[group]

    def publish(self, stream: str, payload: dict[str, Any]) -> str:
        import json

        record_key = payload.get("frame_id") or payload.get("event_id") or ""
        self._producer.produce(topic=stream, key=record_key, value=json.dumps(payload))
        self._producer.flush(timeout=TimeoutConstants.HTTP_CLIENT_TIMEOUT_SECONDS)
        return record_key

    def consume(
        self, stream: str, group: str, consumer_name: str, count: int = 10
    ) -> list[Message]:
        import json

        self.ensure_group(stream, group)
        consumer = self._consumer_for(group)
        messages: list[Message] = []
        deadline_s = TimeoutConstants.STREAM_READ_BLOCK_MILLISECONDS / 1000.0
        records = consumer.consume(num_messages=count, timeout=deadline_s)
        for record in records or []:
            if record is None or record.error():
                continue
            offset_id = f"{record.topic()}:{record.partition()}:{record.offset()}"
            messages.append(Message(message_id=offset_id, payload=json.loads(record.value())))
        return messages

    def ack(self, stream: str, group: str, message_id: str) -> None:
        # Kafka acks by committing the consumer's offset, not by message
        # id, so this is a coarser guarantee than Redis's per-message XACK:
        # committing advances the whole partition's offset for the group.
        consumer = self._consumers.get(group)
        if consumer is not None:
            consumer.commit(asynchronous=False)

    def ensure_group(self, stream: str, group: str) -> None:
        # Kafka creates consumer groups implicitly on first subscribe/poll;
        # this just makes sure the consumer object (and its subscription)
        # exists before the first `consume()` call.
        key = (stream, group)
        if key in self._known_groups:
            return
        consumer = self._consumer_for(group)
        consumer.subscribe([stream])
        self._known_groups.add(key)


__all__ = ["KafkaBroker"]
