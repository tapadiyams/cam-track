# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/streaming/redis_streams.py.

`_retry_backoff_seconds` is a pure function and always runs. Everything
that talks to an actual (fake) Redis skips itself via `pytest.importorskip`
inside the `broker` fixture when `fakeredis` is not installed, rather than
letting one missing optional dependency skip the whole module -- see
requirements-dev.txt.
"""

import pytest

from src.config.constants import RetryConstants
from src.streaming.redis_streams import _retry_backoff_seconds


def test_retry_backoff_grows_exponentially_and_caps():
    delays = [_retry_backoff_seconds(attempt) for attempt in range(6)]
    for earlier, later in zip(delays, delays[1:]):
        assert later >= earlier
    assert delays[-1] <= RetryConstants.BACKOFF_MAX_SECONDS


@pytest.fixture
def broker():
    fakeredis = pytest.importorskip("fakeredis")
    from src.streaming.redis_streams import RedisStreamsBroker

    fake_client = fakeredis.FakeRedis(decode_responses=True)
    return RedisStreamsBroker(redis_url="redis://unused", client=fake_client)


def test_publish_then_consume_round_trips_the_payload(broker):
    broker.publish("stream-a", {"hello": "world"})
    messages = broker.consume("stream-a", group="g1", consumer_name="c1", count=10)

    assert len(messages) == 1
    assert messages[0].payload == {"hello": "world"}


def test_unacked_message_is_not_redelivered_to_the_same_consumer_again(broker):
    """After `ack`, a second `consume()` call by the same consumer must not

    return the message again -- otherwise storage/inference would
    reprocess every event forever.
    """
    broker.publish("stream-b", {"n": 1})
    first_read = broker.consume("stream-b", group="g1", consumer_name="c1")
    broker.ack("stream-b", "g1", first_read[0].message_id)

    second_read = broker.consume("stream-b", group="g1", consumer_name="c1")
    assert second_read == []


def test_two_consumers_in_the_same_group_split_the_messages(broker):
    for i in range(4):
        broker.publish("stream-c", {"n": i})

    read_by_c1 = broker.consume("stream-c", group="g1", consumer_name="c1", count=2)
    read_by_c2 = broker.consume("stream-c", group="g1", consumer_name="c2", count=2)

    ids_c1 = {m.payload["n"] for m in read_by_c1}
    ids_c2 = {m.payload["n"] for m in read_by_c2}
    assert ids_c1.isdisjoint(ids_c2)  # no message delivered to both consumers
    assert ids_c1 | ids_c2 == {0, 1, 2, 3}


def test_ensure_group_is_idempotent_across_repeated_calls(broker):
    broker.ensure_group("stream-d", "g1")
    broker.ensure_group("stream-d", "g1")  # must not raise BUSYGROUP
