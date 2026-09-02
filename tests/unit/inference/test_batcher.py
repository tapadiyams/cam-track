# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/inference/batcher.py -- the size/time dual-bound flush."""

import threading
import time

import pytest

from src.inference.batcher import DynamicBatcher


def test_flushes_on_max_batch_size_without_waiting_for_timeout():
    flushed = []
    flush_event = threading.Event()

    def on_batch(items):
        flushed.append(items)
        flush_event.set()

    batcher = DynamicBatcher(on_batch=on_batch, max_batch_size=3, max_wait_ms=60_000)
    batcher.start()
    try:
        for item in ["a", "b", "c"]:
            batcher.submit(item)
        assert flush_event.wait(timeout=2.0), "batch never flushed on size trigger"
    finally:
        batcher.stop()

    assert flushed == [["a", "b", "c"]]


def test_flushes_on_timeout_with_a_partial_batch():
    flushed = []
    flush_event = threading.Event()

    def on_batch(items):
        flushed.append(items)
        flush_event.set()

    batcher = DynamicBatcher(on_batch=on_batch, max_batch_size=100, max_wait_ms=50)
    batcher.start()
    try:
        batcher.submit("only-item")
        assert flush_event.wait(timeout=2.0), "batch never flushed on timeout"
    finally:
        batcher.stop()

    assert flushed == [["only-item"]]


def test_stop_flushes_any_pending_partial_batch():
    flushed = []
    batcher = DynamicBatcher(
        on_batch=lambda items: flushed.append(items),
        max_batch_size=100,
        max_wait_ms=60_000,  # so only stop() -- not the timeout -- can flush
    )
    batcher.start()
    batcher.submit("leftover")
    time.sleep(0.05)  # give the background thread a chance to enqueue it
    batcher.stop(timeout_s=2.0)

    assert flushed == [["leftover"]]


def test_stop_with_no_pending_items_does_not_hang():
    """A batcher that never received any items must still stop promptly --

    this is the regression test for a real bug where an idle batcher
    blocked forever on `Queue.get(timeout=None)` and never noticed the
    stop signal.
    """
    batcher = DynamicBatcher(on_batch=lambda items: None, max_batch_size=10, max_wait_ms=1000)
    batcher.start()

    stopped = threading.Event()

    def do_stop():
        batcher.stop(timeout_s=2.0)
        stopped.set()

    thread = threading.Thread(target=do_stop)
    thread.start()
    thread.join(timeout=3.0)

    assert stopped.is_set(), "stop() hung on an idle batcher"


def test_rejects_invalid_construction_arguments():
    with pytest.raises(ValueError):
        DynamicBatcher(on_batch=lambda items: None, max_batch_size=0)
    with pytest.raises(ValueError):
        DynamicBatcher(on_batch=lambda items: None, max_wait_ms=-1)
