# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Dynamic batching: trade a small amount of latency for GPU throughput.

Running the detector on one frame at a time wastes most of a GPU's
parallelism; running it on a fixed batch size adds latency when traffic is
light (waiting to fill a batch that may never fill). `DynamicBatcher` flushes
on whichever bound is hit first -- `max_batch_size` frames collected, or
`max_wait_ms` elapsed since the first frame in the pending batch arrived --
which keeps p99 latency bounded even under low load while still batching
under high load. This is the same idea NVIDIA Triton's dynamic batcher and
TensorFlow Serving's batching config implement.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar

from src.config.constants import InferenceConstants

T = TypeVar("T")


class DynamicBatcher(Generic[T]):
    """Accumulates items from `submit()` and flushes batches to `on_batch`.

    `on_batch` runs on the batcher's own background thread, not the
    caller's -- callers only block briefly inside `submit()` to enqueue,
    never for the duration of a batch's inference call.
    """

    def __init__(
        self,
        on_batch: Callable[[list[T]], None],
        max_batch_size: int = InferenceConstants.DEFAULT_MAX_BATCH_SIZE,
        max_wait_ms: int = InferenceConstants.DEFAULT_MAX_BATCH_WAIT_MILLISECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")

        self._on_batch = on_batch
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._clock = clock
        self._queue: queue.Queue[T] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, item: T) -> None:
        """Enqueue `item`. Time: O(1) amortized. Space: O(1) per call."""
        self._queue.put(item)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout_s)

    # Upper bound on any single `Queue.get()` call, regardless of how much
    # of `max_wait_ms` is still remaining for the current partial batch.
    # Without this cap, a long `max_wait_ms` (say 60s) with one item
    # already pending would block the background thread inside a single
    # `Queue.get(timeout=60)` call, and `stop()` -- which only sets a flag
    # the loop checks *between* calls -- would not be noticed for up to
    # 60s. Capping every poll at this interval keeps `stop()` responsive
    # (checked at least this often) without changing *when* a batch
    # actually flushes, which is still decided by `_time_left` below.
    _MAX_POLL_INTERVAL_S = 0.1

    def _run(self) -> None:
        pending: list[T] = []
        batch_started_at: float | None = None

        while not self._stop_event.is_set():
            remaining_wait_s = self._time_left(batch_started_at)
            poll_timeout = (
                min(remaining_wait_s, self._MAX_POLL_INTERVAL_S)
                if remaining_wait_s is not None
                else self._MAX_POLL_INTERVAL_S
            )
            try:
                item = self._queue.get(timeout=poll_timeout)
            except queue.Empty:
                # A poll can time out long before the batch's real deadline
                # (it is capped at `_MAX_POLL_INTERVAL_S`); only flush once
                # the actual remaining wait has reached zero.
                if pending and self._time_left(batch_started_at) <= 0:
                    self._flush(pending)
                    pending = []
                    batch_started_at = None
                continue

            if not pending:
                batch_started_at = self._clock()
            pending.append(item)

            if len(pending) >= self._max_batch_size:
                self._flush(pending)
                pending = []
                batch_started_at = None

        self._flush(pending)

    def _time_left(self, batch_started_at: float | None) -> float | None:
        if batch_started_at is None:
            return None
        elapsed = self._clock() - batch_started_at
        return max(0.0, self._max_wait_s - elapsed)

    def _flush(self, pending: list[T]) -> None:
        # Time: O(1) to hand off; the actual O(batch) inference cost is
        # inside `on_batch`, which the caller supplies and owns.
        if pending:
            self._on_batch(list(pending))
