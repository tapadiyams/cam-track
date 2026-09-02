# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Resilient frame reader over an RTSP stream (or a local file/webcam for

local dev). Wraps OpenCV's `VideoCapture`, which itself wraps GStreamer /
FFmpeg depending on platform build -- see
docs/decisions/0005-edge-vs-cloud.md for why we don't hand-roll a GStreamer
pipeline for v1.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from src.config.constants import TimeoutConstants

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapturedFrame:
    """One frame read from a source, before it is written to frame storage."""

    image: np.ndarray
    sequence_number: int
    captured_at_ms: int


class RtspFrameReader:
    """Reads frames from `source`, reconnecting on failure with backoff.

    A camera going offline (network blip, power cycle) must not crash the
    ingestion worker or silently stop producing frames forever -- both are
    common failure modes of a naive `while cap.read(): ...` loop. This
    class instead treats a read failure as a signal to reconnect, capped by
    exponential backoff so a genuinely dead camera does not spin the CPU.
    """

    def __init__(self, source: str, fps_cap: int) -> None:
        self._source = source
        self._min_frame_interval_s = 1.0 / fps_cap
        self._capture: cv2.VideoCapture | None = None
        self._sequence_number = 0

    def _connect(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self._source)
        connect_timeout_ms = TimeoutConstants.RTSP_CONNECT_TIMEOUT_SECONDS * 1000
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, connect_timeout_ms)
        if not capture.isOpened():
            capture.release()
            raise ConnectionError(f"Could not open camera source: {self._source}")
        return capture

    def frames(self) -> Iterator[CapturedFrame]:
        """Yield frames indefinitely, reconnecting on any read failure.

        Time: O(1) amortized per yielded frame (one `cap.read()` plus a
        constant-time fps-cap check); frames arriving faster than
        `fps_cap` are dropped, not queued, so this reader never builds
        unbounded backlog under load.
        Space: O(1) -- one frame buffer at a time.
        """
        reconnect_attempt = 0
        # `None`, not `0.0`: `time.monotonic()`'s reference point is
        # platform-defined and not guaranteed to start near zero, so
        # seeding this with `0.0` risks silently dropping the very first
        # frame if the real clock value happens to be small.
        last_emitted_at: float | None = None

        while True:
            if self._capture is None:
                try:
                    self._capture = self._connect()
                    reconnect_attempt = 0
                except ConnectionError:
                    delay = min(
                        TimeoutConstants.RTSP_RECONNECT_BACKOFF_BASE_SECONDS
                        * (2**reconnect_attempt),
                        TimeoutConstants.RTSP_RECONNECT_BACKOFF_MAX_SECONDS,
                    )
                    logger.warning(
                        "camera source unreachable, retrying",
                        extra={"source": self._source, "retry_in_s": delay},
                    )
                    time.sleep(delay)
                    reconnect_attempt += 1
                    continue

            ok, frame = self._capture.read()
            if not ok:
                logger.warning("frame read failed, reconnecting", extra={"source": self._source})
                self._capture.release()
                self._capture = None
                continue

            now = time.monotonic()
            elapsed_since_last = None if last_emitted_at is None else now - last_emitted_at
            if elapsed_since_last is not None and elapsed_since_last < self._min_frame_interval_s:
                continue  # drop this frame to respect the fps cap
            last_emitted_at = now

            self._sequence_number += 1
            yield CapturedFrame(
                image=frame,
                sequence_number=self._sequence_number,
                captured_at_ms=int(time.time() * 1000),
            )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
