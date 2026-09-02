# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Wires a `RtspFrameReader` to a `FrameStore` and a `MessageBroker`.

One `FramePublisher` instance runs one camera's ingest loop: read a frame,
persist it, publish a reference. Kept separate from `RtspFrameReader` so
the reconnect/backoff logic can be unit-tested without touching a real
frame store or broker.
"""

from __future__ import annotations

import logging

from src.common.schemas import RawFrame
from src.config.constants import StreamNames
from src.ingestion.camera_config import CameraConfig
from src.ingestion.frame_store import FrameStore
from src.ingestion.rtsp_reader import RtspFrameReader
from src.streaming.base import MessageBroker

logger = logging.getLogger(__name__)


class FramePublisher:
    """Runs the read -> persist -> publish loop for a single camera."""

    def __init__(
        self,
        camera: CameraConfig,
        reader: RtspFrameReader,
        frame_store: FrameStore,
        broker: MessageBroker,
    ) -> None:
        self._camera = camera
        self._reader = reader
        self._frame_store = frame_store
        self._broker = broker

    def run_forever(self) -> None:
        """Consume `self._reader.frames()` until the process is stopped.

        Time per frame: O(1) plus the cost of `FrameStore.save` (a JPEG
        encode, roughly linear in pixel count but bounded by the camera's
        fixed resolution) and one broker publish. Space: O(1) -- frames
        are processed one at a time, never buffered as a list.
        """
        for captured in self._reader.frames():
            frame = RawFrame(
                camera_id=self._camera.camera_id,
                zone=self._camera.zone,
                captured_at_ms=captured.captured_at_ms,
                frame_uri="",  # filled in below, after we know the frame_id
                width_px=captured.image.shape[1],
                height_px=captured.image.shape[0],
                sequence_number=captured.sequence_number,
            )
            frame_uri = self._frame_store.save(
                self._camera.camera_id, frame.frame_id, captured.image
            )
            frame = frame.model_copy(update={"frame_uri": frame_uri})

            self._broker.publish(StreamNames.RAW_FRAMES, frame.model_dump())
            logger.debug(
                "published frame",
                extra={"camera_id": self._camera.camera_id, "frame_id": frame.frame_id},
            )

    def stop(self) -> None:
        self._reader.close()
