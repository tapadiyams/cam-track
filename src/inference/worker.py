# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Inference worker: raw frames in, tracked-object events out.

Pipeline per frame: load image -> batch across cameras for the detector ->
per-camera ByteTrack association -> (optional) re-ID embedding -> publish
`TrackEvent`s -> ack the raw frame. Horizontal scaling is "run more of this
process" -- each worker reads from the same `ConsumerGroups.INFERENCE_
WORKERS` consumer group, so Redis Streams / Kafka hands each raw frame to
exactly one worker and adding workers linearly increases throughput up to
the broker's own limits. See docs/decisions/0005-edge-vs-cloud.md.
"""

from __future__ import annotations

import logging

from src.common.schemas import RawFrame, TrackEvent
from src.config.constants import ConsumerGroups, StreamNames
from src.inference.batcher import DynamicBatcher
from src.inference.detector import Detector
from src.inference.frame_loader import FrameLoader
from src.inference.tracker import ByteTracker, new_tracker
from src.streaming.base import MessageBroker

logger = logging.getLogger(__name__)


class InferenceWorker:
    """Drains `StreamNames.RAW_FRAMES`, emits `StreamNames.TRACK_EVENTS`."""

    def __init__(
        self,
        broker: MessageBroker,
        detector: Detector,
        frame_loader: FrameLoader,
        consumer_name: str,
        max_batch_size: int,
        max_batch_wait_ms: int,
    ) -> None:
        self._broker = broker
        self._detector = detector
        self._frame_loader = frame_loader
        self._consumer_name = consumer_name
        self._trackers: dict[str, ByteTracker] = {}
        self._batcher: DynamicBatcher[tuple[str, RawFrame]] = DynamicBatcher(
            on_batch=self._process_batch,
            max_batch_size=max_batch_size,
            max_wait_ms=max_batch_wait_ms,
        )

    def _tracker_for(self, camera_id: str) -> ByteTracker:
        if camera_id not in self._trackers:
            self._trackers[camera_id] = new_tracker(camera_id)
        return self._trackers[camera_id]

    def _process_batch(self, items: list[tuple[str, RawFrame]]) -> None:
        # Time: O(b) to load images plus one batched detector call
        # (amortized O(1) per frame on the GPU/ONNX runtime side) plus O(b)
        # ByteTrack updates, each independent per camera. Space: O(b)
        # decoded images held at once, bounded by `max_batch_size`.
        message_ids = [message_id for message_id, _frame in items]
        frames = [frame for _message_id, frame in items]
        images = [self._frame_loader.load(frame.frame_uri) for frame in frames]

        detections_per_frame = self._detector.detect_batch(
            frame_ids=[f.frame_id for f in frames],
            camera_ids=[f.camera_id for f in frames],
            images=images,
        )

        for message_id, frame, detections in zip(
            message_ids, frames, detections_per_frame, strict=True
        ):
            tracker = self._tracker_for(frame.camera_id)
            tracks = tracker.update(detections, frame.frame_id)
            for track in tracks:
                event = TrackEvent(track=track)
                self._broker.publish(StreamNames.TRACK_EVENTS, event.model_dump())
            self._broker.ack(
                StreamNames.RAW_FRAMES, ConsumerGroups.INFERENCE_WORKERS, message_id
            )

    def run_forever(self) -> None:
        self._batcher.start()
        logger.info("inference worker started", extra={"consumer": self._consumer_name})
        try:
            while True:
                messages = self._broker.consume(
                    StreamNames.RAW_FRAMES,
                    ConsumerGroups.INFERENCE_WORKERS,
                    self._consumer_name,
                )
                for message in messages:
                    frame = RawFrame.model_validate(message.payload)
                    self._batcher.submit((message.message_id, frame))
        finally:
            self._batcher.stop()
