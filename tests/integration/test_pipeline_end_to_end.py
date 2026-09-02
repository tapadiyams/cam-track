# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""End-to-end pipeline test, in-process, no network/DB/broker involved.

Exercises the actual data flow -- detections in, tracked+re-identified
events out -- across the module boundaries that matter (tracker -> re-ID),
using an in-memory `MessageBroker` fake instead of real Redis/Kafka so this
test has zero external dependencies and runs anywhere `pytest` does.
"""

from src.common.schemas import BoundingBox, Detection, TrackEvent
from src.config.constants import StreamNames
from src.inference.tracker import new_tracker
from src.reid.cross_camera_matcher import CrossCameraMatcher
from src.reid.embedder import ColorHistogramEmbedder
from src.streaming.base import Message, MessageBroker


class InMemoryBroker(MessageBroker):
    """A `MessageBroker` backed by plain Python lists/dicts. Test-only."""

    def __init__(self):
        self._streams: dict[str, list[dict]] = {}
        self._next_id = 0

    def publish(self, stream, payload):
        self._streams.setdefault(stream, []).append(payload)
        self._next_id += 1
        return str(self._next_id)

    def consume(self, stream, group, consumer_name, count=10):
        payloads = self._streams.get(stream, [])
        return [Message(message_id=str(i), payload=p) for i, p in enumerate(payloads)][:count]

    def ack(self, stream, group, message_id):
        pass

    def ensure_group(self, stream, group):
        pass


def _det(x1, y1, x2, y2, confidence=0.9):
    return Detection(
        frame_id="f",
        camera_id="cam-01",
        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_id=0,
        class_name="person",
        confidence=confidence,
    )


def _uniform_crop(bgr_value, size=32):
    import numpy as np

    crop = np.zeros((size, size, 3), dtype=np.uint8)
    crop[:, :] = bgr_value
    return crop


def test_tracked_object_flows_through_to_a_published_track_event():
    """Detections in one camera's frames should turn into `TrackEvent`s on

    the broker with a stable `track_id`, matching what
    `src/inference/worker.py` actually does per frame.
    """
    broker = InMemoryBroker()
    tracker = new_tracker("cam-01")

    for frame_num in range(3):
        detections = [_det(frame_num, frame_num, frame_num + 20, frame_num + 20)]
        tracks = tracker.update(detections, frame_id=f"f{frame_num}")
        for track in tracks:
            event = TrackEvent(track=track)
            broker.publish(StreamNames.TRACK_EVENTS, event.model_dump())

    published = [
        TrackEvent.model_validate(m.payload)
        for m in broker.consume(StreamNames.TRACK_EVENTS, "g", "c", count=100)
    ]

    assert len(published) == 3
    track_ids = {e.track.track_id for e in published}
    assert len(track_ids) == 1  # same object, one continuous identity


def test_cross_camera_reid_links_the_same_object_seen_on_two_cameras():
    """The other half of the pipeline: two cameras' independent trackers

    each produce their own local track for the same physical object, and
    the re-ID matcher is what ties them to one global identity -- this is
    the actual cross-camera capability the project brief calls out as the
    differentiating piece, so it is worth an explicit integration check
    beyond the per-module unit tests.
    """
    tracker_cam1 = new_tracker("cam-01")
    tracker_cam2 = new_tracker("cam-02")
    embedder = ColorHistogramEmbedder()
    matcher = CrossCameraMatcher()

    # Same physical object (same appearance), seen first by camera 1...
    tracks_cam1 = tracker_cam1.update([_det(0, 0, 20, 20)], frame_id="f0")
    embedding_cam1 = embedder.embed(_uniform_crop((10, 20, 200)))
    first_match = matcher.observe(
        "zone-x", "cam-01", tracks_cam1[0].track_id, embedding_cam1
    )

    # ...then by camera 2, moments later.
    tracks_cam2 = tracker_cam2.update([_det(500, 500, 520, 520)], frame_id="f0")
    embedding_cam2 = embedder.embed(_uniform_crop((10, 20, 200)))  # same appearance
    second_match = matcher.observe(
        "zone-x", "cam-02", tracks_cam2[0].track_id, embedding_cam2
    )

    assert first_match is None  # nothing to match against yet
    assert second_match is not None
    assert second_match.matched_camera_id == "cam-01"
    assert second_match.source_camera_id == "cam-02"
    assert second_match.matched_track_id == tracks_cam1[0].track_id
    assert second_match.source_track_id == tracks_cam2[0].track_id
