# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/common/schemas.py -- the wire format every service shares."""

import pytest
from pydantic import ValidationError

from src.common.schemas import (
    BoundingBox,
    Detection,
    RawFrame,
    Track,
    TrackEvent,
    TrackState,
)


def test_bounding_box_geometry_helpers():
    box = BoundingBox(x1=10, y1=20, x2=50, y2=80)
    assert box.width == 40
    assert box.height == 60
    assert box.area == 2400
    assert box.center == (30, 50)
    assert box.as_xyxy() == (10, 20, 50, 80)


def test_bounding_box_degenerate_box_has_zero_area_not_negative():
    """A box where x2 < x1 (a bad detection) must clamp to zero, not

    produce a negative area that would corrupt downstream IoU math.
    """
    box = BoundingBox(x1=50, y1=50, x2=10, y2=10)
    assert box.width == 0
    assert box.height == 0
    assert box.area == 0


def test_detection_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        Detection(
            frame_id="f1",
            camera_id="cam-01",
            box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            class_id=0,
            class_name="person",
            confidence=1.5,
        )


def test_raw_frame_generates_unique_ids_and_timestamps():
    frame_a = RawFrame(
        camera_id="cam-01", zone="store-front", frame_uri="file:///a.jpg",
        width_px=1920, height_px=1080, sequence_number=1,
    )
    frame_b = RawFrame(
        camera_id="cam-01", zone="store-front", frame_uri="file:///b.jpg",
        width_px=1920, height_px=1080, sequence_number=2,
    )
    assert frame_a.frame_id != frame_b.frame_id
    assert frame_a.captured_at_ms > 0


def test_track_event_round_trips_through_json():
    """The wire format must survive a publish/consume cycle exactly --

    this is what every broker hop (src/streaming) actually relies on.
    """
    track = Track(
        track_id="cam-01-1",
        camera_id="cam-01",
        frame_id="f1",
        box=BoundingBox(x1=1, y1=2, x2=3, y2=4),
        class_id=0,
        class_name="person",
        confidence=0.91,
        state=TrackState.CONFIRMED,
        hits=5,
        age_frames=10,
    )
    event = TrackEvent(track=track, global_identity_id="gid-1")

    payload = event.model_dump_json()
    restored = TrackEvent.model_validate_json(payload)

    assert restored.track.track_id == track.track_id
    assert restored.track.state == TrackState.CONFIRMED
    assert restored.global_identity_id == "gid-1"
    assert restored.event_id == event.event_id
