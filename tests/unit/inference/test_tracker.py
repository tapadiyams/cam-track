# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/inference/tracker.py -- the ByteTrack-lite implementation.

This is the hard, differentiating part of the pipeline (see README.md), so
it gets the most thorough coverage: identity continuity across frames,
occlusion recovery via the low-confidence second association stage (the
actual point of ByteTrack over a naive single-stage tracker), confirmation
after enough hits, and removal after the lost-frame budget is exhausted.
"""

from src.common.schemas import BoundingBox, Detection, TrackState
from src.config.constants import InferenceConstants
from src.inference.tracker import iou_matrix, match_by_iou, new_tracker


def _det(x1, y1, x2, y2, confidence=0.9, class_id=0, class_name="person"):
    return Detection(
        frame_id="f",
        camera_id="cam-01",
        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
    )


def test_iou_matrix_shape_and_values():
    matrix = iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (100, 100, 110, 110)])
    assert matrix.shape == (1, 2)
    assert matrix[0, 0] == 1.0
    assert matrix[0, 1] == 0.0


def test_iou_matrix_empty_inputs_return_empty_matrix():
    assert iou_matrix([], [(0, 0, 10, 10)]).shape == (0, 1)
    assert iou_matrix([(0, 0, 10, 10)], []).shape == (1, 0)


def test_match_by_iou_respects_threshold():
    matches, unmatched_tracks, unmatched_dets = match_by_iou(
        [(0, 0, 10, 10)], [(50, 50, 60, 60)], iou_threshold=0.3
    )
    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_dets == [0]


def test_tracker_confirms_a_track_after_enough_consecutive_hits():
    tracker = new_tracker("cam-01")
    min_hits = InferenceConstants.TRACK_MIN_HITS_TO_CONFIRM

    tracks = []
    for frame_num in range(min_hits):
        tracks = tracker.update([_det(0, 0, 20, 20)], frame_id=f"f{frame_num}")

    assert len(tracks) == 1
    assert tracks[0].state == TrackState.CONFIRMED
    assert tracks[0].hits == min_hits


def test_tracker_keeps_same_track_id_across_frames_with_smooth_motion():
    tracker = new_tracker("cam-01")
    track_ids_seen = set()

    for step in range(5):
        offset = step * 3
        tracks = tracker.update([_det(offset, 0, offset + 20, 20)], frame_id=f"f{step}")
        track_ids_seen.update(t.track_id for t in tracks)

    assert len(track_ids_seen) == 1  # one continuous identity, not a new one each frame


def test_tracker_recovers_track_through_a_low_confidence_occlusion_frame():
    """The scenario ByteTrack exists for: an object's detection confidence

    dips during partial occlusion instead of disappearing outright. A
    single-stage tracker that discards low-confidence detections before
    association would lose this track and hand it a new id when confidence
    recovers; ByteTrack's second association stage should instead keep the
    same identity through the dip.
    """
    tracker = new_tracker("cam-01")
    min_hits = InferenceConstants.TRACK_MIN_HITS_TO_CONFIRM

    for step in range(min_hits):
        tracker.update([_det(0, 0, 20, 20, confidence=0.9)], frame_id=f"warmup{step}")

    tracks_before = tracker.update([_det(0, 0, 20, 20, confidence=0.9)], frame_id="before")
    track_id = tracks_before[0].track_id

    low_conf_detection = _det(1, 1, 21, 21, confidence=0.2)  # below TRACK_HIGH_CONF_THRESHOLD
    occluded_tracks = tracker.update([low_conf_detection], frame_id="occluded")

    tracks_after = tracker.update([_det(2, 2, 22, 22, confidence=0.9)], frame_id="after")

    assert occluded_tracks[0].track_id == track_id
    assert tracks_after[0].track_id == track_id


def test_tracker_removes_a_track_after_exceeding_max_lost_frames():
    tracker = new_tracker("cam-01")
    min_hits = InferenceConstants.TRACK_MIN_HITS_TO_CONFIRM
    max_lost = InferenceConstants.TRACK_MAX_LOST_FRAMES

    for step in range(min_hits):
        tracker.update([_det(0, 0, 20, 20)], frame_id=f"warmup{step}")

    tracks = []
    for step in range(max_lost + 2):
        tracks = tracker.update([], frame_id=f"empty{step}")

    assert tracks == []


def test_tracker_spawns_independent_tracks_for_spatially_distinct_objects():
    tracker = new_tracker("cam-01")
    tracks = tracker.update(
        [_det(0, 0, 20, 20), _det(500, 500, 520, 520)], frame_id="f0"
    )
    assert len({t.track_id for t in tracks}) == 2


def test_tracker_preserves_the_originating_detections_class_across_matches():
    """A track's class comes from the detection that created it and should

    stay stable across matched frames -- a regression that started pulling
    class from the wrong side of a match would silently mislabel every
    object in the dashboard's per-class counts.
    """
    tracker = new_tracker("cam-01")
    tracks = tracker.update(
        [_det(0, 0, 20, 20, class_id=0, class_name="person")], frame_id="f0"
    )
    track_id = tracks[0].track_id

    tracks = tracker.update(
        [_det(1, 1, 21, 21, class_id=0, class_name="person")], frame_id="f1"
    )
    assert tracks[0].track_id == track_id
    assert tracks[0].class_name == "person"
