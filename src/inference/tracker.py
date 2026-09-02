# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""ByteTrack-lite: multi-object tracking within a single camera's frames.

This is the hard, differentiating part of the pipeline (see README.md) --
turning a per-frame list of detections into temporally consistent identities.
Implements the core idea from Zhang et al., "ByteTrack: Multi-Object
Tracking by Associating Every Detection Box" (2022): most trackers throw
away low-confidence detections before association and lose objects during
occlusion/motion blur exactly when confidence dips. ByteTrack instead
associates in two passes -- high-confidence detections first, then a second
pass matches leftover low-confidence detections against tracks the first
pass didn't cover -- which recovers many of those objects instead of
starting a new identity for them a few frames later. See
docs/decisions/0002-tracker-choice.md for why this over DeepSORT.

One `ByteTracker` instance tracks one camera's frames; cross-camera identity
is a separate, later stage (src/reid/cross_camera_matcher.py) that operates
on the `Track` objects this module emits, not on raw detections.

`ByteTracker.update()`'s three-stage association per frame is branchy
enough to warrant its own diagram: see
docs/diagrams/bytetrack-association.md.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.schemas import BoundingBox, Detection, Track, TrackState
from src.config.constants import InferenceConstants
from src.inference.kalman import BoxKalmanFilter


def iou_matrix(boxes_a: list[tuple[float, float, float, float]],
                boxes_b: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Pairwise IoU between every box in `boxes_a` and every box in `boxes_b`.

    Time: O(n * m) for n = len(boxes_a), m = len(boxes_b) -- one IoU
    computation per pair, no way around that for exact pairwise IoU.
    Space: O(n * m) for the output matrix.
    """
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)))

    a = np.array(boxes_a, dtype=np.float64)  # (n, 4)
    b = np.array(boxes_b, dtype=np.float64)  # (m, 4)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def match_by_iou(
    track_boxes: list[tuple[float, float, float, float]],
    detection_boxes: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve minimum-cost bipartite matching on IoU distance (1 - IoU).

    Returns `(matches, unmatched_track_indices, unmatched_detection_indices)`
    where `matches` is a list of `(track_index, detection_index)` pairs with
    IoU >= `iou_threshold`.

    Time: O(min(n, m)^2 * max(n, m)) for the Hungarian algorithm (n tracks,
    m detections) via `scipy.optimize.linear_sum_assignment`, dominating the
    O(n * m) cost-matrix build. Space: O(n * m) for the cost matrix.
    """
    if not track_boxes or not detection_boxes:
        return [], list(range(len(track_boxes))), list(range(len(detection_boxes)))

    ious = iou_matrix(track_boxes, detection_boxes)
    row_indices, col_indices = linear_sum_assignment(-ious)  # maximize IoU

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    for row, col in zip(row_indices, col_indices, strict=True):
        if ious[row, col] >= iou_threshold:
            matches.append((int(row), int(col)))
            matched_tracks.add(int(row))
            matched_detections.add(int(col))

    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in matched_tracks]
    unmatched_detections = [
        i for i in range(len(detection_boxes)) if i not in matched_detections
    ]
    return matches, unmatched_tracks, unmatched_detections


class STrack:
    """Internal per-track bookkeeping: identity, Kalman filter, lifecycle."""

    def __init__(self, track_id: str, camera_id: str, detection: Detection) -> None:
        self.track_id = track_id
        self.camera_id = camera_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age_frames = 1
        self.time_since_update = 0
        self.last_confidence = detection.confidence
        self._kalman = BoxKalmanFilter(detection.box.as_xyxy())

    def predict(self) -> tuple[float, float, float, float]:
        return self._kalman.predict()

    def mark_matched(self, detection: Detection) -> None:
        self._kalman.update(detection.box.as_xyxy())
        self.hits += 1
        self.time_since_update = 0
        self.last_confidence = detection.confidence
        if self.state == TrackState.LOST:
            # A LOST track was CONFIRMED before it went unmatched, so its
            # hit count already cleared the confirmation bar; matching it
            # again simply resumes CONFIRMED rather than re-earning it.
            self.state = TrackState.CONFIRMED
        elif (
            self.state == TrackState.TENTATIVE
            and self.hits >= InferenceConstants.TRACK_MIN_HITS_TO_CONFIRM
        ):
            self.state = TrackState.CONFIRMED

    def mark_unmatched(self) -> None:
        self.time_since_update += 1
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST
        elif self.state == TrackState.TENTATIVE and self.time_since_update > 0:
            self.state = TrackState.REMOVED  # a tentative track gets one chance, no more

    def current_box(self) -> tuple[float, float, float, float]:
        return self._kalman.current_box()

    def to_track(self, frame_id: str) -> Track:
        x1, y1, x2, y2 = self.current_box()
        return Track(
            track_id=self.track_id,
            camera_id=self.camera_id,
            frame_id=frame_id,
            box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.last_confidence,
            state=self.state,
            hits=self.hits,
            age_frames=self.age_frames,
        )


class ByteTracker:
    """Tracks one camera's objects across frames. One instance per camera."""

    def __init__(self, camera_id: str, next_track_id: "_TrackIdAllocator") -> None:
        self._camera_id = camera_id
        self._tracks: dict[str, STrack] = {}
        self._id_allocator = next_track_id

    def update(self, detections: list[Detection], frame_id: str) -> list[Track]:
        """Advance the tracker one frame and return every non-removed track.

        Time: O(t) for Kalman predicts (t = active tracks) plus two
        Hungarian solves bounded by O(t * d) each (d = detections this
        frame) -- dominated by the association step, not the predicts.
        Space: O(t + d).
        """
        for track in self._tracks.values():
            track.predict()
            track.age_frames += 1

        high_conf_threshold = InferenceConstants.TRACK_HIGH_CONF_THRESHOLD
        high_conf = [d for d in detections if d.confidence >= high_conf_threshold]
        low_conf = [
            d
            for d in detections
            if InferenceConstants.TRACK_LOW_CONF_THRESHOLD <= d.confidence
            < InferenceConstants.TRACK_HIGH_CONF_THRESHOLD
        ]

        active_ids = list(self._tracks.keys())
        active_boxes = [self._tracks[tid].current_box() for tid in active_ids]

        # Stage 1: high-confidence detections vs. every active track.
        matches, unmatched_track_pos, unmatched_high_pos = match_by_iou(
            active_boxes,
            [d.box.as_xyxy() for d in high_conf],
            InferenceConstants.TRACK_MATCH_IOU_THRESHOLD,
        )
        matched_track_ids: set[str] = set()
        for track_pos, det_pos in matches:
            track_id = active_ids[track_pos]
            self._tracks[track_id].mark_matched(high_conf[det_pos])
            matched_track_ids.add(track_id)

        # Stage 2: low-confidence detections vs. tracks stage 1 left unmatched.
        remaining_track_ids = [active_ids[i] for i in unmatched_track_pos]
        remaining_track_boxes = [self._tracks[tid].current_box() for tid in remaining_track_ids]
        stage2_matches, stage2_unmatched_track_pos, _unused_low_pos = match_by_iou(
            remaining_track_boxes,
            [d.box.as_xyxy() for d in low_conf],
            InferenceConstants.TRACK_LOW_CONF_MATCH_IOU_THRESHOLD,
        )
        for track_pos, det_pos in stage2_matches:
            track_id = remaining_track_ids[track_pos]
            self._tracks[track_id].mark_matched(low_conf[det_pos])
            matched_track_ids.add(track_id)

        # Stage 3: still-unmatched high-confidence detections vs. tentative
        # tracks only, at a stricter threshold, to avoid spawning a
        # duplicate identity for an object a brand-new track already covers.
        still_unmatched_track_ids = [
            remaining_track_ids[i] for i in stage2_unmatched_track_pos
        ]
        tentative_ids = [
            tid
            for tid in still_unmatched_track_ids
            if self._tracks[tid].state == TrackState.TENTATIVE
        ]
        tentative_boxes = [self._tracks[tid].current_box() for tid in tentative_ids]
        leftover_high = [high_conf[i] for i in unmatched_high_pos]
        stage3_matches, _unused_tentative_pos, stage3_unmatched_high_pos = match_by_iou(
            tentative_boxes,
            [d.box.as_xyxy() for d in leftover_high],
            InferenceConstants.TRACK_UNCONFIRMED_MATCH_IOU_THRESHOLD,
        )
        for track_pos, det_pos in stage3_matches:
            track_id = tentative_ids[track_pos]
            self._tracks[track_id].mark_matched(leftover_high[det_pos])
            matched_track_ids.add(track_id)

        # Every track not matched in any of the three stages is unmatched
        # this frame: age it (may transition CONFIRMED -> LOST or
        # TENTATIVE -> REMOVED).
        for track_id in active_ids:
            if track_id not in matched_track_ids:
                self._tracks[track_id].mark_unmatched()

        # Detections stage 3 still couldn't place become brand-new tracks.
        for det_pos in stage3_unmatched_high_pos:
            detection = leftover_high[det_pos]
            new_id = self._id_allocator.next_id()
            self._tracks[new_id] = STrack(new_id, self._camera_id, detection)

        # Drop tracks that have aged out (see TRACK_MAX_LOST_FRAMES) or hit
        # a terminal REMOVED state from `mark_unmatched()`.
        for track_id in list(self._tracks.keys()):
            track = self._tracks[track_id]
            if (
                track.state == TrackState.REMOVED
                or track.time_since_update > InferenceConstants.TRACK_MAX_LOST_FRAMES
            ):
                del self._tracks[track_id]

        return [track.to_track(frame_id) for track in self._tracks.values()]


class _TrackIdAllocator:
    """Generates globally-unique, human-scannable track ids.

    A plain incrementing counter (rather than a uuid) keeps track ids short
    and stable in logs/dashboards for the lifetime of one tracker process;
    uniqueness across a restart is not required since `Track.track_id` is
    only ever compared within one process's in-memory tracker state.
    """

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return f"{self._camera_id}-{self._counter}"


def new_tracker(camera_id: str) -> ByteTracker:
    """Construct a `ByteTracker` with its own id allocator for `camera_id`."""
    return ByteTracker(camera_id, _TrackIdAllocator(camera_id))
