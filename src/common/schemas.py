# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Wire schemas shared by every stage of the pipeline.

Ingestion -> inference -> storage -> dashboard all pass messages through
Redis Streams / Kafka as JSON, so these pydantic models are the single
source of truth for that wire format. Changing a field here changes what
every consumer downstream expects -- see the compatibility policy in
README.md before removing or renaming a field.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field


def _new_id() -> str:
    """Generate a URL-safe unique id for a message or track.

    Time: O(1). Space: O(1) -- fixed-length uuid4 hex string.
    """
    return uuid.uuid4().hex


def _now_ms() -> int:
    """Current wall-clock time in integer milliseconds since epoch."""
    return int(time.time() * 1000)


class BoundingBox(BaseModel):
    """Axis-aligned box in pixel coordinates, top-left origin."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.width / 2.0, self.y1 + self.height / 2.0)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class RawFrame(BaseModel):
    """One captured frame, published by ingestion, consumed by inference.

    The frame payload itself is not embedded in the message -- only a
    reference (`frame_uri`) to where inference can fetch it (a shared
    volume path or object-store key). Streaming raw JPEG/PNG bytes through
    Redis Streams / Kafka bloats every downstream consumer's read path and
    caps broker throughput far below what frame references cost; see
    docs/decisions/0003-message-queue-choice.md.
    """

    frame_id: str = Field(default_factory=_new_id)
    camera_id: str
    zone: str
    captured_at_ms: int = Field(default_factory=_now_ms)
    frame_uri: str
    width_px: int
    height_px: int
    sequence_number: int


class Detection(BaseModel):
    """One detected object in one frame, before tracking is applied."""

    frame_id: str
    camera_id: str
    box: BoundingBox
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class TrackState(str, Enum):
    """Lifecycle states for a per-camera track (ByteTrack-style)."""

    TENTATIVE = "tentative"  # seen, not yet confirmed (< min_hits)
    CONFIRMED = "confirmed"  # confirmed, matched in the current frame
    LOST = "lost"  # confirmed but unmatched for 1..max_lost_frames
    REMOVED = "removed"  # unmatched for > max_lost_frames; terminal


class Track(BaseModel):
    """A per-camera object track: a detection with temporal identity."""

    track_id: str
    camera_id: str
    frame_id: str
    box: BoundingBox
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    state: TrackState
    hits: int = Field(ge=0)
    age_frames: int = Field(ge=0)
    updated_at_ms: int = Field(default_factory=_now_ms)


class TrackEvent(BaseModel):
    """A `Track` update, published to the `TRACK_EVENTS` stream and

    persisted to the time-series store. One event per track per processed
    frame keeps the storage schema append-only and the dashboard's queries
    simple range scans over `event_time`.
    """

    event_id: str = Field(default_factory=_new_id)
    track: Track
    global_identity_id: str | None = None  # set once re-ID resolves it
    event_time_ms: int = Field(default_factory=_now_ms)


class ReidMatch(BaseModel):
    """A cross-camera identity match between two tracks in the same zone."""

    match_id: str = Field(default_factory=_new_id)
    zone: str
    source_track_id: str
    source_camera_id: str
    matched_track_id: str
    matched_camera_id: str
    global_identity_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    matched_at_ms: int = Field(default_factory=_now_ms)
