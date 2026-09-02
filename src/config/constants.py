# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Central constants module. Import from here -- never inline a literal that

is shared across two or more files. Deployment-specific values (URLs,
credentials, ports) still go through env vars / settings.py, not here.
"""

from __future__ import annotations


class StreamNames:
    """Redis Stream / Kafka topic names used across services."""

    RAW_FRAMES = "camtrack.frames.raw.v1"
    DETECTIONS = "camtrack.detections.v1"
    TRACK_EVENTS = "camtrack.track_events.v1"
    TRACK_EVENTS_DLQ = "camtrack.track_events.v1.dlq"
    REID_MATCHES = "camtrack.reid_matches.v1"


class ConsumerGroups:
    """Consumer group names for Redis Streams / Kafka."""

    INFERENCE_WORKERS = "inference-workers"
    STORAGE_WRITERS = "storage-writers"
    REID_WORKERS = "reid-workers"


class TimeoutConstants:
    """Timeouts, in seconds, used for network and queue operations."""

    RTSP_CONNECT_TIMEOUT_SECONDS = 10
    RTSP_READ_TIMEOUT_SECONDS = 5
    RTSP_RECONNECT_BACKOFF_BASE_SECONDS = 2
    RTSP_RECONNECT_BACKOFF_MAX_SECONDS = 30
    STREAM_READ_BLOCK_MILLISECONDS = 2000
    DB_QUERY_TIMEOUT_SECONDS = 5
    DB_CONNECT_TIMEOUT_SECONDS = 10
    HTTP_CLIENT_TIMEOUT_SECONDS = 15


class RetryConstants:
    """Retry / backoff policy shared by network and queue clients."""

    MAX_RETRIES = 5
    BACKOFF_BASE_SECONDS = 1.0
    BACKOFF_MULTIPLIER = 2.0
    BACKOFF_MAX_SECONDS = 60.0


class InferenceConstants:
    """Detection, batching, and tracking thresholds.

    These are algorithm-level defaults, not per-deployment tuning; per-camera
    overrides belong in configs/cameras.yaml and are read via settings.py.
    """

    DEFAULT_MAX_BATCH_SIZE = 8
    DEFAULT_MAX_BATCH_WAIT_MILLISECONDS = 40
    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    DEFAULT_IOU_NMS_THRESHOLD = 0.45
    DEFAULT_INPUT_SIZE_PX = 640

    # ByteTrack two-stage association thresholds (Zhang et al., 2022 defaults).
    TRACK_HIGH_CONF_THRESHOLD = 0.6
    TRACK_LOW_CONF_THRESHOLD = 0.1
    # Stage 1: high-score detections vs. every active track.
    TRACK_MATCH_IOU_THRESHOLD = 0.3
    # Stage 2: low-score detections vs. tracks left unmatched by stage 1 --
    # stricter than stage 1 because low-score boxes are noisier evidence,
    # so we only accept a very confident spatial overlap.
    TRACK_LOW_CONF_MATCH_IOU_THRESHOLD = 0.5
    # Stage 3: brand-new (tentative) tracks vs. high-score detections left
    # unmatched by stages 1-2 -- stricter still, to avoid spawning a
    # duplicate track for an object that a tentative track already covers.
    TRACK_UNCONFIRMED_MATCH_IOU_THRESHOLD = 0.7
    TRACK_MAX_LOST_FRAMES = 30
    TRACK_MIN_HITS_TO_CONFIRM = 3


class ReidConstants:
    """Cross-camera re-identification gallery matching thresholds."""

    EMBEDDING_DIM = 512
    COSINE_SIMILARITY_MATCH_THRESHOLD = 0.72
    GALLERY_MAX_AGE_SECONDS = 300
    GALLERY_MAX_ENTRIES_PER_IDENTITY = 5


class HttpStatusMessages:
    """Canned messages for dashboard API error responses."""

    NOT_FOUND = "The requested resource was not found."
    BAD_REQUEST = "The request was malformed or missing required fields."
    UPSTREAM_UNAVAILABLE = "A dependent service (queue or database) is unavailable."


class LogFields:
    """Structured-logging field names, kept consistent across services."""

    CAMERA_ID = "camera_id"
    FRAME_ID = "frame_id"
    TRACK_ID = "track_id"
    GLOBAL_IDENTITY_ID = "global_identity_id"
    LATENCY_MS = "latency_ms"
    SERVICE = "service"
