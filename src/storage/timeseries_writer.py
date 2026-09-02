# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Batch-writes `TrackEvent`s into the `track_events` hypertable.

Batches, not one INSERT per event: at expected throughput (tens of cameras
x tens of tracks x several fps) a row-at-a-time write would make the DB
round trip the bottleneck of the whole pipeline long before detection or
tracking is. `execute_values` sends one multi-row INSERT per batch instead.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from src.common.schemas import TrackEvent
from src.config.constants import TimeoutConstants


class _ConnectionLike(Protocol):
    """The subset of a DB-API connection this writer needs.

    Typed as a `Protocol` (not `psycopg2.extensions.connection` directly)
    so tests can inject a lightweight fake connection/cursor pair without
    a real Postgres instance -- see tests/unit/storage/test_timeseries_writer.py.
    """

    def cursor(self) -> Any: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


_INSERT_SQL = """
    INSERT INTO track_events (
        event_id, event_time, camera_id, zone, track_id, global_identity_id,
        class_id, class_name, confidence, track_state,
        box_x1, box_y1, box_x2, box_y2
    )
    VALUES %s
    ON CONFLICT (event_id) DO NOTHING
"""


def _event_to_row(event: TrackEvent) -> tuple:
    track = event.track
    event_time = datetime.fromtimestamp(event.event_time_ms / 1000.0, tz=UTC)
    return (
        event.event_id,
        event_time,
        track.camera_id,
        _zone_for(track.camera_id, event),
        track.track_id,
        event.global_identity_id,
        track.class_id,
        track.class_name,
        track.confidence,
        track.state.value,
        track.box.x1,
        track.box.y1,
        track.box.x2,
        track.box.y2,
    )


def _zone_for(camera_id: str, event: TrackEvent) -> str:
    # `Track` does not carry `zone` (it is scoped by camera, and zone is a
    # grouping of cameras -- see configs/cameras.yaml); callers that need
    # zone-accurate rows should attach it before constructing `TrackEvent`
    # in a future revision. Falling back to `camera_id` keeps the column
    # non-null and the row usable for per-camera queries in the meantime.
    return camera_id


class TimeSeriesWriter:
    """Writes batches of `TrackEvent` to the `track_events` hypertable."""

    def __init__(self, connection: _ConnectionLike) -> None:
        self._connection = connection

    def write_batch(self, events: list[TrackEvent]) -> None:
        """Insert `events`, ignoring any whose `event_id` already exists.

        Time: O(n) to build rows plus one round trip for the batched
        INSERT (n = len(events)), instead of O(n) round trips.
        Space: O(n) for the row tuples held before the single execute.

        Idempotent by design (`ON CONFLICT ... DO NOTHING` on the primary
        key `event_id`): safe to call twice with overlapping events, which
        is exactly what happens when a broker redelivers a message that
        was written but not yet acked before a worker crash.
        """
        if not events:
            return

        # Imported here, not at module level, so `_event_to_row` and the
        # rest of this module stay importable/testable without a real
        # `psycopg2` install (tests inject a fake `_ConnectionLike`/cursor
        # pair instead -- see tests/unit/storage/test_timeseries_writer.py).
        from psycopg2.extras import execute_values

        rows = [_event_to_row(event) for event in events]
        cursor = self._connection.cursor()
        try:
            execute_values(cursor, _INSERT_SQL, rows)
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        finally:
            cursor.close()


def connect(dsn: str) -> _ConnectionLike:
    """Open a new DB connection with the standard query/connect timeouts."""
    import psycopg2

    return psycopg2.connect(
        dsn,
        connect_timeout=TimeoutConstants.DB_CONNECT_TIMEOUT_SECONDS,
        options=f"-c statement_timeout={TimeoutConstants.DB_QUERY_TIMEOUT_SECONDS * 1000}",
    )
