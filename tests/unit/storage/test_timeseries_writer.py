# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/storage/timeseries_writer.py.

`_event_to_row` needs nothing but the pydantic schemas and always runs.
`TimeSeriesWriter.write_batch` needs `psycopg2` importable (for
`execute_values`, imported lazily inside the method -- see the module
docstring there) and is skipped otherwise.
"""

from datetime import UTC, datetime

import pytest

from src.common.schemas import BoundingBox, Track, TrackEvent, TrackState
from src.storage.timeseries_writer import TimeSeriesWriter, _event_to_row


def _make_event(event_id="e1", global_identity_id=None):
    track = Track(
        track_id="cam-01-1",
        camera_id="cam-01",
        frame_id="f1",
        box=BoundingBox(x1=1, y1=2, x2=3, y2=4),
        class_id=0,
        class_name="person",
        confidence=0.87,
        state=TrackState.CONFIRMED,
        hits=5,
        age_frames=10,
    )
    return TrackEvent(
        event_id=event_id, track=track, global_identity_id=global_identity_id
    )


def test_event_to_row_maps_every_schema_field_into_the_row_tuple():
    event = _make_event(global_identity_id="gid-1")
    row = _event_to_row(event)

    assert row[0] == "e1"
    assert isinstance(row[1], datetime)
    assert row[1].tzinfo == UTC
    assert row[2] == "cam-01"  # camera_id
    assert row[4] == "cam-01-1"  # track_id
    assert row[5] == "gid-1"  # global_identity_id
    assert row[9] == "confirmed"  # track_state (enum .value)
    assert row[10:14] == (1.0, 2.0, 3.0, 4.0)


def test_event_to_row_allows_null_global_identity_before_reid_resolves_it():
    row = _event_to_row(_make_event(global_identity_id=None))
    assert row[5] is None


class _FakeCursor:
    def __init__(self):
        self.executed = []

    def close(self):
        pass


class _FakeConnection:
    def __init__(self):
        self.committed = False
        self.rolled_back = False
        self._cursor = _FakeCursor()

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_write_batch_commits_on_success(monkeypatch):
    pytest.importorskip("psycopg2")
    calls = []
    monkeypatch.setattr(
        "psycopg2.extras.execute_values",
        lambda cursor, sql, rows: calls.append((sql, rows)),
    )

    connection = _FakeConnection()
    writer = TimeSeriesWriter(connection)
    writer.write_batch([_make_event()])

    assert len(calls) == 1
    assert connection.committed is True
    assert connection.rolled_back is False


def test_write_batch_rolls_back_and_reraises_on_failure(monkeypatch):
    pytest.importorskip("psycopg2")

    def _boom(cursor, sql, rows):
        raise RuntimeError("db exploded")

    monkeypatch.setattr("psycopg2.extras.execute_values", _boom)

    connection = _FakeConnection()
    writer = TimeSeriesWriter(connection)

    with pytest.raises(RuntimeError, match="db exploded"):
        writer.write_batch([_make_event()])

    assert connection.rolled_back is True
    assert connection.committed is False


def test_write_batch_is_a_no_op_for_an_empty_list():
    connection = _FakeConnection()
    writer = TimeSeriesWriter(connection)
    writer.write_batch([])  # must not touch the connection at all
