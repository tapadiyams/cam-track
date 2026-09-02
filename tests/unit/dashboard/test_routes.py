# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/dashboard/api/routes.py.

Skipped entirely when `fastapi` is not installed. The DB layer is faked at
the connection level (see `_FakeConnection`) so these tests exercise real
route logic (query building, error mapping, response shape) without a live
TimescaleDB.
"""

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.dashboard.api import routes as routes_module  # noqa: E402
from src.dashboard.api.routes import router  # noqa: E402


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)


def _client_with_rows(monkeypatch, rows):
    import contextlib

    @contextlib.contextmanager
    def fake_get_connection():
        yield _FakeConnection(rows)

    monkeypatch.setattr(routes_module, "get_connection", fake_get_connection)

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_health_endpoint_returns_ok():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_cameras_returns_camera_and_zone_pairs(monkeypatch):
    client = _client_with_rows(monkeypatch, [("cam-01", "store-front"), ("cam-02", "warehouse")])
    response = client.get("/api/cameras")
    assert response.status_code == 200
    assert response.json() == [
        {"camera_id": "cam-01", "zone": "store-front"},
        {"camera_id": "cam-02", "zone": "warehouse"},
    ]


def test_list_cameras_returns_503_when_the_database_is_unreachable(monkeypatch):
    import contextlib

    @contextlib.contextmanager
    def fake_get_connection():
        raise ConnectionError("db down")
        yield  # pragma: no cover - unreachable, satisfies generator shape

    monkeypatch.setattr(routes_module, "get_connection", fake_get_connection)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/api/cameras")
    assert response.status_code == 503


def test_traffic_rejects_both_camera_id_and_zone_together(monkeypatch):
    client = _client_with_rows(monkeypatch, [])
    response = client.get("/api/traffic", params={"camera_id": "cam-01", "zone": "store-front"})
    assert response.status_code == 400


def test_traffic_returns_rows_shaped_for_the_dashboard_chart(monkeypatch):
    import datetime

    bucket = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    client = _client_with_rows(monkeypatch, [(bucket, "cam-01", "store-front", 3)])

    response = client.get("/api/traffic", params={"camera_id": "cam-01"})
    assert response.status_code == 200
    body = response.json()
    assert body == [
        {
            "bucket": bucket.isoformat(),
            "camera_id": "cam-01",
            "zone": "store-front",
            "distinct_track_count": 3,
        }
    ]


def test_traffic_rejects_a_window_beyond_the_max_lookback(monkeypatch):
    client = _client_with_rows(monkeypatch, [])
    response = client.get("/api/traffic", params={"minutes": 999_999})
    assert response.status_code == 422
