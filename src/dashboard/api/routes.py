# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Read-only analytics endpoints backed by TimescaleDB.

Every query here reads `track_counts_per_minute` (a continuous aggregate,
see src/storage/schema.sql) instead of the raw `track_events` hypertable --
the whole point of the aggregate is that these endpoints answer in
milliseconds regardless of how many raw rows have accumulated.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.config.constants import HttpStatusMessages
from src.dashboard.db import get_connection

router = APIRouter(prefix="/api")

_MAX_LOOKBACK_MINUTES = 7 * 24 * 60  # one week; the aggregate itself retains 30 days


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/cameras")
def list_cameras() -> list[dict[str, str]]:
    """Every camera that has produced at least one track event, with its zone."""
    sql = "SELECT DISTINCT camera_id, zone FROM track_events ORDER BY camera_id"
    try:
        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            cursor.close()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=HttpStatusMessages.UPSTREAM_UNAVAILABLE
        ) from exc
    return [{"camera_id": row[0], "zone": row[1]} for row in rows]


@router.get("/traffic")
def traffic(
    camera_id: str | None = Query(default=None),
    zone: str | None = Query(default=None),
    minutes: int = Query(default=60, gt=0, le=_MAX_LOOKBACK_MINUTES),
) -> list[dict[str, object]]:
    """Per-minute distinct-track counts for the last `minutes` minutes.

    Filters by `camera_id` XOR `zone` when given; with neither, returns
    every camera's series (the dashboard aggregates client-side for the
    "all cameras" view).
    """
    if camera_id and zone:
        raise HTTPException(
            status_code=400, detail=HttpStatusMessages.BAD_REQUEST
        )

    where_clause = "WHERE bucket >= now() - (%(minutes)s || ' minutes')::interval"
    params: dict[str, object] = {"minutes": minutes}
    if camera_id:
        where_clause += " AND camera_id = %(camera_id)s"
        params["camera_id"] = camera_id
    if zone:
        where_clause += " AND zone = %(zone)s"
        params["zone"] = zone

    sql = f"""
        SELECT bucket, camera_id, zone, distinct_track_count
        FROM track_counts_per_minute
        {where_clause}
        ORDER BY bucket ASC
    """

    try:
        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            cursor.close()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=HttpStatusMessages.UPSTREAM_UNAVAILABLE
        ) from exc

    return [
        {
            "bucket": row[0].isoformat(),
            "camera_id": row[1],
            "zone": row[2],
            "distinct_track_count": row[3],
        }
        for row in rows
    ]
