-- Authored by: Shubham Tapadiya
-- Created: 2026-09-02
-- Updated: 2026-09-02
--
-- track_events is the single append-only fact table the dashboard queries.
-- It is a TimescaleDB hypertable partitioned on event_time: analytics
-- queries are almost always "in the last N hours/days," which a hypertable
-- answers by touching only the relevant time chunks instead of scanning
-- the whole table -- see docs/decisions/0004-storage-choice.md.

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS track_events (
    event_id            TEXT PRIMARY KEY,
    event_time          TIMESTAMPTZ NOT NULL,
    camera_id           TEXT NOT NULL,
    zone                TEXT NOT NULL,
    track_id            TEXT NOT NULL,
    global_identity_id  TEXT,
    class_id            INTEGER NOT NULL,
    class_name          TEXT NOT NULL,
    confidence          REAL NOT NULL,
    track_state         TEXT NOT NULL,
    box_x1              REAL NOT NULL,
    box_y1              REAL NOT NULL,
    box_x2              REAL NOT NULL,
    box_y2              REAL NOT NULL
);

-- `create_hypertable` is idempotent via `if_not_exists`; safe to run this
-- script every deploy (see scripts/setup.sh).
SELECT create_hypertable('track_events', 'event_time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_track_events_camera_time
    ON track_events (camera_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_track_events_identity
    ON track_events (global_identity_id)
    WHERE global_identity_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_track_events_zone_time
    ON track_events (zone, event_time DESC);

-- Retention: raw per-frame track events older than 30 days are dropped: the
-- dashboard's historical charts read from continuous aggregates (below),
-- not raw rows, so this does not lose reporting fidelity.
SELECT add_retention_policy('track_events', INTERVAL '30 days', if_not_exists => TRUE);

-- Continuous aggregate: per-camera, per-minute unique-track counts. The
-- dashboard's "foot traffic over time" chart reads this instead of
-- re-scanning and re-counting raw rows on every request.
CREATE MATERIALIZED VIEW IF NOT EXISTS track_counts_per_minute
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', event_time) AS bucket,
    camera_id,
    zone,
    COUNT(DISTINCT track_id) AS distinct_track_count
FROM track_events
GROUP BY bucket, camera_id, zone
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'track_counts_per_minute',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);
