# ADR 0004: TimescaleDB over a dedicated time-series DB or plain Postgres

## Status
Accepted

## Context
Every processed frame produces track events (position, class, confidence,
identity) that the dashboard needs to query as time-bounded aggregates
("foot traffic per minute for the last hour," "distinct visitors today").
The write pattern is a high-volume append-only stream; the read pattern is
almost always a bounded time-range scan with grouping.

## Decision
Use TimescaleDB (a PostgreSQL extension) as the analytics store
(`src/storage/schema.sql`), with a continuous aggregate
(`track_counts_per_minute`) backing the dashboard's chart queries.

## What it is
TimescaleDB adds automatic time-based partitioning ("hypertables"),
continuous aggregates (materialized views that incrementally update as new
data arrives, rather than recomputing from scratch), and retention
policies on top of standard PostgreSQL -- while remaining fully
SQL-queryable and ACID-transactional like any Postgres database.

## Why TimescaleDB
- **Query pattern fit**: "last N hours/days" queries only touch the
  relevant time chunks (`create_hypertable`'s automatic partitioning)
  instead of scanning the whole table, which is exactly the dashboard's
  access pattern.
- **Continuous aggregates remove a whole class of bugs**: without them,
  the dashboard's "counts per minute" chart would either re-scan and
  re-group raw rows on every request (increasingly slow as data
  accumulates) or require a hand-rolled batch job to precompute rollups
  (more infrastructure, more failure modes). TimescaleDB's continuous
  aggregate keeps this incremental and correct automatically.
- **It is still Postgres**: standard SQL, standard tooling (`psycopg2`,
  ORMs, `pg_dump`, connection poolers), and joins against any other
  relational data a deployment might already have (e.g. a `cameras` or
  `zones` reference table) work exactly as they would with vanilla
  Postgres -- there is no separate query language to learn.

## Why not a dedicated time-series database (e.g. InfluxDB)
Purpose-built time-series databases often out-perform TimescaleDB on raw
ingest throughput for pure time-series workloads. That was not the
deciding factor here:
- This data is not *pure* time-series (a single numeric metric per
  timestamp) -- each row carries relational structure (camera, zone,
  track, class, box coordinates, an optional cross-camera identity) that
  benefits from real joins and `WHERE`/`GROUP BY` flexibility, which
  InfluxDB's query model (Flux, or InfluxQL) supports less naturally than
  standard SQL.
- Running a second, unfamiliar database technology only for this one data
  type is an operational cost most teams adopting this pipeline do not
  need to pay when TimescaleDB gets them the partitioning and rollup
  benefits they actually need without leaving the Postgres ecosystem.

## Why not plain PostgreSQL (no TimescaleDB extension)
Plain Postgres can absolutely store this data, but without hypertables a
single `track_events` table's indexes degrade as it grows into the tens or
hundreds of millions of rows (the expected scale after weeks of
multi-camera operation at several events per camera per second), and
without continuous aggregates, every dashboard chart query re-aggregates
raw rows from scratch, growing slower over the deployment's entire
lifetime. TimescaleDB is a drop-in extension (not a fork or a different
wire protocol), so there is no real cost to adopting it upfront versus
migrating to it later under production load pressure.

## Consequences
- Deployments need a TimescaleDB-compatible Postgres image (the official
  `timescale/timescaledb` image, or a managed service that supports the
  extension) rather than an arbitrary Postgres host.
- The continuous aggregate's `schedule_interval` (1 minute, see
  `schema.sql`) trades a small amount of dashboard staleness for
  dramatically cheaper chart queries; a use case needing sub-minute
  freshness would need a shorter interval or a different aggregation
  strategy.
