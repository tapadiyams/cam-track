# ADR 0003: Redis Streams as the default broker, Kafka as a supported alternate

## Status
Accepted

## Context
Ingestion, inference, and storage run as separate, independently scalable
services (see README.md's architecture section) and need a durable,
ordered, at-least-once queue between them: frame references from ingestion
to inference, and track events from inference to storage.

## Decision
Default to Redis Streams (`src/streaming/redis_streams.py`) for
`STREAM_BACKEND`, with a Kafka implementation
(`src/streaming/kafka_backend.py`) behind the same `MessageBroker`
interface as a supported alternative, selected by one setting.

## What they are
Redis Streams is an append-only log data structure inside Redis, with
consumer groups: each message in a stream is delivered to exactly one
consumer within a group, and unacknowledged messages remain in that
consumer's pending-entries list for crash recovery. Kafka is a distributed
commit log purpose-built for high-throughput, multi-datacenter,
long-retention streaming, with topics partitioned across brokers and
consumer groups that own partitions.

## Why Redis Streams as the default
- **Operational simplicity**: most deployments of this pipeline already
  need Redis for nothing else in this design, but standing up a *second*
  piece of infrastructure (a Kafka cluster, with ZooKeeper/KRaft,
  partition planning, and its own monitoring) is a real cost that is not
  justified at the throughput this pipeline runs at (frame references and
  track events -- small JSON payloads -- not raw video).
- **Sub-millisecond publish latency**: Redis Streams' in-memory `XADD` is
  fast enough that the broker is never the pipeline's latency bottleneck;
  the detector and network I/O dominate end-to-end latency long before the
  broker does.
- **Consumer groups are sufficient**: `XREADGROUP`/`XACK` give the same
  at-least-once, one-consumer-per-message guarantee this pipeline needs
  (each raw frame processed by exactly one inference worker, each track
  event written by exactly one storage writer), without Kafka's additional
  complexity of partition count planning ahead of time.

## Why Kafka remains a supported (not default) alternative
Kafka is the better choice once a deployment's requirements shift toward
what it is actually built for:
- **Retention and replay**: Kafka retains messages for a configured
  duration/size regardless of consumption, so a new consumer (e.g. adding
  a fraud-detection service later) can replay history. Redis Streams can
  retain messages too, but it is not Redis's primary design point and
  competes with Redis's in-memory footprint for caching use cases this
  pipeline might also want Redis for.
- **Cross-datacenter replication**: MirrorMaker-style replication for a
  multi-region deployment (e.g. edge sites in different countries feeding
  one central analytics cluster) is a mature, well-supported Kafka
  pattern; Redis Streams has no equivalent built in.
- **Throughput ceiling**: a single Redis instance's Streams are bounded by
  that instance's memory and single-threaded command processing; Kafka's
  partitioned architecture scales horizontally past what one Redis node
  can sustain.

## Consequences
- `MessageBroker` (`src/streaming/base.py`) is the abstraction every
  service codes against; adding a third backend later means implementing
  that interface, not touching ingestion/inference/storage call sites.
- Switching backends mid-deployment (Redis to Kafka) is not a live
  migration in this scaffold -- it requires draining in-flight messages on
  the old backend first, since the two are not bridged.
