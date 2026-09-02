# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Storage service entrypoint: drains `TRACK_EVENTS` into TimescaleDB.

Batches by polling the broker for up to `_BATCH_SIZE` messages at a time
(bounded by the broker's own `count`/timeout behavior) rather than a
`DynamicBatcher` -- storage has no GPU/latency budget to protect, so a
simpler poll-and-flush loop is enough and avoids running a second
background thread for no benefit.
"""

from __future__ import annotations

import os
import socket

from src.common.logging_utils import configure_logging
from src.common.schemas import TrackEvent
from src.config.constants import ConsumerGroups, StreamNames
from src.config.settings import get_settings
from src.storage.timeseries_writer import TimeSeriesWriter, connect
from src.streaming.factory import get_broker

_BATCH_SIZE = 100


def main() -> None:
    settings = get_settings()
    logger = configure_logging("storage", settings.log_level)

    broker = get_broker(settings)
    connection = connect(settings.timescale_dsn)
    writer = TimeSeriesWriter(connection)
    consumer_name = f"{socket.gethostname()}-{os.getpid()}"

    logger.info("starting storage writer", extra={"consumer": consumer_name})

    while True:
        messages = broker.consume(
            StreamNames.TRACK_EVENTS,
            ConsumerGroups.STORAGE_WRITERS,
            consumer_name,
            count=_BATCH_SIZE,
        )
        if not messages:
            continue

        events = [TrackEvent.model_validate(m.payload) for m in messages]
        writer.write_batch(events)

        for message in messages:
            broker.ack(
                StreamNames.TRACK_EVENTS, ConsumerGroups.STORAGE_WRITERS, message.message_id
            )


if __name__ == "__main__":
    main()
