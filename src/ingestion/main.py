# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Ingestion service entrypoint: one thread per configured camera.

Threads, not processes or asyncio, because each camera's loop is I/O-bound
(blocking on `cv2.VideoCapture.read()`, which releases the GIL) and we want
one crashed/reconnecting camera to never block the others -- a thread pool
gives that isolation with far less operational overhead than one process
per camera. A production deployment instead runs one ingestion *container*
per camera (see docker-compose.yml) and this thread pool is what that
container falls back to for local multi-camera dev.
"""

from __future__ import annotations

import threading

from src.common.logging_utils import configure_logging
from src.config.settings import get_settings
from src.ingestion.camera_config import load_camera_configs
from src.ingestion.frame_publisher import FramePublisher
from src.ingestion.frame_store import LocalDiskFrameStore
from src.ingestion.rtsp_reader import RtspFrameReader
from src.streaming.factory import get_broker


def main() -> None:
    settings = get_settings()
    logger = configure_logging("ingestion", settings.log_level)

    cameras = load_camera_configs(settings.camera_config_path)
    broker = get_broker(settings)
    frame_store = LocalDiskFrameStore()

    logger.info("starting ingestion", extra={"camera_count": len(cameras)})

    threads: list[threading.Thread] = []
    for camera in cameras:
        reader = RtspFrameReader(source=camera.source, fps_cap=camera.fps_cap)
        publisher = FramePublisher(camera, reader, frame_store, broker)
        thread = threading.Thread(
            target=publisher.run_forever,
            name=f"ingest-{camera.camera_id}",
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
