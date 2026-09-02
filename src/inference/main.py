# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Inference service entrypoint. One process = one horizontally-scalable

inference worker; run N of these (see docker-compose.yml `--scale`) to
scale detection+tracking throughput.
"""

from __future__ import annotations

import os
import socket

from src.common.logging_utils import configure_logging
from src.config.settings import get_settings
from src.inference.detector import OnnxYoloDetector, UltralyticsYoloDetector
from src.inference.frame_loader import LocalDiskFrameLoader
from src.inference.worker import InferenceWorker
from src.streaming.factory import get_broker

# COCO class names -- the default YOLOv8 pretrained checkpoint's label set.
# A fine-tuned model ships its own names via the .onnx metadata in a fuller
# implementation; hardcoded here to keep the demo path dependency-free.
_COCO_CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _build_detector(settings) -> object:
    if settings.detector_use_onnx_runtime:
        return OnnxYoloDetector(settings.detector_weights_path, class_names=_COCO_CLASS_NAMES)
    return UltralyticsYoloDetector(
        settings.detector_weights_path, device=settings.detector_device
    )


def main() -> None:
    settings = get_settings()
    logger = configure_logging("inference", settings.log_level)

    broker = get_broker(settings)
    detector = _build_detector(settings)
    frame_loader = LocalDiskFrameLoader()
    consumer_name = f"{socket.gethostname()}-{os.getpid()}"

    logger.info("starting inference worker", extra={"consumer": consumer_name})

    worker = InferenceWorker(
        broker=broker,
        detector=detector,
        frame_loader=frame_loader,
        consumer_name=consumer_name,
        max_batch_size=settings.max_batch_size,
        max_batch_wait_ms=settings.max_batch_wait_ms,
    )
    worker.run_forever()


if __name__ == "__main__":
    main()
