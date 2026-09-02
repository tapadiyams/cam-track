# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Where captured frame images actually live.

Only a `frame_uri` travels through the message broker (see
`RawFrame` in src/common/schemas.py); this module writes the bytes
somewhere inference can read them back from.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import cv2
import numpy as np


class FrameStore(ABC):
    """Persists one frame image and returns a URI inference can resolve."""

    @abstractmethod
    def save(self, camera_id: str, frame_id: str, image: np.ndarray) -> str:
        """Persist `image` and return its `frame_uri`."""

    @abstractmethod
    def load(self, frame_uri: str) -> np.ndarray:
        """Resolve `frame_uri` (as returned by `save`) back to an image."""


class LocalDiskFrameStore(FrameStore):
    """Writes frames as JPEG files under a shared local/mounted directory.

    Adequate for a single-host demo or a Docker Compose stack sharing a
    volume between the ingestion and inference containers. A multi-host
    edge/cloud deployment would swap this for an S3/MinIO-backed
    implementation behind the same `FrameStore` interface -- see
    docs/decisions/0005-edge-vs-cloud.md.
    """

    def __init__(self, base_dir: str = "data/frames", jpeg_quality: int = 90) -> None:
        self._base_dir = base_dir
        self._jpeg_quality = jpeg_quality
        os.makedirs(self._base_dir, exist_ok=True)

    def save(self, camera_id: str, frame_id: str, image: np.ndarray) -> str:
        camera_dir = os.path.join(self._base_dir, camera_id)
        os.makedirs(camera_dir, exist_ok=True)
        path = os.path.join(camera_dir, f"{frame_id}.jpg")
        params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        if not cv2.imwrite(path, image, params):
            raise OSError(f"Failed to write frame image to {path}")
        return f"file://{os.path.abspath(path)}"

    def load(self, frame_uri: str) -> np.ndarray:
        path = frame_uri.removeprefix("file://")
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"No frame image found at {frame_uri}")
        return image
