# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Reads a frame image back given the `frame_uri` ingestion published.

Deliberately independent of `src.ingestion.frame_store.FrameStore` (even
though the local-disk implementation is nearly identical): inference and
ingestion are separate deployable services and neither should import the
other's internals, only agree on the `frame_uri` string contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class FrameLoader(ABC):
    """Resolves a `frame_uri` (as published in a `RawFrame`) to pixels."""

    @abstractmethod
    def load(self, frame_uri: str) -> np.ndarray: ...


class LocalDiskFrameLoader(FrameLoader):
    """Reads `file://` URIs written by `LocalDiskFrameStore`."""

    def load(self, frame_uri: str) -> np.ndarray:
        path = frame_uri.removeprefix("file://")
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"No frame image found at {frame_uri}")
        return image
