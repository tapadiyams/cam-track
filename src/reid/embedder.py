# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Appearance embedding extraction for cross-camera re-identification.

The tracker (src/inference/tracker.py) gives an identity that is stable
*within* one camera; re-identifying the same object across cameras needs a
signal that survives a change in viewpoint, lighting, and scale, which pixel
position cannot provide. An appearance embedding -- a fixed-length vector
such that the same physical object's crops land close together under cosine
similarity, regardless of camera -- is that signal. See
docs/decisions/0002-tracker-choice.md for why this is a separate stage
rather than baked into the tracker itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.config.constants import ReidConstants


class Embedder(ABC):
    """Maps an image crop to an L2-normalized embedding vector."""

    @abstractmethod
    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Return a unit-norm vector of length `ReidConstants.EMBEDDING_DIM`."""


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector
    return vector / norm


class OnnxReidEmbedder(Embedder):
    """A small CNN (e.g. OSNet, a ResNet18 trained with triplet loss)

    exported to ONNX. This is the production backend -- accurate enough to
    survive real viewpoint/lighting changes, but it requires trained
    weights (`settings.reid_weights_path`) that ship separately from this
    scaffold.
    """

    def __init__(self, onnx_path: str, input_size_px: int = 128) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._input_size_px = input_size_px

    def embed(self, crop: np.ndarray) -> np.ndarray:
        import cv2

        resized = cv2.resize(crop, (self._input_size_px, self._input_size_px))
        blob = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        raw_output = self._session.run(None, {self._input_name: blob})[0]
        return _l2_normalize(raw_output[0])


class ColorHistogramEmbedder(Embedder):
    """Dependency-free HSV color-histogram embedding. Dev/demo fallback only.

    Trained re-ID weights are not part of this scaffold, so the demo and
    unit tests need a signal that needs no model file and no GPU. A
    normalized HSV histogram is a weak but real appearance signal (distinct
    clothing/vehicle colors separate reasonably well) -- explicitly not a
    production-quality substitute for a learned embedding; swap in
    `OnnxReidEmbedder` once trained weights are available.
    """

    def __init__(self, bins_per_channel: int = 8) -> None:
        self._bins = bins_per_channel

    def embed(self, crop: np.ndarray) -> np.ndarray:
        import cv2

        if crop.size == 0:
            return np.zeros(ReidConstants.EMBEDDING_DIM, dtype=np.float64)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [self._bins] * 3,
            [0, 180, 0, 256, 0, 256],
        ).flatten()
        histogram = _l2_normalize(histogram.astype(np.float64))

        # Pad/truncate to the standard embedding dimension so this fallback
        # is a drop-in replacement for `OnnxReidEmbedder` in the matcher.
        target_dim = ReidConstants.EMBEDDING_DIM
        if histogram.size >= target_dim:
            return histogram[:target_dim]
        padded = np.zeros(target_dim, dtype=np.float64)
        padded[: histogram.size] = histogram
        return padded
