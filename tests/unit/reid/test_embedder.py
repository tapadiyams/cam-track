# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/reid/embedder.py -- the dependency-free color-histogram

fallback embedder (`OnnxReidEmbedder` needs trained weights this scaffold
does not ship, so it is not exercised here).
"""

import numpy as np

from src.config.constants import ReidConstants
from src.reid.embedder import ColorHistogramEmbedder


def _solid_color_crop(bgr, size=32):
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    crop[:, :] = bgr
    return crop


def test_embedding_has_expected_dimension_and_unit_norm():
    embedder = ColorHistogramEmbedder()
    embedding = embedder.embed(_solid_color_crop((0, 0, 255)))  # solid red (BGR)
    assert embedding.shape == (ReidConstants.EMBEDDING_DIM,)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-6 or norm == 0.0


def test_identical_crops_produce_identical_embeddings():
    embedder = ColorHistogramEmbedder()
    a = embedder.embed(_solid_color_crop((0, 255, 0)))
    b = embedder.embed(_solid_color_crop((0, 255, 0)))
    assert np.allclose(a, b)


def test_distinct_colors_produce_less_similar_embeddings_than_identical_ones():
    embedder = ColorHistogramEmbedder()
    red = embedder.embed(_solid_color_crop((0, 0, 255)))
    also_red = embedder.embed(_solid_color_crop((0, 0, 255)))
    blue = embedder.embed(_solid_color_crop((255, 0, 0)))

    same_color_similarity = float(np.dot(red, also_red))
    different_color_similarity = float(np.dot(red, blue))
    assert same_color_similarity > different_color_similarity


def test_empty_crop_returns_zero_vector_instead_of_raising():
    embedder = ColorHistogramEmbedder()
    embedding = embedder.embed(np.zeros((0, 0, 3), dtype=np.uint8))
    assert embedding.shape == (ReidConstants.EMBEDDING_DIM,)
    assert np.all(embedding == 0.0)
