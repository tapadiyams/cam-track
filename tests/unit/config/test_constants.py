# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/config/constants.py.

Mostly guards against accidental regressions to values other modules
depend on for correctness (not just "does the constant exist").
"""

from src.config.constants import InferenceConstants, ReidConstants, RetryConstants


def test_track_confidence_thresholds_are_ordered():
    """The tracker's two-stage logic assumes low < high; a regression here

    would silently merge or invert the two association stages.
    """
    low = InferenceConstants.TRACK_LOW_CONF_THRESHOLD
    high = InferenceConstants.TRACK_HIGH_CONF_THRESHOLD
    assert low < high


def test_track_iou_thresholds_increase_by_stage():
    """Stage 3 (unconfirmed) must be stricter than stage 2, which must be

    stricter than stage 1 is lenient -- see the rationale comments in
    src/config/constants.py and src/inference/tracker.py.
    """
    assert (
        InferenceConstants.TRACK_MATCH_IOU_THRESHOLD
        < InferenceConstants.TRACK_LOW_CONF_MATCH_IOU_THRESHOLD
        < InferenceConstants.TRACK_UNCONFIRMED_MATCH_IOU_THRESHOLD
    )


def test_reid_similarity_threshold_is_a_valid_cosine_bound():
    assert 0.0 < ReidConstants.COSINE_SIMILARITY_MATCH_THRESHOLD <= 1.0


def test_retry_backoff_is_monotonically_increasing():
    """A misconfigured multiplier <= 1 would make backoff useless (constant

    or shrinking retry delay under sustained failure).
    """
    assert RetryConstants.BACKOFF_MULTIPLIER > 1.0
    assert RetryConstants.BACKOFF_BASE_SECONDS > 0
    assert RetryConstants.BACKOFF_MAX_SECONDS >= RetryConstants.BACKOFF_BASE_SECONDS
