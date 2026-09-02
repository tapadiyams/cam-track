# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/reid/cross_camera_matcher.py."""

import numpy as np

from src.reid.cross_camera_matcher import CrossCameraMatcher


def _unit_vector(*values):
    vector = np.array(values, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def test_first_sighting_in_a_zone_creates_a_new_identity_and_returns_none():
    matcher = CrossCameraMatcher()
    result = matcher.observe("zone-a", "cam-01", "track-1", _unit_vector(1, 0, 0))
    assert result is None
    assert matcher.resolve_identity_id() != ""


def test_same_object_seen_on_a_different_camera_returns_a_cross_camera_match():
    matcher = CrossCameraMatcher()
    matcher.observe("zone-a", "cam-01", "track-1", _unit_vector(1, 0, 0))

    # Same appearance, different camera and track id -- this is exactly what
    # a cross-camera re-identification event looks like.
    match = matcher.observe("zone-a", "cam-02", "track-9", _unit_vector(1, 0, 0.001))

    assert match is not None
    assert match.source_camera_id == "cam-02"
    assert match.matched_camera_id == "cam-01"
    assert match.similarity > 0.99


def test_same_object_seen_again_on_the_same_camera_does_not_report_a_match():
    """Re-matching within the same camera is identity continuity (e.g.

    recovering from a brief tracker dropout), not a *cross-camera* event --
    the dashboard should not count this as a new re-ID sighting.
    """
    matcher = CrossCameraMatcher()
    matcher.observe("zone-a", "cam-01", "track-1", _unit_vector(1, 0, 0))
    result = matcher.observe("zone-a", "cam-01", "track-2", _unit_vector(1, 0, 0.001))
    assert result is None


def test_dissimilar_appearance_creates_a_separate_identity():
    matcher = CrossCameraMatcher()
    matcher.observe("zone-a", "cam-01", "track-1", _unit_vector(1, 0, 0))
    matcher.observe("zone-a", "cam-01", "track-1", _unit_vector(1, 0, 0))
    first_identity = matcher.resolve_identity_id()

    matcher.observe("zone-a", "cam-02", "track-2", _unit_vector(0, 1, 0))
    second_identity = matcher.resolve_identity_id()

    assert first_identity != second_identity


def test_zones_are_isolated_from_each_other():
    """The same appearance vector observed in two different zones must not

    match across zones -- each zone is a separate physical area, and a
    cross-zone "match" would be a false positive by construction.
    """
    matcher = CrossCameraMatcher()
    matcher.observe("zone-a", "cam-01", "track-1", _unit_vector(1, 0, 0))
    result = matcher.observe("zone-b", "cam-02", "track-2", _unit_vector(1, 0, 0))
    assert result is None

    # Confirm it became a brand-new identity in zone-b, not a silent no-op.
    identity_in_zone_b = matcher.resolve_identity_id()
    matcher.observe("zone-a", "cam-01", "track-1", _unit_vector(1, 0, 0))
    identity_in_zone_a = matcher.resolve_identity_id()
    assert identity_in_zone_a != identity_in_zone_b


def test_gallery_entries_older_than_max_age_are_pruned():
    fake_time = {"now": 0.0}
    matcher = CrossCameraMatcher(clock=lambda: fake_time["now"])

    matcher.observe("zone-a", "cam-01", "track-1", _unit_vector(1, 0, 0))
    first_identity = matcher.resolve_identity_id()

    from src.config.constants import ReidConstants

    fake_time["now"] += ReidConstants.GALLERY_MAX_AGE_SECONDS + 1

    # The old entry should have expired, so this is treated as a brand-new
    # identity rather than matched against the (now-stale) first sighting.
    matcher.observe("zone-a", "cam-02", "track-2", _unit_vector(1, 0, 0))
    second_identity = matcher.resolve_identity_id()

    assert first_identity != second_identity
