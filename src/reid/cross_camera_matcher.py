# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Matches per-camera tracks to a shared "global identity" within a zone.

A `zone` (see configs/cameras.yaml) is the unit of matching: cameras in the
same zone are assumed to physically overlap or sit close enough that the
same object can plausibly appear in more than one of them, so the gallery
-- and therefore the search space for a match -- is scoped per zone rather
than searched globally across the whole deployment. That keeps matching
O(gallery size within one zone) instead of O(every track ever seen).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from src.common.schemas import ReidMatch
from src.config.constants import ReidConstants


@dataclass
class _GalleryEntry:
    embedding: np.ndarray
    camera_id: str
    track_id: str
    observed_at_s: float


@dataclass
class _Identity:
    global_identity_id: str
    entries: list[_GalleryEntry] = field(default_factory=list)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


class CrossCameraMatcher:
    """Maintains one appearance gallery per zone and matches new sightings

    against it.

    Time per `observe()` call: O(g) where g is the number of gallery
    entries currently held for this zone (bounded by
    `GALLERY_MAX_ENTRIES_PER_IDENTITY * distinct identities seen recently`,
    since expired entries are pruned on every call) -- a linear scan for
    cosine similarity against every live entry, no ANN index. Fine at the
    scale of "identities active in one zone in the last
    `GALLERY_MAX_AGE_SECONDS`"; would need an approximate nearest-neighbor
    index (e.g. FAISS) if a zone's live gallery grows past a few thousand
    entries. Space: O(g) for the gallery itself.
    """

    def __init__(self, clock=time.time) -> None:
        self._clock = clock
        self._galleries: dict[str, list[_Identity]] = {}
        self._last_resolved_id: str = ""

    def observe(
        self,
        zone: str,
        camera_id: str,
        track_id: str,
        embedding: np.ndarray,
    ) -> ReidMatch | None:
        """Match `embedding` against `zone`'s gallery, or start a new identity.

        Returns a `ReidMatch` only when this observation matched an
        *existing* identity that was last seen on a *different* camera --
        that is the actual cross-camera re-identification event worth
        publishing. A same-camera re-match (recovering from occlusion) or a
        brand-new identity returns `None`; the caller still gets the
        assigned `global_identity_id` via the returned identity id on the
        `_Identity`, exposed through `resolve_identity_id`.
        """
        now = self._clock()
        identities = self._prune_and_get(zone, now)

        best_identity: _Identity | None = None
        best_similarity = -1.0
        best_entry: _GalleryEntry | None = None
        for identity in identities:
            for entry in identity.entries:
                similarity = _cosine_similarity(embedding, entry.embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_identity = identity
                    best_entry = entry

        if (
            best_identity is not None
            and best_similarity >= ReidConstants.COSINE_SIMILARITY_MATCH_THRESHOLD
        ):
            self._add_entry(best_identity, camera_id, track_id, embedding, now)
            self._last_resolved_id = best_identity.global_identity_id
            if best_entry is not None and best_entry.camera_id != camera_id:
                return ReidMatch(
                    zone=zone,
                    source_track_id=track_id,
                    source_camera_id=camera_id,
                    matched_track_id=best_entry.track_id,
                    matched_camera_id=best_entry.camera_id,
                    global_identity_id=best_identity.global_identity_id,
                    similarity=best_similarity,
                )
            return None

        new_identity = _Identity(global_identity_id=uuid.uuid4().hex)
        self._add_entry(new_identity, camera_id, track_id, embedding, now)
        identities.append(new_identity)
        self._last_resolved_id = new_identity.global_identity_id
        return None

    def resolve_identity_id(self) -> str:
        """The `global_identity_id` assigned by the most recent `observe()`.

        A small convenience so callers that only need the id (not a cross-
        camera match event) don't have to branch on `observe()`'s `None`
        return -- e.g. to stamp `TrackEvent.global_identity_id` every frame.
        """
        return self._last_resolved_id

    def _add_entry(
        self,
        identity: _Identity,
        camera_id: str,
        track_id: str,
        embedding: np.ndarray,
        now: float,
    ) -> None:
        identity.entries.append(_GalleryEntry(embedding, camera_id, track_id, now))
        max_entries = ReidConstants.GALLERY_MAX_ENTRIES_PER_IDENTITY
        if len(identity.entries) > max_entries:
            identity.entries = identity.entries[-max_entries:]

    def _prune_and_get(self, zone: str, now: float) -> list[_Identity]:
        identities = self._galleries.setdefault(zone, [])
        max_age = ReidConstants.GALLERY_MAX_AGE_SECONDS
        live_identities: list[_Identity] = []
        for identity in identities:
            identity.entries = [e for e in identity.entries if now - e.observed_at_s <= max_age]
            if identity.entries:
                live_identities.append(identity)
        self._galleries[zone] = live_identities
        return live_identities
