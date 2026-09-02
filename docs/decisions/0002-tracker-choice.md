# ADR 0002: ByteTrack over DeepSORT, with re-ID as a separate later stage

## Status
Accepted

## Context
Per-frame detections alone do not give a "how many distinct people crossed
this camera" or "did this person move from camera A to camera B" answer --
that requires associating detections into tracks over time (within one
camera) and then across cameras. This is the differentiating engineering
problem the project brief calls out.

## Decision
Use a ByteTrack-style tracker (`src/inference/tracker.py`) for within-camera
association, and a separate appearance-embedding gallery
(`src/reid/cross_camera_matcher.py`) for cross-camera identity matching,
rather than folding appearance matching into the tracker itself (as
DeepSORT does).

## What ByteTrack and DeepSORT are
Both are multi-object trackers that associate each frame's detections with
existing tracks using a motion model (a Kalman filter predicting where each
track should be) and the Hungarian algorithm for optimal bipartite
matching. DeepSORT additionally computes an appearance embedding for every
detection and folds a cosine-distance term into the association cost
alongside motion (IoU/Mahalanobis distance). ByteTrack instead runs
**two** motion-only association passes per frame: first high-confidence
detections against all tracks, then low-confidence detections (which most
trackers discard outright) against whatever tracks the first pass left
unmatched.

## Why ByteTrack for within-camera tracking
- **Occlusion/motion-blur recovery**: a real object's detection confidence
  dips during partial occlusion or fast motion -- exactly when you most
  need the tracker not to drop it. Discarding low-confidence detections
  (as a naive tracker, or DeepSORT's default confidence filtering, would)
  throws away real signal at the worst possible moment. ByteTrack's second
  association stage recovers many of these cases instead of terminating
  the track and starting a new identity a few frames later.
- **No embedding model in the hot loop**: ByteTrack's association is pure
  motion/IoU, computed from the Kalman filter's predicted box -- no
  forward pass through an embedding network per detection per frame. That
  keeps the tracker itself cheap enough to run at full frame rate even
  when the detector is the throughput bottleneck.
- **Published state of the art at low complexity**: ByteTrack (Zhang et
  al., 2022) is simpler to implement and reason about than DeepSORT's
  combined cost function, while matching or beating it on standard MOT
  benchmarks.

## Why not fold appearance matching into the tracker (DeepSORT-style)
- Running an embedding model on every detection in every frame, for every
  camera, is a real cost multiplier on the inference workers -- and it is
  mostly wasted, because within one camera, motion alone resolves the
  overwhelming majority of associations correctly (objects do not teleport
  between frames at 15-30fps).
- Appearance matching is genuinely needed for a different problem:
  identifying the *same* object across *different* cameras, where motion
  continuity does not exist at all (there is no shared coordinate space
  between two camera views). That is a fundamentally separate computation
  from within-camera frame-to-frame association, so keeping it a separate
  stage (`src/reid/`) that only runs on already-tracked objects -- not
  every raw detection -- means the expensive embedding step runs once per
  track update, not once per detection candidate.

## Consequences
- Within-camera identity switches are still possible when two objects
  cross paths with very similar motion (ByteTrack has no appearance signal
  to disambiguate that case); this is an accepted trade-off for the
  throughput this design achieves.
- Cross-camera re-identification depends entirely on the quality of the
  appearance embedding (`src/reid/embedder.py`) -- see that module's
  docstring for the gap between the shipped color-histogram fallback and a
  production-quality learned embedding.
