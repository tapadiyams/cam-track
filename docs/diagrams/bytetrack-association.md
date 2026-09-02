# ByteTrack-lite: per-frame association flow

Referenced from `src/inference/tracker.py`'s `ByteTracker.update()`. Shows
the three association stages every frame's detections go through before
becoming published `Track`s -- see
[ADR 0002](../decisions/0002-tracker-choice.md) for why this shape (two
confidence tiers, three matching passes) beats a single-pass tracker.

```mermaid
flowchart TD
    START([New frame: detections + existing tracks]) --> PREDICT[Kalman-predict
every active track]
    PREDICT --> SPLIT{Split detections
by confidence}
    SPLIT -- "conf >= HIGH_CONF" --> HIGH[High-confidence
detections]
    SPLIT -- "LOW_CONF <= conf < HIGH_CONF" --> LOW[Low-confidence
detections]
    SPLIT -- "conf < LOW_CONF" --> DROP[[Discarded: too
noisy to associate]]

    HIGH --> S1{Stage 1: Hungarian match
vs. ALL active tracks
IoU >= MATCH_IOU_THRESHOLD}
    S1 -- matched --> UPDATE1[Update track:
Kalman-correct, hits+=1]
    S1 -- unmatched tracks --> S2POOL[Tracks left over]
    S1 -- unmatched detections --> S3POOL[High-conf detections
left over]

    LOW --> S2{Stage 2: Hungarian match
leftover tracks vs. LOW-conf dets
IoU >= LOW_CONF_MATCH_IOU_THRESHOLD}
    S2POOL --> S2
    S2 -- matched --> UPDATE2[Update track:
recovers through
occlusion/blur dip]
    S2 -- still unmatched --> AGE[mark_unmatched:
CONFIRMED to LOST,
or TENTATIVE to REMOVED]

    S3POOL --> S3{Stage 3: Hungarian match
TENTATIVE tracks only vs.
leftover high-conf dets
IoU >= UNCONFIRMED_IOU_THRESHOLD}
    S3 -- matched --> UPDATE3[Update tentative track
avoids duplicate identity]
    S3 -- still unmatched dets --> NEW[Spawn new TENTATIVE
track, hits=1]

    UPDATE1 --> CLEANUP
    UPDATE2 --> CLEANUP
    UPDATE3 --> CLEANUP
    NEW --> CLEANUP
    AGE --> CLEANUP{Cleanup: drop tracks
REMOVED or
time_since_update > MAX_LOST_FRAMES}
    CLEANUP --> OUTPUT([Return every
remaining track])
```
