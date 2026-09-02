# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Constant-velocity Kalman filter over a single bounding box.

This is the standard SORT/DeepSORT/ByteTrack state representation: track
`[u, v, s, r]` (box center x, center y, area, aspect ratio) plus their
velocities `[u_dot, v_dot, s_dot]` (aspect ratio is assumed constant frame
to frame, so it has no velocity term). Implemented directly with numpy
rather than pulling in `filterpy` -- the model is small and fixed, and
inlining it keeps the predict/update math auditable in one place.
"""

from __future__ import annotations

import numpy as np

_STATE_DIM = 7
_MEASUREMENT_DIM = 4


def _box_to_measurement(xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """Convert `(x1, y1, x2, y2)` to the filter's `[u, v, s, r]` measurement."""
    x1, y1, x2, y2 = xyxy
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    area = max(width, 0.0) * max(height, 0.0)
    aspect_ratio = width / height if height > 1e-6 else 0.0
    return np.array([center_x, center_y, area, aspect_ratio], dtype=np.float64)


def _state_to_box(state: np.ndarray) -> tuple[float, float, float, float]:
    """Convert the filter's `[u, v, s, r, ...]` state back to `(x1, y1, x2, y2)`."""
    center_x, center_y, area, aspect_ratio = state[0], state[1], state[2], state[3]
    area = max(area, 0.0)
    width = np.sqrt(area * aspect_ratio) if aspect_ratio > 0 else 0.0
    height = area / width if width > 1e-6 else 0.0
    return (
        float(center_x - width / 2.0),
        float(center_y - height / 2.0),
        float(center_x + width / 2.0),
        float(center_y + height / 2.0),
    )


class BoxKalmanFilter:
    """One Kalman filter instance tracking one box across frames.

    Time: O(1) per `predict()`/`update()` call -- all matrices are fixed at
    7x7 (state) or 4x7 (measurement), independent of how many tracks or
    frames exist. Space: O(1) -- one 7x7 covariance matrix per instance.
    """

    def __init__(self, initial_xyxy: tuple[float, float, float, float]) -> None:
        self._F = np.eye(_STATE_DIM)
        for i in range(3):
            self._F[i, i + 4] = 1.0  # u += u_dot, v += v_dot, s += s_dot

        self._H = np.zeros((_MEASUREMENT_DIM, _STATE_DIM))
        self._H[:4, :4] = np.eye(4)

        # Process noise: velocities are noisier/less certain than position.
        self._Q = np.eye(_STATE_DIM) * 1.0
        self._Q[4:, 4:] *= 0.01

        # Measurement noise: detector boxes are reasonably precise.
        self._R = np.eye(_MEASUREMENT_DIM) * 1.0
        self._R[2:, 2:] *= 10.0  # area/aspect-ratio measurements are noisier

        self._P = np.eye(_STATE_DIM) * 10.0
        self._P[4:, 4:] *= 1000.0  # velocity starts highly uncertain

        self._x = np.zeros((_STATE_DIM,))
        self._x[:4] = _box_to_measurement(initial_xyxy)

    def predict(self) -> tuple[float, float, float, float]:
        """Advance the state one frame and return the predicted box."""
        if self._x[6] + self._x[2] <= 0:  # predicted area would go non-positive
            self._x[6] = 0.0
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return _state_to_box(tuple(self._x[:4]))

    def update(self, measured_xyxy: tuple[float, float, float, float]) -> None:
        """Correct the state with an observed box (a matched detection)."""
        z = _box_to_measurement(measured_xyxy)
        y = z - self._H @ self._x  # innovation
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)  # Kalman gain
        self._x = self._x + K @ y
        self._P = (np.eye(_STATE_DIM) - K @ self._H) @ self._P

    def current_box(self) -> tuple[float, float, float, float]:
        return _state_to_box(tuple(self._x[:4]))
