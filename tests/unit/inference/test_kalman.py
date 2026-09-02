# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/inference/kalman.py -- the constant-velocity box filter."""

from src.inference.kalman import BoxKalmanFilter


def _approx_box(box, expected, tolerance=1.0):
    return all(abs(a - b) <= tolerance for a, b in zip(box, expected, strict=True))


def test_predict_without_update_holds_position_with_no_velocity_prior():
    """A freshly created filter has zero velocity prior, so its first

    predict (before any `update()`) should not move the box materially.
    """
    kf = BoxKalmanFilter(initial_xyxy=(10.0, 10.0, 30.0, 30.0))
    predicted = kf.predict()
    assert _approx_box(predicted, (10.0, 10.0, 30.0, 30.0), tolerance=2.0)


def test_filter_tracks_a_box_moving_at_constant_velocity():
    """Feed a box moving +5px/frame in x; after a few predict/update

    cycles the filter should learn the velocity and its *prediction*
    (before seeing the next measurement) should already be close to where
    the object actually is.
    """
    kf = BoxKalmanFilter(initial_xyxy=(0.0, 0.0, 20.0, 20.0))

    x_offset = 0.0
    for _ in range(10):
        kf.predict()
        x_offset += 5.0
        kf.update((x_offset, 0.0, x_offset + 20.0, 20.0))

    next_predicted = kf.predict()
    expected_x_offset = x_offset + 5.0
    assert _approx_box(
        next_predicted,
        (expected_x_offset, 0.0, expected_x_offset + 20.0, 20.0),
        tolerance=3.0,
    )


def test_update_corrects_toward_the_measurement():
    """A large, sudden jump in the measured box should pull the filter's

    state substantially toward it, not leave it near the stale prediction.
    """
    kf = BoxKalmanFilter(initial_xyxy=(0.0, 0.0, 20.0, 20.0))
    kf.predict()
    kf.update((0.0, 0.0, 20.0, 20.0))

    kf.predict()
    kf.update((200.0, 200.0, 220.0, 220.0))  # object jumped far away

    box = kf.current_box()
    center_x = (box[0] + box[2]) / 2.0
    assert center_x > 50.0  # moved substantially toward the new measurement
