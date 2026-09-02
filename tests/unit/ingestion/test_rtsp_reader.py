# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/ingestion/rtsp_reader.py -- reconnect/backoff and fps cap.

Everything here runs against a fake `cv2.VideoCapture` (no real camera or
video file needed): the class under test only calls `isOpened`, `read`,
`set`, and `release`, so a minimal stand-in is enough to exercise every
branch of the reconnect state machine.
"""

import numpy as np

import src.ingestion.rtsp_reader as rtsp_reader_module
from src.ingestion.rtsp_reader import RtspFrameReader

_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


class _FakeCapture:
    """Stands in for `cv2.VideoCapture`. `open_results` and `read_results`

    are consumed in order, one per matching call; the last value repeats
    once exhausted so tests don't have to enumerate every future call.
    """

    def __init__(self, open_results, read_results):
        self._open_results = list(open_results)
        self._read_results = list(read_results)
        self.released = False

    def isOpened(self):
        if not self._open_results:
            return True
        return self._open_results.pop(0) if len(self._open_results) > 1 else self._open_results[0]

    def set(self, *_args, **_kwargs):
        pass

    def read(self):
        if not self._read_results:
            return (True, _FRAME)
        return self._read_results.pop(0) if len(self._read_results) > 1 else self._read_results[0]

    def release(self):
        self.released = True


def test_frames_reconnects_after_initial_connect_failure(monkeypatch):
    captures = [
        _FakeCapture(open_results=[False], read_results=[]),
        _FakeCapture(open_results=[True], read_results=[(True, _FRAME)]),
    ]
    monkeypatch.setattr(rtsp_reader_module.cv2, "VideoCapture", lambda _src: captures.pop(0))
    monkeypatch.setattr(rtsp_reader_module.time, "sleep", lambda _seconds: None)

    reader = RtspFrameReader(source="rtsp://fake", fps_cap=1000)
    generator = reader.frames()

    first_frame = next(generator)
    assert first_frame.sequence_number == 1


def test_frames_reconnects_after_a_failed_read(monkeypatch):
    capture = _FakeCapture(
        open_results=[True],
        read_results=[(False, None), (True, _FRAME), (True, _FRAME)],
    )
    monkeypatch.setattr(rtsp_reader_module.cv2, "VideoCapture", lambda _src: capture)
    monkeypatch.setattr(rtsp_reader_module.time, "sleep", lambda _seconds: None)

    reader = RtspFrameReader(source="rtsp://fake", fps_cap=1000)
    generator = reader.frames()

    frame = next(generator)
    assert frame.sequence_number == 1
    assert capture.released is True  # release() was called on the failed read


def test_frames_drops_frames_faster_than_fps_cap(monkeypatch):
    capture = _FakeCapture(open_results=[True], read_results=[])
    monkeypatch.setattr(rtsp_reader_module.cv2, "VideoCapture", lambda _src: capture)

    # One `time.monotonic()` read per loop iteration: frame 1 emits at
    # 100.0s, the 100.05s read is dropped (< the 1s min interval for a
    # 1 fps cap), and the 101.0s read emits frame 2.
    fake_clock = iter([100.0, 100.05, 101.0])
    monkeypatch.setattr(rtsp_reader_module.time, "monotonic", lambda: next(fake_clock))

    reader = RtspFrameReader(source="rtsp://fake", fps_cap=1)  # 1 fps => 1s min interval
    generator = reader.frames()

    first = next(generator)
    second = next(generator)  # only 1s later does the cap allow another frame
    assert first.sequence_number == 1
    assert second.sequence_number == 2


def test_close_releases_the_underlying_capture(monkeypatch):
    capture = _FakeCapture(open_results=[True], read_results=[(True, _FRAME)])
    monkeypatch.setattr(rtsp_reader_module.cv2, "VideoCapture", lambda _src: capture)

    reader = RtspFrameReader(source="rtsp://fake", fps_cap=1000)
    next(reader.frames())
    reader.close()

    assert capture.released is True
