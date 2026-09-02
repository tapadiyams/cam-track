# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for the pure post-processing helpers in src/inference/detector.py.

Deliberately does not instantiate `OnnxYoloDetector` or
`UltralyticsYoloDetector` (both require the optional ONNX Runtime /
ultralytics packages -- see requirements-ml.txt); the NMS, IoU, and output
decode logic those classes call is implemented as free functions precisely
so it can be tested without either heavy dependency installed.
"""

import numpy as np

from src.inference.detector import _decode_yolo_output, _iou_xyxy, _nms


def test_iou_xyxy_identical_boxes_have_iou_one():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    others = np.array([[0.0, 0.0, 10.0, 10.0]])
    result = _iou_xyxy(box, others)
    assert result[0] == 1.0


def test_iou_xyxy_disjoint_boxes_have_iou_zero():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    others = np.array([[100.0, 100.0, 110.0, 110.0]])
    result = _iou_xyxy(box, others)
    assert result[0] == 0.0


def test_iou_xyxy_half_overlap():
    box = np.array([0.0, 0.0, 10.0, 10.0])  # area 100
    others = np.array([[5.0, 0.0, 15.0, 10.0]])  # area 100, overlap area 50
    result = _iou_xyxy(box, others)
    # intersection 50, union 100+100-50=150 -> IoU = 1/3
    assert abs(result[0] - (50.0 / 150.0)) < 1e-9


def test_nms_suppresses_heavily_overlapping_lower_score_box():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # heavily overlaps box 0
            [100.0, 100.0, 110.0, 110.0],  # independent
        ]
    )
    scores = np.array([0.9, 0.8, 0.7])
    kept = _nms(boxes, scores, iou_threshold=0.5)
    assert kept == [0, 2]  # box 1 suppressed by box 0; box 2 always kept


def test_nms_keeps_all_boxes_when_none_overlap():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]])
    scores = np.array([0.5, 0.6])
    kept = _nms(boxes, scores, iou_threshold=0.5)
    assert sorted(kept) == [0, 1]


def _make_yolo_raw_output(cx, cy, w, h, class_scores):
    """Build a raw YOLOv8-ONNX-shaped output `(4 + num_classes, 1)` for one

    candidate box.
    """
    box_row = np.array([[cx], [cy], [w], [h]])
    class_rows = np.array([[s] for s in class_scores])
    return np.vstack([box_row, class_rows])


def test_decode_yolo_output_recovers_box_in_original_image_coordinates():
    # A 20x20 box centered at (50, 50) in a 640x640 model input; scaling by
    # 2.0 in both axes should land it at (80,80)-(120,120) in a 1280x1280
    # original image.
    raw = _make_yolo_raw_output(cx=50, cy=50, w=20, h=20, class_scores=[0.9, 0.1])
    class_names = {0: "person", 1: "car"}

    detections = _decode_yolo_output(
        raw,
        frame_id="f1",
        camera_id="cam-01",
        scale_x=2.0,
        scale_y=2.0,
        class_names=class_names,
        confidence_threshold=0.25,
        iou_threshold=0.45,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_name == "person"
    assert detection.confidence == 0.9
    assert detection.box.x1 == 80.0
    assert detection.box.y1 == 80.0
    assert detection.box.x2 == 120.0
    assert detection.box.y2 == 120.0


def test_decode_yolo_output_filters_below_confidence_threshold():
    raw = _make_yolo_raw_output(cx=50, cy=50, w=20, h=20, class_scores=[0.1, 0.05])
    detections = _decode_yolo_output(
        raw,
        frame_id="f1",
        camera_id="cam-01",
        scale_x=1.0,
        scale_y=1.0,
        class_names={0: "person", 1: "car"},
        confidence_threshold=0.25,
        iou_threshold=0.45,
    )
    assert detections == []
