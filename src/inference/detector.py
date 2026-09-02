# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Object detector: a `Detector` interface plus two backends.

`UltralyticsYoloDetector` (PyTorch, via the `ultralytics` package) is the
training/dev-time backend -- easy to fine-tune, easy to debug. `OnnxYolo
Detector` (ONNX Runtime) is the deployment backend, produced by
`scripts/export_onnx.py`, and is what the inference workers actually load
in docker-compose.yml and at the edge. See
docs/decisions/0001-detector-choice.md for why YOLOv8 over a two-stage
detector, and docs/decisions/0005-edge-vs-cloud.md for why ONNX export
matters for edge deployment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.common.schemas import BoundingBox, Detection
from src.config.constants import InferenceConstants


class Detector(ABC):
    """Runs object detection on a batch of frames."""

    @abstractmethod
    def detect_batch(
        self, frame_ids: list[str], camera_ids: list[str], images: list[np.ndarray]
    ) -> list[list[Detection]]:
        """Return one detection list per input image, same order as input.

        Time: O(b) forward passes are fused into one batched GPU/CPU call
        by the underlying runtime (not b separate calls), where b is
        `len(images)`; post-processing (NMS) is O(b * k log k) for k raw
        boxes per image. Space: O(b * k) for the raw and filtered boxes.
        """


def _postprocess_ultralytics_result(
    result, frame_id: str, camera_id: str, class_names: dict[int, str]
) -> list[Detection]:
    """Convert one Ultralytics `Results` object into our `Detection` schema.

    Time: O(k) for k boxes in this frame's result. Space: O(k).
    """
    detections: list[Detection] = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        detections.append(
            Detection(
                frame_id=frame_id,
                camera_id=camera_id,
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                class_id=class_id,
                class_name=class_names.get(class_id, str(class_id)),
                confidence=float(box.conf[0]),
            )
        )
    return detections


class UltralyticsYoloDetector(Detector):
    """YOLOv8 via the `ultralytics` package. Dev/training-time backend."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        confidence_threshold: float = InferenceConstants.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = InferenceConstants.DEFAULT_IOU_NMS_THRESHOLD,
    ) -> None:
        from ultralytics import YOLO  # imported lazily: heavy, torch-dependent

        self._model = YOLO(weights_path)
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold

    def detect_batch(
        self, frame_ids: list[str], camera_ids: list[str], images: list[np.ndarray]
    ) -> list[list[Detection]]:
        if len(frame_ids) != len(images) or len(camera_ids) != len(images):
            raise ValueError("frame_ids, camera_ids, and images must be the same length")
        if not images:
            return []

        results = self._model.predict(
            images,
            device=self._device,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            imgsz=InferenceConstants.DEFAULT_INPUT_SIZE_PX,
            verbose=False,
        )
        class_names = self._model.names
        return [
            _postprocess_ultralytics_result(result, frame_id, camera_id, class_names)
            for result, frame_id, camera_id in zip(results, frame_ids, camera_ids, strict=True)
        ]


class OnnxYoloDetector(Detector):
    """YOLOv8 exported to ONNX and run via ONNX Runtime. Deployment backend.

    Trades the flexibility of the PyTorch model for a smaller runtime with
    no CUDA/PyTorch dependency, quantization support (INT8), and
    consistent latency across CPU/GPU/edge-accelerator execution providers
    -- see docs/decisions/0005-edge-vs-cloud.md.
    """

    def __init__(
        self,
        onnx_path: str,
        class_names: dict[int, str],
        input_size_px: int = InferenceConstants.DEFAULT_INPUT_SIZE_PX,
        confidence_threshold: float = InferenceConstants.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = InferenceConstants.DEFAULT_IOU_NMS_THRESHOLD,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._class_names = class_names
        self._input_size_px = input_size_px
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Letterbox-free resize to a square input; returns scale factors

        needed to map predicted boxes back to the original image size.
        """
        size = self._input_size_px
        resized = _resize_bgr(image, size, size)
        blob = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        scale_x = image.shape[1] / size
        scale_y = image.shape[0] / size
        return blob, scale_x, scale_y

    def detect_batch(
        self, frame_ids: list[str], camera_ids: list[str], images: list[np.ndarray]
    ) -> list[list[Detection]]:
        if len(frame_ids) != len(images) or len(camera_ids) != len(images):
            raise ValueError("frame_ids, camera_ids, and images must be the same length")

        detections: list[list[Detection]] = []
        for frame_id, camera_id, image in zip(frame_ids, camera_ids, images, strict=True):
            blob, scale_x, scale_y = self._preprocess(image)
            raw_output = self._session.run(None, {self._input_name: blob})[0]
            detections.append(
                _decode_yolo_output(
                    raw_output[0],
                    frame_id,
                    camera_id,
                    scale_x,
                    scale_y,
                    self._class_names,
                    self._confidence_threshold,
                    self._iou_threshold,
                )
            )
        return detections


def _resize_bgr(image: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    return cv2.resize(image, (width, height))


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between box `a` (1, 4) and boxes `b` (n, 4).

    Time: O(n). Space: O(n) for the intermediate arrays.
    """
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a + area_b - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-max suppression. Returns indices of kept boxes, highest

    score first.

    Time: O(n^2) worst case (n boxes, each compared against remaining
    survivors) -- fine for YOLO's post-NMS candidate counts (low
    hundreds), not appropriate unmodified for n in the tens of thousands.
    Space: O(n) for the survivor bookkeeping.
    """
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break
        remaining = order[1:]
        ious = _iou_xyxy(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]
    return keep


def _decode_yolo_output(
    raw: np.ndarray,
    frame_id: str,
    camera_id: str,
    scale_x: float,
    scale_y: float,
    class_names: dict[int, str],
    confidence_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    """Decode a raw YOLOv8 ONNX output tensor (shape `(4 + num_classes, n)`)

    into `Detection` objects in original-image pixel coordinates.

    Time: O(n * c) to find the best class per candidate (n candidates, c
    classes), plus O(m^2) for NMS over the m candidates surviving the
    confidence threshold. Space: O(n).
    """
    predictions = raw.T  # (n, 4 + num_classes)
    box_params = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)
    keep_mask = confidences >= confidence_threshold
    if not keep_mask.any():
        return []

    box_params, class_ids, confidences = (
        box_params[keep_mask],
        class_ids[keep_mask],
        confidences[keep_mask],
    )

    # YOLOv8 exports (cx, cy, w, h) in the model's input-pixel space.
    cx, cy, w, h = box_params[:, 0], box_params[:, 1], box_params[:, 2], box_params[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    kept_indices = _nms(xyxy, confidences, iou_threshold)

    detections: list[Detection] = []
    for idx in kept_indices:
        x1, y1, x2, y2 = xyxy[idx]
        class_id = int(class_ids[idx])
        detections.append(
            Detection(
                frame_id=frame_id,
                camera_id=camera_id,
                box=BoundingBox(
                    x1=float(x1 * scale_x),
                    y1=float(y1 * scale_y),
                    x2=float(x2 * scale_x),
                    y2=float(y2 * scale_y),
                ),
                class_id=class_id,
                class_name=class_names.get(class_id, str(class_id)),
                confidence=float(confidences[idx]),
            )
        )
    return detections
