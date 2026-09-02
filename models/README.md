# models/

Trained model weights are not committed to this repository. Place the
following here (or point `settings.detector_weights_path` /
`settings.reid_weights_path` elsewhere) before running the demo:

- `yolov8n.onnx` -- exported with `python scripts/export_onnx.py
  --weights yolov8n.pt --output models/yolov8n.onnx`. A pretrained
  `yolov8n.pt` can be downloaded via the `ultralytics` package
  (`from ultralytics import YOLO; YOLO("yolov8n.pt")` fetches it on first
  use) for a COCO-class demo; fine-tune on your own data for anything
  beyond a generic person/vehicle demo.
- `reid_resnet18.onnx` -- optional. Without it, `src/reid/embedder.py`'s
  `ColorHistogramEmbedder` fallback is used, which is dependency-free but
  a much weaker appearance signal (see that module's docstring).
