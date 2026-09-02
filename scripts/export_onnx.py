#!/usr/bin/env python3
# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Export a trained YOLOv8 (.pt) checkpoint to ONNX for deployment.

Usage:
    python scripts/export_onnx.py --weights models/yolov8n.pt \\
        --output models/yolov8n.onnx [--imgsz 640] [--dynamic] [--int8]

Run once per trained checkpoint, not at service startup -- `OnnxYoloDetector`
(src/inference/detector.py) only ever loads the resulting .onnx file.
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to a YOLOv8 .pt checkpoint")
    parser.add_argument("--output", required=True, help="Path to write the .onnx file")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input resolution")
    parser.add_argument(
        "--dynamic", action="store_true", help="Export with a dynamic batch axis"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Apply INT8 post-training quantization (smaller, faster on edge CPUs)",
    )
    return parser.parse_args(argv)


def export(weights: str, output: str, imgsz: int, dynamic: bool, int8: bool) -> str:
    """Run the export and return the path Ultralytics actually wrote to.

    Ultralytics writes next to `weights` by convention (`<name>.onnx`); we
    move the result to `output` afterward so callers control the final
    path regardless of that convention.
    """
    import shutil

    from ultralytics import YOLO

    model = YOLO(weights)
    exported_path = model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, int8=int8)
    if str(exported_path) != output:
        shutil.move(str(exported_path), output)
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    written_path = export(args.weights, args.output, args.imgsz, args.dynamic, args.int8)
    print(f"Exported ONNX model to {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
