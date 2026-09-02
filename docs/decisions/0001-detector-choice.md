# ADR 0001: YOLOv8 over a two-stage detector

## Status
Accepted

## Context
The pipeline needs a per-frame object detector that runs fast enough for
real-time multi-camera video (tens of cameras, several fps each) on a mix
of cloud GPUs and, eventually, edge hardware.

## Decision
Use YOLOv8 (a single-stage, anchor-free detector) as the primary detector,
with a fine-tunable PyTorch path (`ultralytics`) for training and an ONNX
Runtime path for deployment (see ADR 0005).

## What it is
YOLOv8 predicts bounding boxes and class scores directly from a single
forward pass over the whole image (a "single-stage" detector). A two-stage
detector such as Faster R-CNN instead first proposes candidate regions
(a Region Proposal Network), then classifies and refines each proposal in
a second pass.

## Why YOLOv8
- **Latency budget**: a single forward pass is what makes real-time
  multi-camera throughput achievable at all -- the batching strategy in
  `src/inference/batcher.py` assumes single-digit-millisecond per-image
  inference, which two-stage detectors do not deliver on comparable
  hardware.
- **Deployment ecosystem**: first-class ONNX export, INT8 post-training
  quantization, and broad edge-runtime support (see ADR 0005) are
  well-trodden paths for YOLO specifically.
- **Fine-tuning cost**: a pretrained COCO checkpoint fine-tunes on a small
  labeled set (hundreds, not tens of thousands, of images) for a narrow
  domain like "retail foot traffic" or "warehouse pallets," which matters
  because most deployments of this pipeline will need domain-specific
  classes beyond COCO's.

## Why not a two-stage detector (e.g. Faster R-CNN, Cascade R-CNN)
Two-stage detectors generally achieve slightly higher mAP, especially on
small or heavily occluded objects, because the second stage refines each
proposal individually. That accuracy gain does not offset the cost here:
- Two-stage inference is meaningfully slower per image (the RPN and the
  per-proposal classifier are two separate network passes), which directly
  shrinks how many camera streams one inference worker can serve within a
  fixed latency budget.
- The accuracy gap matters most for small/rare objects in dense scenes;
  the target scenarios (retail foot traffic, warehouse inventory, traffic
  monitoring) mostly involve large, common object classes (people,
  vehicles, pallets) where YOLOv8 already performs well.
- Two-stage architectures are harder to export and quantize cleanly for
  edge deployment -- the RPN's dynamic proposal count does not map onto a
  fixed-shape ONNX graph as naturally as YOLO's fixed-size output tensor.

## Consequences
- Detection quality on small, distant, or heavily occluded objects will be
  weaker than a two-stage detector's; ByteTrack's low-confidence
  association stage (ADR 0002) is partly a mitigation for exactly this
  failure mode.
- Class coverage beyond COCO's 80 classes requires fine-tuning; the
  scaffold assumes this is done offline via `ultralytics`, not at
  inference time.
