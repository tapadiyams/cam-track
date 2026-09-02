# ADR 0005: ONNX export + dynamic batching for both edge and cloud inference

## Status
Accepted

## Context
The same detector needs to run in at least two very different
environments: a cloud GPU host serving many camera streams from one
process (maximize throughput), and an edge device physically near a
camera with limited compute and no reliable connection to a central
cluster (minimize latency and dependency footprint). The project brief
explicitly calls out this trade-off.

## Decision
- Export trained models to ONNX (`scripts/export_onnx.py`) and run
  inference through ONNX Runtime (`OnnxYoloDetector`) in every deployment
  target, rather than shipping the PyTorch/`ultralytics` runtime to
  production.
- Support INT8 post-training quantization at export time for edge targets.
- Use dynamic batching (`src/inference/batcher.py`) with a bounded latency
  budget rather than either no batching or fixed-size batching.
- Scale inference horizontally by running more stateless worker processes
  in the same consumer group (`ConsumerGroups.INFERENCE_WORKERS`), not by
  making a single worker handle more cameras.

## Why ONNX export instead of shipping PyTorch to production
- **Dependency footprint**: PyTorch (with CUDA support) is a multi-gigabyte
  dependency; ONNX Runtime's CPU build is a small fraction of that size,
  which matters directly for edge devices with limited storage and for
  cloud container image pull times/cold-start latency.
- **Execution provider portability**: the same `.onnx` file runs under
  ONNX Runtime's CPU, CUDA, TensorRT, OpenVINO, or CoreML execution
  providers depending on target hardware, without retraining or
  re-exporting per platform -- only the `providers` list passed to
  `OnnxYoloDetector` changes.
- **Quantization support**: INT8 post-training quantization is a
  first-class ONNX Runtime feature, giving edge CPUs (which usually lack
  fast FP16/FP32 tensor cores) a meaningful latency win at a small,
  usually acceptable, accuracy cost.
- **Decoupling training from serving**: fine-tuning stays entirely on the
  `ultralytics`/PyTorch side (`UltralyticsYoloDetector`, used only for
  training/dev); production inference code never imports `torch` at all,
  which also means a security/dependency vulnerability in the PyTorch
  ecosystem does not automatically become a production inference-worker
  vulnerability.

## Why dynamic batching instead of the alternatives
- **No batching (process frames one at a time)** wastes most of a GPU's
  parallelism -- a batch of 8 images through a convolutional network is
  nowhere near 8x the latency of one image, so throughput-per-dollar drops
  sharply without batching.
- **Fixed-size batching (always wait for N frames)** adds unbounded
  latency under light load: if traffic is slow, a worker could wait a long
  time for a batch that may not fill for seconds, which is unacceptable
  for anything resembling "real-time."
- **Dynamic batching** (flush on whichever bound -- size or a wait-time
  budget -- is hit first) gets most of the throughput benefit under load
  while capping worst-case added latency under light load to
  `max_batch_wait_ms`. This is the same strategy NVIDIA Triton's dynamic
  batcher and TensorFlow Serving's batching config use, for the same
  reason.

## Why horizontal scaling of stateless workers, not per-worker camera limits
Each `InferenceWorker` is stateless with respect to the broker (Redis
Streams' consumer groups own the work-distribution decision); the only
per-worker state is each camera's in-memory `ByteTracker`, which is why a
worker owns whichever cameras' frames it happens to be handed rather than
a fixed camera assignment. Adding inference throughput is therefore
"start another worker process" (`docker compose up --scale inference=N`,
or another pod in Kubernetes) -- no rebalancing logic, no sharding
configuration to maintain, and no single point that must know the full
camera topology.

**Caveat**: because ByteTrack state is per-worker and Redis Streams does
not guarantee a given camera's frames always land on the same worker
across redeliveries, a worker crash/restart can in principle hand a
camera's next frame to a different worker with a cold tracker, causing an
identity discontinuity for that camera at that moment. This scaffold
accepts that trade-off for horizontal-scaling simplicity; a deployment
that cannot tolerate it would need to pin cameras to workers (e.g. one
Redis Streams consumer group per camera) at some cost to elastic scaling.

## Consequences
- A trained checkpoint must go through `scripts/export_onnx.py` before it
  is usable in any deployed environment; there is no "just point
  production at the .pt file" path by design.
- Edge deployments still need to choose per-device execution providers and
  validate INT8 accuracy loss for their specific classes -- this ADR
  documents the *mechanism*, not a specific edge hardware target's tuned
  configuration.
