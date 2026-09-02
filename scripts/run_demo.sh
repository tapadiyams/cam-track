#!/usr/bin/env bash
# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
#
# Brings up the full stack via Docker Compose and prints where to look.
# Prerequisite: an ONNX model at models/yolov8n.onnx -- see models/README.md.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -f models/yolov8n.onnx ]; then
    echo "models/yolov8n.onnx not found -- see models/README.md for how to get one." >&2
    exit 1
fi

docker compose up --build --scale inference=2 -d

echo "Stack is up. Dashboard: http://localhost:8080"
echo "Tail logs with: docker compose logs -f"
echo "Stop with: docker compose down"
