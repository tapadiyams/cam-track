#!/usr/bin/env bash
# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
#
# Installs the optional ML/edge-runtime dependencies (requirements-ml.txt):
# ultralytics, onnxruntime, and confluent-kafka. confluent-kafka needs the
# librdkafka C library to compile -- this checks for it (and installs it
# via Homebrew/apt where possible) first, instead of letting 'pip install'
# fail partway through with an opaque compiler error.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "No virtualenv appears to be active." >&2
    echo "Run './scripts/setup.sh' first if you haven't, then" >&2
    echo "'source .venv/bin/activate' (or .venv2, etc.) before running this." >&2
    exit 1
fi

if command -v brew >/dev/null 2>&1; then
    if ! brew list librdkafka >/dev/null 2>&1; then
        echo "librdkafka not found -- installing via Homebrew..."
        brew install librdkafka
    fi
elif command -v apt-get >/dev/null 2>&1; then
    if ! pkg-config --exists rdkafka 2>/dev/null; then
        echo "librdkafka not found -- installing via apt-get..."
        sudo apt-get update && sudo apt-get install -y librdkafka-dev
    fi
else
    echo "Could not detect Homebrew or apt-get to check/install librdkafka" >&2
    echo "automatically. If 'pip install' below fails while building" >&2
    echo "confluent-kafka, install librdkafka yourself first (see" >&2
    echo "https://github.com/confluentinc/librdkafka)." >&2
fi

pip install -r requirements-ml.txt
echo "ML dependencies installed."
