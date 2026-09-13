#!/usr/bin/env bash
# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
#
# Brings up the full stack via Docker Compose and prints where to look.
#
# Checks every prerequisite first and fails with a clear, specific message
# instead of a cryptic Docker/Python error mid-startup -- see
# TROUBLESHOOTING.md for the actual errors this replaces (e.g. a bare
# "docker: command not found" with no indication of what to do about it).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- 1. Docker must be installed. ---
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker was not found on this machine." >&2
    if command -v brew >/dev/null 2>&1; then
        echo "Installing Docker Desktop via Homebrew (brew install --cask docker)..." >&2
        brew install --cask docker
        echo "" >&2
        echo "Docker Desktop is installed but has never been opened. Open it from" >&2
        echo "Applications (or run: open -a Docker), wait for the whale icon in your" >&2
        echo "menu bar to settle, then re-run this script." >&2
    else
        echo "Install Docker Desktop from https://www.docker.com/products/docker-desktop," >&2
        echo "open it, then re-run this script." >&2
    fi
    exit 1
fi

# --- 2. Docker must actually be running, not just installed. ---
if ! docker info >/dev/null 2>&1; then
    echo "Docker is installed but not running." >&2
    echo "Open Docker Desktop (the whale icon in your menu bar), wait for it to" >&2
    echo "finish starting, then re-run this script." >&2
    exit 1
fi

# --- 3. Docker Compose v2 (the 'docker compose' subcommand, no hyphen). ---
if ! docker compose version >/dev/null 2>&1; then
    echo "'docker compose' is not available -- update Docker Desktop to a" >&2
    echo "recent version (it bundles Compose v2)." >&2
    exit 1
fi

# --- 4. .env must exist -- create it from the example if this is a first run. ---
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo ".env not found -- created it from .env.example with default values."
    else
        echo ".env and .env.example are both missing -- cannot continue." >&2
        exit 1
    fi
fi

# --- 5. The exported detector model must exist. ---
if [ ! -f models/yolov8n.onnx ]; then
    echo "models/yolov8n.onnx not found -- see models/README.md and the" >&2
    echo "'How to run it' section of README.md for how to get one." >&2
    exit 1
fi

# --- 6. Every local-file camera source in cameras.yaml must actually exist. ---
# RTSP/network sources (anything containing "://") are skipped -- there's
# nothing on disk to check for those. This is what catches "which video do
# I need to put where" before it becomes a confusing runtime failure inside
# a container instead of a clear message here.
CAMERA_CONFIG="configs/cameras.yaml"
if [ ! -f "$CAMERA_CONFIG" ]; then
    echo "$CAMERA_CONFIG not found." >&2
    exit 1
fi

missing_sources=0
while IFS= read -r raw_source; do
    source_value="$(echo "$raw_source" | sed -E 's/^[[:space:]]*source:[[:space:]]*//' | tr -d '"'"'"'')"
    case "$source_value" in
        *://*) ;;  # RTSP/network URL -- nothing local to check
        "") ;;
        *)
            if [ ! -f "$source_value" ]; then
                echo "Camera source not found: $source_value (referenced in $CAMERA_CONFIG)" >&2
                missing_sources=1
            fi
            ;;
    esac
done < <(grep -E '^[[:space:]]*source:' "$CAMERA_CONFIG")

if [ "$missing_sources" -eq 1 ]; then
    echo "" >&2
    echo "Fix: save a video file at each missing path above (see" >&2
    echo "sample_data/README.md), or point that camera's 'source:' entry in" >&2
    echo "$CAMERA_CONFIG at a file that exists." >&2
    exit 1
fi

# --- All checks passed -- bring up the stack. ---
docker compose up --build --scale inference=2 -d

echo "Stack is up. Dashboard: http://localhost:8080"
echo "Tail logs with: docker compose logs -f"
echo "Stop with: docker compose down"
