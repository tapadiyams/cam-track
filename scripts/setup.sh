#!/usr/bin/env bash
# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
#
# One-time local dev setup: check for python3, create a virtualenv, install
# dependencies, and seed .env.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- 1. Make sure python3 exists, offering to install it if not. ---
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 was not found on this machine."
    if command -v brew >/dev/null 2>&1; then
        echo "Installing Python via Homebrew (brew install python@3.11)..."
        brew install python@3.11
    elif command -v apt-get >/dev/null 2>&1; then
        echo "Installing Python via apt-get..."
        sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    else
        echo "Could not find a package manager to install Python automatically."
        echo "Install Python 3.11+ yourself from https://www.python.org/downloads/"
        echo "(or install Homebrew from https://brew.sh, then re-run this script), then try again."
        exit 1
    fi
fi

PYTHON_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "Using python3 $PYTHON_VERSION ($(command -v python3))"

# --- 2. Pick a virtualenv directory. ---
# Reuses ".venv" when it is a healthy, already-created virtualenv. If
# ".venv" exists but is missing "bin/activate" (an interrupted/corrupted
# creation, or -- as happened once during development -- a directory a
# stale filesystem lock made impossible to delete or rename), this falls
# forward to ".venv2", ".venv3", etc. instead of failing outright, so a
# single wedged leftover directory never blocks setup from working.
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    n=2
    while [ -d ".venv${n}" ] && [ ! -f ".venv${n}/bin/activate" ]; do
        n=$((n + 1))
    done
    VENV_DIR=".venv${n}"
    echo "Existing .venv looks incomplete/broken -- using ${VENV_DIR} instead."
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# --- 3. Install dependencies into it. ---
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

# --- 4. Seed .env if this is a first run. ---
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example -- edit it for your environment."
fi

echo ""
echo "Setup complete. Activate with: source ${VENV_DIR}/bin/activate"
echo "Run the test suite with: pytest"
echo "For the ONNX/Kafka backends, also run: pip install -r requirements-ml.txt"
