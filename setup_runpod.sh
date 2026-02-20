#!/usr/bin/env bash
# RunPod Setup Script for drstrange-process
#
# Usage:
#   chmod +x setup_runpod.sh
#   ./setup_runpod.sh
#
# This sets up:
#   - uv (Python package manager with PEP 723 support)
#   - Basic terminal tools (tmux, btop, nvtop)
#   - Pre-caches Python dependencies for all scripts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  RunPod Setup - drstrange-process"
echo "============================================================"
echo ""

# 1. System update and upgrade
echo "[setup] Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq 2>&1 | tail -3

# 2. Terminal tools
echo "[setup] Installing terminal tools..."
apt-get install -y -qq tmux btop nvtop 2>&1 | tail -3

# 3. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[setup] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "[setup] uv installed: $(uv --version)"
else
    echo "[setup] uv already installed: $(uv --version)"
fi

# 4. System info
echo ""
echo "[setup] System Info:"
echo "  Python: $(python3 --version 2>&1)"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
fi

# 5. Pre-cache Python dependencies for each script
echo ""
echo "[setup] Pre-caching Python dependencies..."

for script in extract_toc.py; do
    if [ -f "$script" ]; then
        echo "[setup]   Caching deps for $script..."
        uv run "$script" --help > /dev/null 2>&1 || true
    fi
done

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  Available scripts:"
echo "    uv run extract_toc.py <pdf>     - Extract TOC from PDF"
echo ""
echo "  Terminal tools installed: tmux, btop, nvtop"
echo ""
