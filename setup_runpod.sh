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
    UV_INSTALLER=$(mktemp)
    curl -LsSf https://astral.sh/uv/install.sh -o "$UV_INSTALLER"
    sh "$UV_INSTALLER"
    rm -f "$UV_INSTALLER"
    export PATH="$HOME/.local/bin:$PATH"
    # Persist PATH for future shell sessions
    if ! grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
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

# Cache deps for all scripts with PEP 723 inline metadata
for script in extract_toc.py; do
    if [ -f "$script" ]; then
        echo "[setup]   Caching deps for $script..."
        uv run "$script" --help > /dev/null 2>&1 || true
    fi
done
# Note: run_docstrange_textbook.py requires a pre-built venv with docstrange,
# torch, and transformers.  It cannot be run via 'uv run' standalone.
echo "[setup]   Skipping run_docstrange_textbook.py (requires GPU venv with docstrange)"

# 6. Pre-download Nanonets-OCR2-3B model (~6 GB)
# This avoids a long first-run download when processing PDFs.
OCR_MODEL="nanonets/Nanonets-OCR2-3B"
echo ""
echo "[setup] Pre-downloading OCR model: $OCR_MODEL"

if python3 -c "from huggingface_hub import snapshot_download" 2>/dev/null; then
    python3 -c "
from huggingface_hub import snapshot_download
print('[setup]   Downloading ${OCR_MODEL} (this may take a few minutes)...')
path = snapshot_download('${OCR_MODEL}')
print(f'[setup]   Model cached at: {path}')
" && echo "[setup]   Model download complete." \
  || echo "[setup]   WARNING: Model download failed. It will be downloaded on first run."
else
    echo "[setup]   huggingface_hub not installed, trying pip install..."
    pip install -q huggingface_hub 2>/dev/null && \
    python3 -c "
from huggingface_hub import snapshot_download
print('[setup]   Downloading ${OCR_MODEL} (this may take a few minutes)...')
path = snapshot_download('${OCR_MODEL}')
print(f'[setup]   Model cached at: {path}')
" && echo "[setup]   Model download complete." \
  || echo "[setup]   WARNING: Model download failed. It will be downloaded on first run."
fi

# 7. Pre-download flash-attn if GPU is available (for faster inference)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "[setup] Installing flash-attn for faster inference..."
    pip install -q flash-attn --no-build-isolation 2>&1 | tail -3 \
        && echo "[setup]   flash-attn installed." \
        || echo "[setup]   WARNING: flash-attn install failed. Will fall back to SDPA attention."
fi

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  Available scripts:"
echo "    uv run extract_toc.py <pdf>     - Extract TOC from PDF"
echo ""
echo "  Pre-cached models:"
echo "    $OCR_MODEL (Nanonets OCR2, Qwen2.5-VL-3B fine-tune)"
echo ""
echo "  Terminal tools installed: tmux, btop, nvtop"
echo ""
