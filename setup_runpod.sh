#!/usr/bin/env bash
# RunPod Setup Script for drstrange-process
#
# Usage:
#   chmod +x setup_runpod.sh
#   ./setup_runpod.sh
#
# This sets up:
#   - System packages (apt update/upgrade, terminal tools, poppler, pandoc)
#   - uv (Python package manager)
#   - Python venv with all pipeline dependencies (docstrange, torch, etc.)
#   - Pre-downloads Nanonets-OCR2-3B model (~6 GB)
#
# After setup, activate the venv and run any script with plain `python`:
#   source .venv/bin/activate
#   python extract_toc.py BATES.pdf
#   python run_docstrange_textbook.py BATES.pdf results/textbook

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
OCR_MODEL="nanonets/Nanonets-OCR2-3B"

echo "============================================================"
echo "  RunPod Setup - drstrange-process"
echo "============================================================"
echo ""

# 1. System update and upgrade
echo "[setup] Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq 2>&1 | tail -3

# 2. Terminal tools + docstrange system deps
echo "[setup] Installing system dependencies..."
apt-get install -y -qq tmux btop nvtop poppler-utils pandoc 2>&1 | tail -3

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

# ============================================================
# 5. Create Python venv with all pipeline dependencies
# ============================================================
echo ""
echo "============================================================"
echo "  Creating Python environment"
echo "============================================================"

if [ -d "$VENV_DIR" ] && "$VENV_DIR/bin/python" -c "import docstrange; import torch" 2>/dev/null; then
    echo "[venv] Venv already exists and looks good"
else
    echo "[venv] Creating venv at $VENV_DIR"
    rm -rf "$VENV_DIR"
    uv venv "$VENV_DIR" --python 3.11

    # Install torch first (needed by docstrange and flash-attn)
    echo "[venv] Installing torch..."
    uv pip install --python "$VENV_DIR/bin/python" \
        "torch>=2.0.0" \
        "torchvision>=0.15.0"

    # Install docstrange + all pipeline dependencies
    # NOTE: Pin exact versions to avoid API breaking changes between releases.
    # Check what version works on your RTX 4000 pod with: pip show docstrange docling-ibm-models
    echo "[venv] Installing docstrange and pipeline dependencies..."
    uv pip install --python "$VENV_DIR/bin/python" \
        "docstrange==1.1.8" \
        "docling-ibm-models==3.11.0" \
        "pdf2image>=1.17.0" \
        "Pillow>=10.0.0" \
        "pymupdf>=1.24.0" \
        "huggingface_hub>=0.20.0"

    # Install Milvus ingestion dependencies
    echo "[venv] Installing Milvus ingestion dependencies..."
    uv pip install --python "$VENV_DIR/bin/python" \
        "pymilvus>=2.5.0" \
        "sentence-transformers>=3.0.0"

    # Note: flash-attn is NOT installed here because:
    # 1. Build often hangs/fails (requires compiling from source, 5-15 min)
    # 2. Even when manually installed, often doesn't work due to CUDA version mismatches
    # 3. The code in run_docstrange_textbook.py already falls back to SDPA which works fine
    # If you really want flash-attn for ~15% speedup, install manually after setup:
    #   pip install flash-attn --no-build-isolation

    # Verify installation
    echo "[venv] Verifying..."
    "$VENV_DIR/bin/python" -c "
import docstrange
print(f'  docstrange: OK')
import torch
print(f'  torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
import fitz
print(f'  pymupdf: {fitz.version}')
print('  flash-attn: NOT INSTALLED (using SDPA - ~15% slower but more stable)')
import pymilvus
print(f'  pymilvus: {pymilvus.__version__}')
import sentence_transformers
print(f'  sentence-transformers: {sentence_transformers.__version__}')
"
fi

# ============================================================
# 6. Pre-download Nanonets-OCR2-3B model (~6 GB)
# ============================================================
echo ""
echo "[setup] Pre-downloading OCR model: $OCR_MODEL"

"$VENV_DIR/bin/python" -c "
from huggingface_hub import snapshot_download
print('[setup]   Downloading ${OCR_MODEL} (this may take a few minutes)...')
path = snapshot_download('${OCR_MODEL}')
print(f'[setup]   Model cached at: {path}')
" && echo "[setup]   Model download complete." \
  || echo "[setup]   WARNING: Model download failed. It will be downloaded on first run."

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  Activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  Then run any script with python:"
echo "    python extract_toc.py BATES.pdf"
echo "    python run_docstrange_textbook.py BATES.pdf results/textbook"
echo "    python ingest_milvus.py results_book1/BATES.chunks.json"
echo ""
echo "  Pre-cached models:"
echo "    $OCR_MODEL (Nanonets OCR2, Qwen2.5-VL-3B fine-tune)"
echo ""
echo "  Terminal tools installed: tmux, btop, nvtop"
echo ""
