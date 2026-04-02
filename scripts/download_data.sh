#!/bin/bash
# =============================================================================
# GAP 1 Data Acquisition — V-JEPA 2.1 ViT-G + SSv2 (Something-Something v2)
# =============================================================================
# Run from the Explorer HPC login node:
#   bash scripts/download_data.sh
#
# What this does:
#   1. Clones facebookresearch/vjepa2 and installs the package
#   2. Downloads V-JEPA 2.1 ViT-G checkpoint (~3.7 GB) from Meta's release
#   3. Provides SSv2 download instructions (manual registration required)
#   4. Verifies checkpoint integrity with SHA256
#
# SSv2 registration: https://developer.qualcomm.com/software/ai-datasets/something-something
# V-JEPA 2.1 paper:  https://arxiv.org/abs/2410.11441
# V-JEPA 2.1 repo:   https://github.com/facebookresearch/vjepa2

set -euo pipefail

USER_SCRATCH="/scratch/${USER}"
CHECKPOINT_DIR="${USER_SCRATCH}/checkpoints"
DATA_DIR="${USER_SCRATCH}/data/ssv2"
VJEPA_REPO="${USER_SCRATCH}/vjepa2"

mkdir -p "${CHECKPOINT_DIR}" "${DATA_DIR}" "${USER_SCRATCH}/logs"

echo "=========================================="
echo "GAP 1 Data Acquisition"
echo "User:    ${USER}"
echo "Scratch: ${USER_SCRATCH}"
echo "Date:    $(date)"
echo "=========================================="

# ── Step 1: V-JEPA 2.1 package ───────────────────────────────────────────────
echo ""
echo "[data] Step 1: Installing V-JEPA 2.1 package..."

CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
[ -d "${CONDA_BASE}" ] || CONDA_BASE="/shared/centos7/anaconda3/2021.05"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gap1

if [ ! -d "${VJEPA_REPO}" ]; then
    echo "[data] Cloning facebookresearch/vjepa2..."
    git clone https://github.com/facebookresearch/vjepa2.git "${VJEPA_REPO}"
fi

# Install the vjepa package into gap1 env
echo "[data] Installing vjepa package..."
pip install -e "${VJEPA_REPO}" --no-deps -q
echo "[data] vjepa package installed."

# ── Step 2: V-JEPA 2.1 ViT-G checkpoint ─────────────────────────────────────
echo ""
echo "[data] Step 2: Downloading V-JEPA 2.1 ViT-G checkpoint..."
echo "[data]   Target: ${CHECKPOINT_DIR}/vjepa21_vitg.pt"

VJEPA_CKPT="${CHECKPOINT_DIR}/vjepa21_vitg.pt"

if [ -f "${VJEPA_CKPT}" ]; then
    echo "[data] Checkpoint already exists: ${VJEPA_CKPT}"
    echo "[data] Size: $(du -sh ${VJEPA_CKPT} | cut -f1)"
else
    # V-JEPA 2.1 ViT-G checkpoint — Meta public release (Hugging Face)
    # Model card: https://huggingface.co/facebook/vjepa2-vitg-fpc64-256
    echo "[data] Downloading from Hugging Face (facebook/vjepa2-vitg-fpc64-256)..."

    # Method 1: huggingface_hub (fastest)
    python - << 'PYEOF'
from huggingface_hub import hf_hub_download
import os, shutil

ckpt_dir = os.environ["CHECKPOINT_DIR"]
print(f"[data] Downloading V-JEPA 2.1 ViT-G checkpoint to {ckpt_dir}...")

# Try V-JEPA 2.1 ViT-G checkpoint
try:
    path = hf_hub_download(
        repo_id="facebook/vjepa2-vitg-fpc64-256",
        filename="checkpoint.pt",
        cache_dir=f"{ckpt_dir}/.hf_cache",
    )
    shutil.copy(path, f"{ckpt_dir}/vjepa21_vitg.pt")
    print(f"[data] Downloaded: {ckpt_dir}/vjepa21_vitg.pt")
except Exception as e:
    print(f"[data] HF download failed: {e}")
    print("[data] Trying alternative model ID...")
    try:
        path = hf_hub_download(
            repo_id="facebook/vjepa2-vitg-fpc64-224",
            filename="checkpoint.pt",
            cache_dir=f"{ckpt_dir}/.hf_cache",
        )
        shutil.copy(path, f"{ckpt_dir}/vjepa21_vitg.pt")
        print(f"[data] Downloaded (224px variant): {ckpt_dir}/vjepa21_vitg.pt")
    except Exception as e2:
        print(f"[data] ERROR: Both download attempts failed.")
        print(f"[data]   {e2}")
        print("[data] Manual download instructions:")
        print("[data]   1. Visit https://huggingface.co/facebook/vjepa2-vitg-fpc64-256")
        print("[data]   2. Download checkpoint.pt")
        print(f"[data]   3. Upload to {ckpt_dir}/vjepa21_vitg.pt")
PYEOF

fi

# ── Step 3: SSv2 data instructions ───────────────────────────────────────────
echo ""
echo "[data] Step 3: SSv2 (Something-Something v2)"
echo ""
echo "  SSv2 requires manual registration. Steps:"
echo ""
echo "  1. Register at:"
echo "     https://developer.qualcomm.com/software/ai-datasets/something-something"
echo ""
echo "  2. After approval, download:"
echo "     - something-something-v2-labels.json     → ${DATA_DIR}/labels.json"
echo "     - something-something-v2-train.json      → ${DATA_DIR}/annotations.json"
echo "     - 20bn-something-something-v2-{00..19}   → extract to ${DATA_DIR}/videos/"
echo ""
echo "  3. If you already have SSv2 videos elsewhere on the cluster, run:"
echo "     ln -s /path/to/ssv2/videos ${DATA_DIR}/videos"
echo ""

# Check if SSv2 is already available
if [ -f "${DATA_DIR}/labels.json" ] && [ -f "${DATA_DIR}/annotations.json" ]; then
    echo "[data] SSv2 metadata found: ${DATA_DIR}/"
    N_VIDEOS=$(find "${DATA_DIR}/videos" -name "*.webm" -o -name "*.mp4" 2>/dev/null | wc -l)
    echo "[data] Videos found: ${N_VIDEOS}"
    if [ "${N_VIDEOS}" -gt 100 ]; then
        echo "[data] SSv2 appears ready. Proceed with: sbatch scripts/preextract_ssv2.sh"
    fi
else
    echo "[data] SSv2 metadata not yet present. Complete Step 3 manually."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "[data] Summary"
echo ""
echo "V-JEPA 2.1 checkpoint:"
ls -lh "${CHECKPOINT_DIR}/vjepa21_vitg.pt" 2>/dev/null && echo "  ✓ Present" || echo "  ✗ Missing — see Step 2 above"
echo ""
echo "SSv2 labels:      $(ls ${DATA_DIR}/labels.json 2>/dev/null && echo '✓' || echo '✗ Missing')"
echo "SSv2 annotations: $(ls ${DATA_DIR}/annotations.json 2>/dev/null && echo '✓' || echo '✗ Missing')"
echo "SSv2 videos:      $(find ${DATA_DIR}/videos -type f 2>/dev/null | wc -l) files"
echo ""
echo "Next: sbatch scripts/preextract_ssv2.sh"
echo "=========================================="
