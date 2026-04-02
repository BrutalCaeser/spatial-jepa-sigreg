#!/bin/bash
# =============================================================================
# GAP 1 Data Acquisition — V-JEPA 2.1 ViT-L + SSv2 (Something-Something v2)
# =============================================================================
# Run from the Explorer HPC login node:
#   bash scripts/download_data.sh
#
# What this does:
#   1. Clones facebookresearch/vjepa2 and installs the package
#   2. Downloads V-JEPA 2.1 ViT-L checkpoint (~4.8 GB) from Meta's release
#   3. Provides SSv2 download instructions (manual registration required)
#   4. Verifies checkpoint integrity (ema_encoder key present)
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
    # Patch Python version requirement (repo requires >=3.11, cluster has 3.10)
    sed -i 's/python_requires=">=3.11"/python_requires=">=3.10"/' "${VJEPA_REPO}/setup.py" 2>/dev/null || true
    # Patch dev artifact: VJEPA_BASE_URL pointing to localhost
    sed -i 's|http://localhost:8300|https://dl.fbaipublicfiles.com/vjepa2|g' \
        "${VJEPA_REPO}/app/vjepa_2_1/models/vision_transformer.py" 2>/dev/null || true
fi

# Install the vjepa package into gap1 env
echo "[data] Installing vjepa package..."
pip install -e "${VJEPA_REPO}" --no-deps -q
echo "[data] vjepa package installed."

# ── Step 2: V-JEPA 2.1 ViT-L checkpoint ─────────────────────────────────────
echo ""
echo "[data] Step 2: Downloading V-JEPA 2.1 ViT-L checkpoint..."
echo "[data]   Model: vjepa2_1_vit_large_384 (embed_dim=1024, N=196 @ img_size=224)"
echo "[data]   Target: ${CHECKPOINT_DIR}/vjepa21_vitl.pt"
echo "[data]   URL: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"

VJEPA_CKPT="${CHECKPOINT_DIR}/vjepa21_vitl.pt"

if [ -f "${VJEPA_CKPT}" ] && [ "$(stat -c%s "${VJEPA_CKPT}" 2>/dev/null || echo 0)" -gt 100000000 ]; then
    echo "[data] Checkpoint already exists: $(du -sh "${VJEPA_CKPT}" | cut -f1)"
else
    echo "[data] Downloading (~4.8 GB, may take 5-15 minutes)..."
    wget -q --tries=3 --timeout=600 \
        -O "${VJEPA_CKPT}" \
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
    echo "[data] Downloaded: $(du -sh "${VJEPA_CKPT}" | cut -f1)"
fi

# Verify checkpoint
CKPT_PATH="${VJEPA_CKPT}" python - << 'PYEOF'
import os, torch
ckpt_path = os.environ["CKPT_PATH"]
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
keys = list(ckpt.keys()) if isinstance(ckpt, dict) else [str(type(ckpt))]
print(f"[data] Checkpoint keys: {keys}")
assert "ema_encoder" in keys, f"ema_encoder key missing! Got: {keys}"
print("[data] V-JEPA 2.1 ViT-L checkpoint verified.")
PYEOF

# ── Step 3: SSv2 download from HuggingFace ───────────────────────────────────
echo ""
echo "[data] Step 3: SSv2 (Something-Something v2)"
echo "[data]   Source: HuggingFaceM4/something_something_v2"
echo "[data]   License: Qualcomm research license (non-commercial)"
echo ""

# Install datasets library if not present
python -c "import datasets" 2>/dev/null || pip install datasets -q

# Check if SSv2 is already available
if [ -f "${DATA_DIR}/labels.json" ] && [ -f "${DATA_DIR}/annotations.json" ]; then
    N_VIDEOS=$(find "${DATA_DIR}/videos" -name "*.webm" -o -name "*.mp4" 2>/dev/null | wc -l)
    echo "[data] SSv2 metadata found. Videos: ${N_VIDEOS}"
    if [ "${N_VIDEOS}" -gt 15000 ]; then
        echo "[data] ✓ SSv2 ready. Proceed with: sbatch scripts/preextract_ssv2.sh"
    else
        echo "[data] Resuming download (${N_VIDEOS}/20000 clips so far)..."
        python scripts/download_ssv2_hf.py \
            --output_dir "${DATA_DIR}" \
            --split train \
            --n_clips 20000 \
            --hf_cache "/scratch/${USER}/hf_cache" \
            --seed 42
    fi
else
    echo "[data] Downloading SSv2 from HuggingFace (20K clips, ~15-30 min)..."
    echo "[data] Note: First run requires agreeing to Qualcomm license at:"
    echo "[data]   https://huggingface.co/datasets/HuggingFaceM4/something_something_v2"
    echo "[data] If authentication fails, run: huggingface-cli login"
    echo ""
    python scripts/download_ssv2_hf.py \
        --output_dir "${DATA_DIR}" \
        --split train \
        --n_clips 20000 \
        --hf_cache "/scratch/${USER}/hf_cache" \
        --seed 42
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "[data] Summary"
echo ""
echo "V-JEPA 2.1 checkpoint:"
if ls -lh "${CHECKPOINT_DIR}/vjepa21_vitl.pt" 2>/dev/null; then
    echo "  ✓ Present"
else
    echo "  ✗ Missing — see Step 2 above"
fi
echo ""
echo "SSv2 labels:      $(ls "${DATA_DIR}/labels.json" 2>/dev/null && echo '✓' || echo '✗ Missing')"
echo "SSv2 annotations: $(ls "${DATA_DIR}/annotations.json" 2>/dev/null && echo '✓' || echo '✗ Missing')"
echo "SSv2 videos:      $(find "${DATA_DIR}/videos" -type f 2>/dev/null | wc -l) files"
echo ""
echo "Next: sbatch scripts/preextract_ssv2.sh"
echo "=========================================="
