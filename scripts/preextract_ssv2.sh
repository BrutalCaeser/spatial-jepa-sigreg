#!/bin/bash
#SBATCH --job-name=gap1-extract
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/%u/logs/gap1_extract_%j.out
#SBATCH --error=/scratch/%u/logs/gap1_extract_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gupta.yashv@northeastern.edu

# =============================================================================
# GAP 1 Feature Extraction — V-JEPA 2.1 ViT-G patch tokens for SSv2
# =============================================================================
# Cluster: Northeastern Explorer HPC (explorer.neu.edu)
# GPU:     L40 48GB via 'sharing' partition (max 1h per job)
# For 20K clips (~30GB output) this will need multiple 1h job segments.
#
# Before running:
#   1. bash scripts/download_data.sh     (downloads V-JEPA 2.1 + SSv2)
#   2. sbatch scripts/preextract_ssv2.sh
#
# Data paths expected:
#   /scratch/$USER/data/ssv2/videos/            ← SSv2 video files
#   /scratch/$USER/data/ssv2/labels.json        ← class label map
#   /scratch/$USER/data/ssv2/annotations.json   ← clip annotations
#   /scratch/$USER/checkpoints/vjepa21_vitg.pt  ← V-JEPA 2.1 ViT-G

set -euo pipefail

echo "=========================================="
echo "GAP 1 Feature Extraction"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:    $(date)"
echo "=========================================="

# ── Paths ────────────────────────────────────────────────────────────────────
VIDEO_DIR="/scratch/${USER}/data/ssv2/videos"
LABEL_FILE="/scratch/${USER}/data/ssv2/labels.json"
ANNOTATION_FILE="/scratch/${USER}/data/ssv2/annotations.json"
OUTPUT_DIR="/scratch/${USER}/data/ssv2_vjepa21_features"
CHECKPOINT="/scratch/${USER}/checkpoints/vjepa21_vitl.pt"   # V-JEPA 2.1 ViT-L, 4.8GB
N_CLIPS=20000

# ── Validate prerequisites ────────────────────────────────────────────────────
for f in "${LABEL_FILE}" "${ANNOTATION_FILE}" "${CHECKPOINT}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: Required file not found: ${f}"
        echo "       Run: bash scripts/download_data.sh"
        exit 1
    fi
done

# ── Activate conda environment ────────────────────────────────────────────────
module purge
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null
CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
[ -d "${CONDA_BASE}" ] || CONDA_BASE="/shared/centos7/anaconda3/2021.05"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gap1

echo "[preextract] Python: $(python --version)"
echo "[preextract] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}" "/scratch/${USER}/logs"

# ── Run extraction ────────────────────────────────────────────────────────────
cd /scratch/${USER}/gap1-experiment

echo "[preextract] Starting extraction of ${N_CLIPS} clips..."
python data/preextract_ssv2.py \
    --video_dir       "${VIDEO_DIR}" \
    --label_file      "${LABEL_FILE}" \
    --annotation_file "${ANNOTATION_FILE}" \
    --output_dir      "${OUTPUT_DIR}" \
    --checkpoint      "${CHECKPOINT}" \
    --n_clips         "${N_CLIPS}" \
    --device          cuda \
    --train_frac      0.8 \
    --val_frac        0.1 \
    --seed            42

echo "[preextract] Extraction complete: $(date)"
echo "[preextract] Feature files: $(ls ${OUTPUT_DIR}/*.pt 2>/dev/null | wc -l)"

# ── Verify baseline quality (FM5 check) ──────────────────────────────────────
echo "[preextract] Running FM5 baseline verification..."
python scripts/verify_baseline.py \
    --feature_dir "${OUTPUT_DIR}" \
    --split       val \
    --n_samples   64

echo "=========================================="
echo "Feature extraction finished: $(date)"
echo "=========================================="
