#!/bin/bash
#SBATCH --job-name=gap1-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/gap1_extract_%j.out
#SBATCH --error=/scratch/%u/logs/gap1_extract_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gupta.yashv@northeastern.edu

# =============================================================================
# GAP 1 Feature Extraction — V-JEPA 2.1 ViT-L patch tokens for SSv2
# =============================================================================
# Cluster: Northeastern Explorer HPC (explorer.northeastern.edu)
# GPU:     V100-SXM2 via 'gpu' partition (8h limit)
# SSv2 data: /datasets/something_v2/ (shared read-only, VideoFolder format)
# Checkpoint: /scratch/$USER/checkpoints/vjepa21_vitl.pt (~1.2 GB)
#
# VideoFolder format:
#   /datasets/something_v2/
#     category.txt                        ← 174 class names (0-indexed)
#     train_videofolder.txt               ← "rel_path n_frames label_idx" per line
#     val_videofolder.txt                 ← same format
#     20bn-something-something-v2/{id}/   ← JPEG frame directories
#
# Before running:
#   1. bash scripts/download_data.sh     (downloads V-JEPA 2.1 checkpoint)
#   2. sbatch scripts/preextract_ssv2.sh

set -euo pipefail

echo "=========================================="
echo "GAP 1 Feature Extraction"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:    $(date)"
echo "=========================================="

# ── Resolve username (USER may be unbound on some SLURM nodes) ───────────────
USER="${USER:-${SLURM_JOB_USER:-$(whoami)}}"
export USER

# ── Paths ────────────────────────────────────────────────────────────────────
# SSv2 is available read-only on Explorer at /datasets/something_v2/
SSV2_BASE="/datasets/something_v2"
VIDEO_DIR="${SSV2_BASE}"                                # parent of 20bn-something-something-v2/
LABEL_FILE="${SSV2_BASE}"                               # contains category.txt (passed as base_dir)
ANNOTATION_FILE="${SSV2_BASE}/train_videofolder.txt"    # 168,912 training clips
OUTPUT_DIR="/scratch/${USER}/data/ssv2_vjepa21_features"
CHECKPOINT="/scratch/${USER}/checkpoints/vjepa21_vitl.pt"   # V-JEPA 2.1 ViT-L, ~1.2 GB
N_CLIPS=20000
ANNOTATION_FORMAT="videofolder"

# ── Validate prerequisites ────────────────────────────────────────────────────
for f in "${ANNOTATION_FILE}" "${CHECKPOINT}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: Required file not found: ${f}"
        if [ "${f}" = "${CHECKPOINT}" ]; then
            echo "       Run: bash scripts/download_data.sh"
        else
            echo "       SSv2 data expected at /datasets/something_v2/ on Explorer HPC"
        fi
        exit 1
    fi
done

if [ ! -d "${SSV2_BASE}/20bn-something-something-v2" ]; then
    echo "ERROR: SSv2 video directory not found: ${SSV2_BASE}/20bn-something-something-v2"
    echo "       Ensure you are on explorer.northeastern.edu with /datasets/ mounted"
    exit 1
fi

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
echo "[preextract] SSv2 base:    ${SSV2_BASE}"
echo "[preextract] Checkpoint:   ${CHECKPOINT}"
echo "[preextract] Output dir:   ${OUTPUT_DIR}"
python data/preextract_ssv2.py \
    --video_dir          "${VIDEO_DIR}" \
    --label_file         "${LABEL_FILE}" \
    --annotation_file    "${ANNOTATION_FILE}" \
    --output_dir         "${OUTPUT_DIR}" \
    --checkpoint         "${CHECKPOINT}" \
    --n_clips            "${N_CLIPS}" \
    --annotation_format  "${ANNOTATION_FORMAT}" \
    --device             cuda \
    --train_frac         0.8 \
    --val_frac           0.1 \
    --seed               42

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
