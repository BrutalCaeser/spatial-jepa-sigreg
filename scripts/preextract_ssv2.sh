#!/bin/bash
#SBATCH --job-name=gap1-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/%u/logs/gap1_extract_%j.out
#SBATCH --error=/scratch/%u/logs/gap1_extract_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@discovery.usc.edu

# =============================================================================
# GAP 1 Feature Extraction — V-JEPA 2.1 ViT-G patch tokens for SSv2
# =============================================================================
# Expected runtime: 6-8 hours on A100 for 20K clips (build_spec.md Week 1 Day 3)
# Storage output: ~30 GB of .pt feature files
#
# Before running:
#   1. Download V-JEPA 2.1 ViT-G checkpoint (Meta public release)
#   2. Register for SSv2 and download part-00 annotations + videos
#   3. Set paths below
#   4. sbatch scripts/preextract_ssv2.sh

set -euo pipefail

echo "=========================================="
echo "GAP 1 Feature Extraction"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

# ── User configuration (edit these) ──────────────────────────────────────────
VIDEO_DIR="/scratch/${USER}/data/ssv2/videos/part-00"
LABEL_FILE="/scratch/${USER}/data/ssv2/something-something-v2-labels.json"
ANNOTATION_FILE="/scratch/${USER}/data/ssv2/something-something-v2-train.json"
OUTPUT_DIR="/scratch/${USER}/data/ssv2_vjepa21_features"
CHECKPOINT="/scratch/${USER}/checkpoints/vjepa21_vitg.pt"
N_CLIPS=20000
# ─────────────────────────────────────────────────────────────────────────────

# Activate conda/venv environment.
module load cuda/12.1
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate gap1

# Create output directory.
mkdir -p "${OUTPUT_DIR}"
mkdir -p "/scratch/${USER}/logs"

# Run extraction.
cd /scratch/${USER}/gap1-experiment

echo "[preextract] Starting feature extraction..."
python data/preextract_ssv2.py \
    --video_dir    "${VIDEO_DIR}" \
    --label_file   "${LABEL_FILE}" \
    --annotation_file "${ANNOTATION_FILE}" \
    --output_dir   "${OUTPUT_DIR}" \
    --checkpoint   "${CHECKPOINT}" \
    --n_clips      "${N_CLIPS}" \
    --device       cuda \
    --train_frac   0.8 \
    --val_frac     0.1 \
    --seed         42

echo "[preextract] Extraction complete: $(date)"
echo "[preextract] Output directory contents: $(ls ${OUTPUT_DIR} | wc -l) files"

# Verify baseline quality (FM5 check).
echo "[preextract] Running baseline verification..."
python scripts/verify_baseline.py \
    --feature_dir "${OUTPUT_DIR}" \
    --split val \
    --n_samples 64

echo "=========================================="
echo "Feature extraction finished: $(date)"
echo "=========================================="
