#!/bin/bash
#SBATCH --job-name=gap1-cond-${CONDITION:-X}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/logs/gap1_cond_${CONDITION:-X}_%j.out
#SBATCH --error=/scratch/%u/logs/gap1_cond_${CONDITION:-X}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@discovery.usc.edu

# =============================================================================
# GAP 1 Single Condition Runner
# =============================================================================
# Usage:
#   export CONDITION=E  # Set condition before sbatch
#   sbatch scripts/run_condition.sh
#
# Or with --export:
#   sbatch --export=CONDITION=E scripts/run_condition.sh
#
# Valid conditions: A, B, C, D1, D2, D3, E, F
# Expected runtime per condition: 8 GPU-hours on A100 (build_spec.md Section 13)
#
# All conditions MUST pass unit tests before submission:
#   python -m pytest tests/ -v --timeout=120
#
# Smoke test before full run:
#   python -m training.trainer --config configs/base.yaml \
#          --override configs/condition_${CONDITION}.yaml \
#          --smoke_test --max_steps 100

set -euo pipefail

CONDITION="${CONDITION:-E}"   # Default to E if not set

echo "=========================================="
echo "GAP 1 Training — Condition ${CONDITION}"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:    $(date)"
echo "=========================================="

# Validate condition.
VALID_CONDITIONS="A B C D1 D2 D3 E F"
if ! echo "${VALID_CONDITIONS}" | grep -qw "${CONDITION}"; then
    echo "ERROR: Invalid condition '${CONDITION}'. Valid: ${VALID_CONDITIONS}"
    exit 1
fi

# ── User configuration ────────────────────────────────────────────────────────
FEATURE_DIR="/scratch/${USER}/data/ssv2_vjepa21_features"
OUTPUT_DIR="/scratch/${USER}/outputs/gap1"
WANDB_ENTITY="YOUR_WANDB_ENTITY"      # <-- Set your W&B entity here
PROJECT_DIR="/scratch/${USER}/gap1-experiment"
# ─────────────────────────────────────────────────────────────────────────────

# Activate environment.
module load cuda/12.1
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate gap1

cd "${PROJECT_DIR}"
mkdir -p "/scratch/${USER}/logs"

# Verify features exist.
if [ ! -f "${FEATURE_DIR}/index.json" ]; then
    echo "ERROR: Feature index not found at ${FEATURE_DIR}/index.json"
    echo "       Run scripts/preextract_ssv2.sh first."
    exit 1
fi

# ── Pre-run checks ────────────────────────────────────────────────────────────
echo "[run] Running unit tests..."
python -m pytest tests/ -v --timeout=120 -q
if [ $? -ne 0 ]; then
    echo "ERROR: Unit tests FAILED. Do not submit job."
    exit 1
fi
echo "[run] Unit tests PASSED."

echo "[run] Running smoke test for Condition ${CONDITION}..."
python -m training.trainer \
    --config   configs/base.yaml \
    --override "configs/condition_${CONDITION}.yaml" \
    --smoke_test \
    --max_steps 100
if [ $? -ne 0 ]; then
    echo "ERROR: Smoke test FAILED for Condition ${CONDITION}."
    exit 1
fi
echo "[run] Smoke test PASSED."

# ── Full training run ─────────────────────────────────────────────────────────
echo "[run] Starting full training for Condition ${CONDITION}..."

python -m training.trainer \
    --config         configs/base.yaml \
    --override       "configs/condition_${CONDITION}.yaml" \
    --wandb_project  gap1-sigreg-spatial \
    --wandb_entity   "${WANDB_ENTITY}"

echo "=========================================="
echo "Condition ${CONDITION} training complete: $(date)"
echo "=========================================="
