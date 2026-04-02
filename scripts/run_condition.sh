#!/bin/bash
#SBATCH --job-name=gap1-cond
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/logs/gap1_cond_%x_%j.out
#SBATCH --error=/scratch/%u/logs/gap1_cond_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@northeastern.edu

# =============================================================================
# GAP 1 Single Condition Runner
# =============================================================================
# Usage:
#   sbatch --export=CONDITION=E,WANDB_ENTITY=your_entity scripts/run_condition.sh
#
# Or set CONDITION and WANDB_API_KEY in environment:
#   export CONDITION=E WANDB_API_KEY=xxx WANDB_ENTITY=yyy
#   sbatch scripts/run_condition.sh
#
# Valid conditions: A B C D1 D2 D3 E F
# Runtime per condition: ~8 GPU-hours on A100 (build_spec.md Section 13)

set -euo pipefail

CONDITION="${CONDITION:-E}"
WANDB_ENTITY="${WANDB_ENTITY:-BrutalCaeser}"

echo "=========================================="
echo "GAP 1 Training — Condition ${CONDITION}"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Entity:   ${WANDB_ENTITY}"
echo "Start:    $(date)"
echo "=========================================="

# Validate condition.
VALID_CONDITIONS="A B C D1 D2 D3 E F"
if ! echo "${VALID_CONDITIONS}" | grep -qw "${CONDITION}"; then
    echo "ERROR: Invalid condition '${CONDITION}'. Valid: ${VALID_CONDITIONS}"
    exit 1
fi

# ── Paths (all on scratch for I/O performance) ────────────────────────────────
FEATURE_DIR="/scratch/${USER}/data/ssv2_vjepa21_features"
OUTPUT_DIR="/scratch/${USER}/outputs/gap1"
PROJECT_DIR="${HOME}/gap1-experiment"   # cloned repo lives in $HOME
LOG_DIR="/scratch/${USER}/logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ── W&B: load API key from secure file if not already in environment ──────────
if [ -z "${WANDB_API_KEY:-}" ]; then
    WANDB_KEY_FILE="${HOME}/.wandb_api_key"
    if [ -f "${WANDB_KEY_FILE}" ]; then
        export WANDB_API_KEY="$(cat ${WANDB_KEY_FILE})"
        echo "[run] Loaded WANDB_API_KEY from ${WANDB_KEY_FILE}"
    else
        echo "[run] WARNING: WANDB_API_KEY not set and ${WANDB_KEY_FILE} not found."
        echo "[run]   Metrics will log to console only. Create ${WANDB_KEY_FILE} to enable W&B."
        echo "[run]   File format: single line containing your API key from wandb.ai/settings"
    fi
fi

# ── Activate environment ──────────────────────────────────────────────────────
module purge
module load cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gap1

cd "${PROJECT_DIR}"

# ── Verify features exist ─────────────────────────────────────────────────────
if [ ! -f "${FEATURE_DIR}/index.json" ]; then
    echo "ERROR: Feature index not found at ${FEATURE_DIR}/index.json"
    echo "       Run: sbatch scripts/preextract_ssv2.sh"
    exit 1
fi
echo "[run] Feature index found: $(wc -l < ${FEATURE_DIR}/index.json) bytes"

# ── Pre-run checks (unit tests + smoke test) ──────────────────────────────────
echo "[run] Running unit tests..."
python -m pytest tests/ -q --tb=short
echo "[run] Unit tests PASSED."

echo "[run] Running smoke test for Condition ${CONDITION}..."
python -m training.trainer \
    --config   configs/base.yaml \
    --override "configs/condition_${CONDITION}.yaml" \
    --smoke_test \
    --max_steps 100 \
    --no_wandb
echo "[run] Smoke test PASSED."

# ── Full training run ─────────────────────────────────────────────────────────
echo "[run] Starting full training — Condition ${CONDITION}..."
echo "[run] W&B project: gap1-sigreg-spatial | entity: ${WANDB_ENTITY}"

python -m training.trainer \
    --config        configs/base.yaml \
    --override      "configs/condition_${CONDITION}.yaml" \
    --wandb_project gap1-sigreg-spatial \
    --wandb_entity  "${WANDB_ENTITY}"

EXIT_CODE=$?
echo "=========================================="
echo "Condition ${CONDITION} finished (exit ${EXIT_CODE}): $(date)"
echo "Output dir: ${OUTPUT_DIR}/${CONDITION}/"
echo "=========================================="
exit ${EXIT_CODE}
