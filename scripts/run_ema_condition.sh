#!/bin/bash
#SBATCH --job-name=gap1-ema
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/phase3_ema/%x_%j.out
#SBATCH --error=/scratch/%u/logs/phase3_ema/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gupta.yashv@northeastern.edu

# =============================================================================
# GAP 1 Phase 3 — EMA Target Adapter Conditions
# =============================================================================
# Usage:
#   sbatch --job-name=gap1-B-ema --export=CONDITION=B_ema scripts/run_ema_condition.sh
#   sbatch --job-name=gap1-E-ema --export=CONDITION=E_ema scripts/run_ema_condition.sh
#
# Valid conditions: B_ema C_ema D1_ema D2_ema D3_ema E_ema F_ema

set -euo pipefail

CONDITION="${CONDITION:-B_ema}"

echo "=========================================="
echo "GAP 1 Phase 3 (EMA) — Condition ${CONDITION}"
echo "Job: ${SLURM_JOB_ID} Node: ${SLURMD_NODENAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "=========================================="

# ── Resolve username and home ────────────────────────────────────────────────
USER="${USER:-${SLURM_JOB_USER:-$(whoami)}}"
export USER
HOME="${HOME:-/home/${USER}}"
export HOME

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURE_DIR="/scratch/${USER}/data/ssv2_vjepa21_features"
PROJECT_DIR="/scratch/${USER}/gap1-experiment"
LOG_DIR="/scratch/${USER}/logs/phase3_ema"

mkdir -p "${LOG_DIR}"

# ── W&B offline ──────────────────────────────────────────────────────────────
export WANDB_MODE="${WANDB_MODE:-offline}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    WANDB_KEY_FILE="${HOME}/.wandb_api_key"
    [ -f "${WANDB_KEY_FILE}" ] && export WANDB_API_KEY="$(cat ${WANDB_KEY_FILE})"
fi

# ── Activate environment ─────────────────────────────────────────────────────
if ! command -v module &>/dev/null; then
    for _mod_init in \
        /usr/share/Modules/init/bash \
        /etc/profile.d/modules.sh \
        /shared/EL9/explorer/Modules/init/bash \
        /shared/centos7/Modules/init/bash; do
        [ -f "${_mod_init}" ] && source "${_mod_init}" && break
    done
fi
if command -v module &>/dev/null; then
    module purge
    module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true
else
    echo "[run] WARNING: module system not available. Using conda CUDA runtime."
fi
CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
[ -d "${CONDA_BASE}" ] || CONDA_BASE="/shared/centos7/anaconda3/2021.05"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gap1

cd "${PROJECT_DIR}"

# ── Verify features ──────────────────────────────────────────────────────────
if [ ! -f "${FEATURE_DIR}/index.json" ]; then
    echo "ERROR: Feature index not found at ${FEATURE_DIR}/index.json"
    exit 1
fi

# ── Training ─────────────────────────────────────────────────────────────────
echo "[run] Starting training — Condition ${CONDITION}..."

RESUME_ARG=""
if [ -n "${RESUME_CKPT:-}" ]; then
    echo "[run] Resuming from: ${RESUME_CKPT}"
    RESUME_ARG="--resume ${RESUME_CKPT}"
fi

python -m training.trainer \
    --config        configs/base.yaml \
    --override      "configs/condition_${CONDITION}.yaml" \
    --wandb_project gap1-sigreg-spatial \
    --wandb_entity  "brutalcaesar-northeastern-university" \
    ${RESUME_ARG}

EXIT_CODE=$?
echo "=========================================="
echo "Condition ${CONDITION} finished (exit ${EXIT_CODE}): $(date)"
echo "=========================================="
exit ${EXIT_CODE}
