#!/bin/bash
#SBATCH --job-name=gap1-lamsweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/%u/logs/gap1_lamsweep_%x_%j.out
#SBATCH --error=/scratch/%u/logs/gap1_lamsweep_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gupta.yashv@northeastern.edu

# =============================================================================
# Lambda Sweep — 2000 steps each, tests lambda_1 ∈ {10, 25, 50}
# =============================================================================
# Usage:
#   sbatch --export=LAMBDA_VAL=10 scripts/run_lambda_sweep.sh
#   sbatch --export=LAMBDA_VAL=25 scripts/run_lambda_sweep.sh
#   sbatch --export=LAMBDA_VAL=50 scripts/run_lambda_sweep.sh

set -euo pipefail

LAMBDA_VAL="${LAMBDA_VAL:-10}"
WANDB_ENTITY="${WANDB_ENTITY:-brutalcaesar-northeastern-university}"

echo "=========================================="
echo "Lambda Sweep — lambda_1 = ${LAMBDA_VAL}"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:    $(date)"
echo "=========================================="

# ── Resolve username and home ─────────────────────────────────────────────────
USER="${USER:-${SLURM_JOB_USER:-$(whoami)}}"
export USER
HOME="${HOME:-/home/${USER}}"
export HOME

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURE_DIR="/scratch/${USER}/data/ssv2_vjepa21_features"
PROJECT_DIR="/scratch/${USER}/gap1-experiment"
LOG_DIR="/scratch/${USER}/logs"
mkdir -p "${LOG_DIR}"

# ── W&B offline ───────────────────────────────────────────────────────────────
export WANDB_MODE="${WANDB_MODE:-offline}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    WANDB_KEY_FILE="${HOME}/.wandb_api_key"
    [ -f "${WANDB_KEY_FILE}" ] && export WANDB_API_KEY="$(cat ${WANDB_KEY_FILE})"
fi

# ── Environment ───────────────────────────────────────────────────────────────
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
    echo "[sweep] WARNING: module system not available. Using conda CUDA runtime."
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

# ── Run sweep job — 2000 steps, eval every 200 steps for finer tracking ──────
echo "[sweep] Running lambda_1=${LAMBDA_VAL}, 2000 steps..."

python -m training.trainer \
    --config        configs/base.yaml \
    --override      "configs/sweep_lambda${LAMBDA_VAL}.yaml" \
    --max_steps     2000 \
    --wandb_project gap1-sigreg-spatial \
    --wandb_entity  "${WANDB_ENTITY}" \
    --no_wandb

EXIT_CODE=$?
echo "=========================================="
echo "Lambda sweep (lambda_1=${LAMBDA_VAL}) finished (exit ${EXIT_CODE}): $(date)"
echo "=========================================="
exit ${EXIT_CODE}
