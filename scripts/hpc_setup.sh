#!/bin/bash
# =============================================================================
# HPC Environment Setup — Run ONCE after cloning the repo on Explorer
# =============================================================================
# Usage (on the HPC login node):
#   bash scripts/hpc_setup.sh
#
# What this does:
#   1. Loads CUDA module
#   2. Creates conda environment 'gap1' with all dependencies
#   3. Validates W&B login via WANDB_API_KEY
#   4. Creates required scratch directories
#   5. Runs a CPU smoke test to verify the environment is correct
#
# Prerequisites:
#   - Conda/mamba available on the cluster
#   - WANDB_API_KEY exported in your ~/.bashrc or passed directly:
#       WANDB_API_KEY=xxx bash scripts/hpc_setup.sh

set -euo pipefail

CONDA_ENV="gap1"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=========================================="
echo "GAP 1 — HPC Environment Setup"
echo "Repo:   ${REPO_DIR}"
echo "Conda:  ${CONDA_ENV}"
echo "User:   $(whoami)"
echo "Host:   $(hostname)"
echo "Date:   $(date)"
echo "=========================================="

# ── 1. Load modules ──────────────────────────────────────────────────────────
echo "[setup] Loading modules..."
module purge
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "[setup] WARNING: No CUDA module found — will use CPU"
module list 2>&1 | grep -i cuda || true

# ── 2. Create conda environment ──────────────────────────────────────────────
echo "[setup] Creating conda environment '${CONDA_ENV}'..."

# Initialize conda for this shell.
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "[setup] Environment '${CONDA_ENV}' already exists. Updating..."
    conda activate "${CONDA_ENV}"
else
    echo "[setup] Creating new environment '${CONDA_ENV}' (Python 3.11)..."
    conda create -n "${CONDA_ENV}" python=3.11 -y
    conda activate "${CONDA_ENV}"
fi

# Install PyTorch with CUDA (adjust CUDA version if needed).
echo "[setup] Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# Install all other dependencies.
echo "[setup] Installing project dependencies..."
pip install -r "${REPO_DIR}/requirements.txt" -q

echo "[setup] Installed packages:"
pip list | grep -E "torch|wandb|numpy|scipy|sklearn|yaml|tqdm|einops"

# ── 3. Verify W&B ────────────────────────────────────────────────────────────
echo "[setup] Configuring W&B..."
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[setup] WARNING: WANDB_API_KEY not set."
    echo "[setup] Add to ~/.bashrc:  export WANDB_API_KEY=your_key_here"
    echo "[setup] Get your key at:   https://wandb.ai/settings"
else
    wandb login "${WANDB_API_KEY}" --relogin
    echo "[setup] W&B login successful."
fi

# ── 4. Create scratch directories ────────────────────────────────────────────
echo "[setup] Creating scratch directories..."
mkdir -p "/scratch/${USER}/data/ssv2_vjepa21_features"
mkdir -p "/scratch/${USER}/outputs/gap1"
mkdir -p "/scratch/${USER}/checkpoints"
mkdir -p "/scratch/${USER}/logs"
echo "[setup] Scratch dirs ready:"
ls -la "/scratch/${USER}/"

# ── 5. CPU smoke test ────────────────────────────────────────────────────────
echo "[setup] Running CPU smoke test (100 steps, synthetic data)..."
cd "${REPO_DIR}"

python -m training.trainer \
    --config   configs/base.yaml \
    --override configs/condition_E.yaml \
    --smoke_test \
    --max_steps 50 \
    --no_wandb

if [ $? -eq 0 ]; then
    echo "[setup] Smoke test PASSED."
else
    echo "[setup] ERROR: Smoke test FAILED. Check installation above."
    exit 1
fi

# ── 6. Run full unit tests ───────────────────────────────────────────────────
echo "[setup] Running unit tests..."
python -m pytest tests/ -v -q --tb=short

if [ $? -eq 0 ]; then
    echo "[setup] All unit tests PASSED."
else
    echo "[setup] ERROR: Unit tests failed."
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup complete. Next steps:"
echo ""
echo "  1. Download V-JEPA 2.1 checkpoint to:"
echo "     /scratch/${USER}/checkpoints/vjepa21_vitg.pt"
echo ""
echo "  2. Download SSv2 videos to:"
echo "     /scratch/${USER}/data/ssv2/videos/part-00/"
echo ""
echo "  3. Extract features:"
echo "     sbatch scripts/preextract_ssv2.sh"
echo ""
echo "  4. Verify baseline quality:"
echo "     python scripts/verify_baseline.py \\"
echo "       --feature_dir /scratch/${USER}/data/ssv2_vjepa21_features"
echo ""
echo "  5. Submit training conditions:"
echo "     bash scripts/submit_gap1.sh --critical-only"
echo "=========================================="
