#!/bin/bash
#SBATCH --job-name=gap1-ssv2-dl
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/%u/logs/gap1_ssv2_dl_%j.out
#SBATCH --error=/scratch/%u/logs/gap1_ssv2_dl_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gupta.yashv@northeastern.edu

# =============================================================================
# GAP 1 SSv2 Data Download — HuggingFace → Explorer HPC
# =============================================================================
# Downloads 20K clips from HuggingFaceM4/something_something_v2.
# No GPU required; runs on CPU partition.
#
# Submit:
#   sbatch scripts/download_ssv2.sh
#
# Resume (if interrupted — script skips existing files):
#   sbatch scripts/download_ssv2.sh
#
# After completion, run:
#   sbatch scripts/preextract_ssv2.sh
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "GAP 1 SSv2 Download"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURMD_NODENAME}"
echo "Start:   $(date)"
echo "=========================================="

DATA_DIR="/scratch/${USER}/data/ssv2"
HF_CACHE="/scratch/${USER}/hf_cache"
PROJECT_DIR="/scratch/${USER}/gap1-experiment"

mkdir -p "${DATA_DIR}/videos" "${HF_CACHE}" "/scratch/${USER}/logs"

# ── Activate conda environment ────────────────────────────────────────────────
CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
[ -d "${CONDA_BASE}" ] || CONDA_BASE="/shared/centos7/anaconda3/2021.05"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gap1

echo "[dl] Python: $(python --version)"

# The vjepa2 repo (pip install -e) adds vjepa2/src to sys.path which contains
# a src/datasets/ namespace dir that shadows the HuggingFace datasets package.
# Verify we can import HuggingFace datasets without that path conflict.
python - << 'PYEOF'
import sys
sys.path = [p for p in sys.path if "/vjepa2/src" not in p]
try:
    from datasets import load_dataset
    import importlib.metadata
    print(f"[dl] datasets OK: {importlib.metadata.version('datasets')}")
except ImportError:
    import subprocess, sys
    print("[dl] Installing HuggingFace datasets...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets>=2.18.0", "-q"])
    from datasets import load_dataset
    print("[dl] datasets installed OK")
PYEOF

# ── Run download ─────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"

echo "[dl] Starting SSv2 download from HuggingFace..."
echo "[dl] Target: ${DATA_DIR}"
echo "[dl] N clips: 20000"
echo ""

python scripts/download_ssv2_hf.py \
    --output_dir  "${DATA_DIR}" \
    --split       train \
    --n_clips     20000 \
    --hf_cache    "${HF_CACHE}" \
    --num_proc    8 \
    --seed        42

echo ""
echo "[dl] =========================================="
echo "[dl] Download finished: $(date)"
echo ""
echo "[dl] Checking video count:"
N_VIDEOS=$(find "${DATA_DIR}/videos" -name "*.mp4" -o -name "*.webm" 2>/dev/null | wc -l)
echo "[dl] Videos: ${N_VIDEOS}"

if [ "${N_VIDEOS}" -gt 15000 ]; then
    echo "[dl] ✓ Sufficient videos available."
    echo "[dl] Next step: sbatch scripts/preextract_ssv2.sh"
else
    echo "[dl] ✗ Fewer than 15K videos. Re-run this job to resume."
fi
echo "[dl] =========================================="
