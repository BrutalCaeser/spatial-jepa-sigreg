#!/bin/bash
#SBATCH --job-name=gap1-synth-gen
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch/%u/logs/gap1_synth_%j.out
#SBATCH --error=/scratch/%u/logs/gap1_synth_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gupta.yashv@northeastern.edu

# =============================================================================
# GAP 1 Synthetic Feature Generation — pipeline validation without real SSv2
# =============================================================================
# Generates 400 synthetic [196, 1024] patch feature files in the exact format
# expected by SSv2FeatureDataset, for immediate pipeline validation.
# Fast: ~30 seconds on CPU (row-normalised basis, no QR decomposition).
#
# Submit standalone:
#   sbatch scripts/generate_synthetic_features.sh
#
# Or as dependency before training conditions (see submit_gap1.sh).
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "GAP 1 Synthetic Feature Generation"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURMD_NODENAME}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="/scratch/${USER}/data/ssv2_vjepa21_features"
PROJECT_DIR="/scratch/${USER}/gap1-experiment"

mkdir -p "${OUTPUT_DIR}" "/scratch/${USER}/logs"

CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
[ -d "${CONDA_BASE}" ] || CONDA_BASE="/shared/centos7/anaconda3/2021.05"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gap1

echo "[synth] Python: $(python --version)"

cd "${PROJECT_DIR}"

echo "[synth] Generating 400 synthetic clips..."
python scripts/generate_synthetic_features.py \
    --output_dir "${OUTPUT_DIR}" \
    --n_clips 400 \
    --n_classes 174 \
    --seed 42 \
    --overwrite

echo ""
echo "[synth] Verifying output..."
python - << 'PYEOF'
import json
from pathlib import Path
d = Path("/scratch/gupta.yashv/data/ssv2_vjepa21_features")
idx = d / "index.json"
if not idx.exists():
    print("[synth] ERROR: index.json not created!")
    exit(1)
data = json.load(open(idx))
print(f"[synth] ✓ index.json: train={len(data['train'])}, val={len(data['val'])}, test={len(data['test'])}")
n_fc = len(list(d.glob("*_fc.pt")))
print(f"[synth] ✓ Feature files: {n_fc} fc.pt files")
PYEOF

echo "[synth] =========================================="
echo "[synth] Synthetic generation complete: $(date)"
echo "[synth] Output: ${OUTPUT_DIR}"
echo "[synth] Next: training conditions will start automatically if submitted with --dependency"
echo "[synth] =========================================="
