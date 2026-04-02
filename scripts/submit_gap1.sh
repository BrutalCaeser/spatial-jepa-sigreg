#!/bin/bash
# =============================================================================
# Submit all 8 GAP 1 conditions to SLURM HPC scheduler.
# =============================================================================
# Usage:
#   bash scripts/submit_gap1.sh [--critical-only] [--dry-run]
#
# Options:
#   --critical-only  Submit only A, B, D3, E, F (minimum viable experiment)
#   --dry-run        Print sbatch commands without executing them
#
# Total compute: 76 GPU-hours across 8 jobs on A100 (build_spec.md Section 13)
# Minimum viable: 44 GPU-hours (A, B, D3, E, F)
#
# Prerequisites (all must pass before running):
#   1. python -m pytest tests/ -v --timeout=120   (zero failures)
#   2. python scripts/verify_baseline.py ...      (raw_erank>20, raw_ncorr>0.3)

set -euo pipefail

CRITICAL_ONLY=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --critical-only) CRITICAL_ONLY=true ;;
        --dry-run)       DRY_RUN=true ;;
    esac
done

# All conditions.
ALL_CONDITIONS=(A B C D1 D2 D3 E F)
# Minimum viable experiment (build_spec.md Section 3.2).
CRITICAL_CONDITIONS=(A B D3 E F)

if [ "$CRITICAL_ONLY" = true ]; then
    CONDITIONS=("${CRITICAL_CONDITIONS[@]}")
    echo "[submit] Submitting CRITICAL conditions only: ${CONDITIONS[*]}"
else
    CONDITIONS=("${ALL_CONDITIONS[@]}")
    echo "[submit] Submitting ALL conditions: ${CONDITIONS[*]}"
fi

# Run unit tests first.
if [ "$DRY_RUN" = false ]; then
    echo "[submit] Running unit tests before submission..."
    python -m pytest tests/ -v --timeout=120 -q
    if [ $? -ne 0 ]; then
        echo "ERROR: Unit tests failed. Fix before submitting."
        exit 1
    fi
    echo "[submit] All tests PASSED."
fi

# Submit each condition.
JOB_IDS=()
for CONDITION in "${CONDITIONS[@]}"; do
    CMD="sbatch --export=CONDITION=${CONDITION} scripts/run_condition.sh"
    echo "[submit] ${CMD}"

    if [ "$DRY_RUN" = false ]; then
        JOB_ID=$(${CMD} | awk '{print $4}')
        JOB_IDS+=("${CONDITION}:${JOB_ID}")
        echo "[submit]   -> Job ${JOB_ID} submitted for Condition ${CONDITION}"
        sleep 1   # Small delay to avoid scheduler flooding
    fi
done

echo ""
echo "[submit] ============================================="
echo "[submit] Submitted conditions: ${CONDITIONS[*]}"
if [ "$DRY_RUN" = false ]; then
    echo "[submit] Job IDs:"
    for entry in "${JOB_IDS[@]}"; do
        echo "[submit]   ${entry}"
    done
    echo ""
    echo "[submit] Monitor with: squeue -u \$USER"
    echo "[submit] W&B dashboard: https://wandb.ai/YOUR_ENTITY/gap1-sigreg-spatial"
fi
echo "[submit] ============================================="

# Reminder: Check Condition A collapse within 4 hours.
if echo "${CONDITIONS[@]}" | grep -qw "A"; then
    echo ""
    echo "[submit] IMPORTANT: Monitor Condition A every hour for first 4 hours."
    echo "[submit]   Expected: erank < 3 by step 2000."
    echo "[submit]   If erank > 5 after step 5000: HALT and execute FM1 (build_spec.md)."
fi
