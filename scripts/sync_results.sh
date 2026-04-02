#!/bin/bash
# =============================================================================
# Sync training results from HPC back to local machine
# =============================================================================
# Run from LOCAL machine after HPC jobs complete.
#
# Usage:
#   bash scripts/sync_results.sh                    # sync checkpoints
#   bash scripts/sync_results.sh --logs-only        # sync SLURM logs only
#   bash scripts/sync_results.sh --condition E      # sync single condition
#
# What syncs:
#   HPC:/scratch/$USER/outputs/gap1/       -> ./outputs/
#   HPC:/scratch/$USER/logs/gap1_*.out     -> ./logs/
#
# Metrics (erank, probe_acc, etc.) come from W&B — no sync needed.
# Checkpoints are synced here for post-hoc analysis and eigenspectrum plots.

set -euo pipefail

HPC_HOST="discovery"              # ~/.ssh/config alias → login.discovery.neu.edu
HPC_USER="gupta.yashv"            # Northeastern HPC username
HPC_SCRATCH="/scratch/${HPC_USER}"

LOCAL_OUTPUTS="./outputs"
LOCAL_LOGS="./logs"

LOGS_ONLY=false
CONDITION=""

for arg in "$@"; do
    case $arg in
        --logs-only)        LOGS_ONLY=true ;;
        --condition)        shift; CONDITION="${1:-}" ;;
        --condition=*)      CONDITION="${arg#--condition=}" ;;
    esac
done

mkdir -p "${LOCAL_OUTPUTS}" "${LOCAL_LOGS}"

echo "=========================================="
echo "GAP 1 — Sync Results from HPC"
echo "HPC:   ${HPC_USER}@${HPC_HOST}"
echo "Date:  $(date)"
echo "=========================================="

# ── Test connectivity ─────────────────────────────────────────────────────────
echo "[sync] Testing SSH connection..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${HPC_HOST}" "echo ok" 2>/dev/null; then
    echo "[sync] ERROR: Cannot connect to ${HPC_HOST}."
    echo "[sync]   Check that your public key is in authorized_keys on the cluster."
    echo "[sync]   Public key: $(cat ~/.ssh/hpc_explorer.pub)"
    exit 1
fi
echo "[sync] Connection OK."

# ── Sync SLURM logs ───────────────────────────────────────────────────────────
echo "[sync] Syncing SLURM logs..."
rsync -avz --progress \
    "${HPC_HOST}:${HPC_SCRATCH}/logs/gap1_*.out" \
    "${HPC_HOST}:${HPC_SCRATCH}/logs/gap1_*.err" \
    "${LOCAL_LOGS}/" 2>/dev/null || echo "[sync] No SLURM logs found yet."

if [ "${LOGS_ONLY}" = true ]; then
    echo "[sync] --logs-only: skipping checkpoint sync."
    exit 0
fi

# ── Sync checkpoints ─────────────────────────────────────────────────────────
if [ -n "${CONDITION}" ]; then
    echo "[sync] Syncing checkpoints for Condition ${CONDITION}..."
    rsync -avz --progress \
        "${HPC_HOST}:${HPC_SCRATCH}/outputs/gap1/${CONDITION}/" \
        "${LOCAL_OUTPUTS}/${CONDITION}/"
else
    echo "[sync] Syncing all checkpoints..."
    rsync -avz --progress \
        "${HPC_HOST}:${HPC_SCRATCH}/outputs/gap1/" \
        "${LOCAL_OUTPUTS}/"
fi

echo ""
echo "=========================================="
echo "[sync] Sync complete: $(date)"
echo "[sync] Local outputs: ${LOCAL_OUTPUTS}/"
du -sh "${LOCAL_OUTPUTS}"/ 2>/dev/null || true
echo ""
echo "[sync] Metrics are in W&B — run analysis scripts:"
echo "       python analysis/generate_results_table.py --entity BrutalCaeser"
echo "=========================================="
