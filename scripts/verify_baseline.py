#!/usr/bin/env python3
"""
Verify baseline quality of pre-extracted V-JEPA 2.1 features.

Run once after feature extraction, before any training:
    python scripts/verify_baseline.py \
        --feature_dir /scratch/$USER/data/ssv2_vjepa21_features \
        --n_samples 32

Required (build_spec.md Section 4.3):
    raw_erank > 20   (features are diverse, not collapsed)
    raw_ncorr > 0.3  (features have spatial structure)

If these fail -> FM5: V-JEPA version or token type is wrong. Halt and debug.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data.ssv2_dataset import SSv2FeatureDataset
from training.metrics import effective_rank, neighbor_corr


def main():
    parser = argparse.ArgumentParser(description="Verify V-JEPA 2.1 feature baseline quality")
    parser.add_argument("--feature_dir", required=True, help="Pre-extracted features directory")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=32, help="Number of clips to check")
    args = parser.parse_args()

    print(f"[verify] Loading {args.n_samples} clips from {args.feature_dir} ({args.split} split)...")

    dataset = SSv2FeatureDataset(args.feature_dir, split=args.split)
    n = min(args.n_samples, len(dataset))

    f_list = []
    for i in range(n):
        f_c, _, _ = dataset[i]
        f_list.append(f_c)  # [196, 1024]

    f_all = torch.stack(f_list)  # [n, 196, 1024]
    print(f"[verify] Loaded features shape: {f_all.shape}")

    # Compute metrics on raw frozen features.
    raw_erank = effective_rank(f_all.reshape(-1, 1024))
    raw_ncorr = neighbor_corr(f_all, grid_size=14)

    print(f"\n[verify] ========== BASELINE RESULTS ==========")
    print(f"[verify] raw_erank = {raw_erank:.2f}   (required > 20)")
    print(f"[verify] raw_ncorr = {raw_ncorr:.3f}  (required > 0.30)")
    print(f"[verify] ==========================================\n")

    # Check requirements.
    erank_ok = raw_erank > 20
    ncorr_ok = raw_ncorr > 0.3

    if erank_ok and ncorr_ok:
        print("[verify] PASS: V-JEPA 2.1 features are high-quality patch tokens.")
        print("[verify] Ready to run training conditions.")
        sys.exit(0)
    else:
        if not erank_ok:
            print(f"[verify] FAIL: raw_erank={raw_erank:.2f} < 20.")
            print("[verify]   -> Features are collapsed. Check:")
            print("[verify]      1. You are using V-JEPA 2.1 (not V-JEPA 2)")
            print("[verify]      2. You extracted PATCH tokens (not CLS or pooled)")
            print("[verify]      3. The ViT-G checkpoint is correct")
        if not ncorr_ok:
            print(f"[verify] FAIL: raw_ncorr={raw_ncorr:.3f} < 0.30.")
            print("[verify]   -> Features lack spatial structure. Check same points above.")
            print("[verify]   -> FM5 from build_spec.md: pivot experiment framing if confirmed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
