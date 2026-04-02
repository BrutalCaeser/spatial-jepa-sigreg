#!/usr/bin/env python3
"""
Generate synthetic V-JEPA 2.1 patch features for pipeline validation.

Creates a small dataset (default 400 clips) in the exact format expected by
SSv2FeatureDataset, allowing the full training pipeline to run without real
SSv2 data.  Swap the feature_dir path once real SSv2 features are extracted.

Feature properties designed to match real V-JEPA 2.1 patch tokens:
  - Shape:    [196, 1024]  (14x14 spatial grid, 1024-dim ViT-L tokens)
  - erank:    > 20         (passes verify_baseline.py requirement)
  - ncorr:    > 0.30       (spatial smoothness via Gaussian blur on grid)
  - Class structure: 174 SSv2 classes, ~100 clips/class in full data

Usage:
    python scripts/generate_synthetic_features.py \
        --output_dir /scratch/$USER/data/ssv2_vjepa21_features \
        --n_clips 400 \
        --n_classes 174 \
        --seed 42

For smoke tests (fast, 60 clips):
    python scripts/generate_synthetic_features.py \
        --output_dir /scratch/$USER/data/ssv2_vjepa21_features \
        --n_clips 60 --seed 42
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


# ── Spatial smoothing via 2D Gaussian ────────────────────────────────────────
def _gaussian_kernel_2d(sigma: float = 1.5, size: int = 5) -> np.ndarray:
    """Create a 2D Gaussian kernel for spatial smoothing."""
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * ax ** 2 / sigma ** 2)
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()


def _smooth_grid(tokens: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Apply spatial Gaussian smoothing to patch tokens arranged on a 14x14 grid.

    Args:
        tokens: [196, 1024] — patch tokens in raster order (row-major, 14x14)
        sigma:  Gaussian smoothness (larger = smoother spatial correlation)

    Returns:
        Smoothed tokens [196, 1024]
    """
    from scipy.ndimage import gaussian_filter
    grid = tokens.reshape(14, 14, 1024)  # [14, 14, D]
    # Apply 2D Gaussian separately to each channel
    smoothed = gaussian_filter(grid, sigma=[sigma, sigma, 0])
    return smoothed.reshape(196, 1024)


def _make_class_basis(label: int, n_components: int, d: int) -> np.ndarray:
    """
    Fast low-rank basis for a class: sample random vectors, normalise rows only.
    Avoids expensive QR decomposition while still producing a structured subspace.

    Returns: [n_components, d] float32 array with unit-norm rows.
    """
    class_rng = np.random.default_rng(seed=label * 7919 + 42)
    basis = class_rng.standard_normal((n_components, d)).astype(np.float32)
    norms = np.linalg.norm(basis, axis=1, keepdims=True).clip(min=1e-8)
    return basis / norms  # unit-norm rows, no QR needed


def generate_clip_features(
    label: int,
    n_classes: int,
    d: int,
    rng: np.random.Generator,
    sigma: float = 1.8,
    n_components: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a (f_c, f_t) pair of synthetic patch token arrays.

    Feature construction:
      1. Sample a class-specific "prototype" in a low-rank subspace [n_components << d]
         so effective rank > 20 but representation is structured.  Uses row-normalised
         random basis (no QR) for speed: ~0.05s/clip vs ~22s with QR.
      2. Add per-clip Gaussian noise.
      3. Apply 2D Gaussian smoothing over the 14x14 spatial grid → ncorr > 0.3.
      4. f_t = f_c + small temporally-correlated noise.

    Args:
        label:        Integer class index [0, n_classes)
        n_classes:    Total number of classes
        d:            Feature dimension (1024)
        rng:          Seeded numpy Generator
        sigma:        Spatial smoothing sigma (controls ncorr)
        n_components: Subspace rank (controls erank; 64 → erank ≈ 30-50)

    Returns:
        (f_c, f_t): each [196, d] float32 numpy arrays
    """
    # ── 1. Class basis (row-normalised, precomputed per label) ───────────────
    basis = _make_class_basis(label, n_components, d)  # [n_components, d]

    # Per-clip coefficients: [196, n_components]
    coefs = rng.standard_normal((196, n_components)).astype(np.float32) * 0.8
    # Add a class-mean offset so different classes cluster differently
    class_mean = np.sin(np.arange(n_components) * (label + 1) * 0.3).astype(np.float32)
    coefs += class_mean[np.newaxis, :]

    tokens = coefs @ basis  # [196, d]

    # ── 2. Per-clip noise ────────────────────────────────────────────────────
    tokens += rng.standard_normal((196, d)).astype(np.float32) * 0.2

    # ── 3. Spatial smoothing ─────────────────────────────────────────────────
    f_c = _smooth_grid(tokens, sigma=sigma)

    # ── 4. Target frame: f_c + small temporally-correlated noise ─────────────
    t_noise = rng.standard_normal((196, d)).astype(np.float32) * 0.05
    s_noise = _smooth_grid(
        rng.standard_normal((196, d)).astype(np.float32) * 0.1, sigma=sigma * 0.8
    )
    f_t = f_c + t_noise + s_noise

    return f_c.astype(np.float32), f_t.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic V-JEPA 2.1 features")
    parser.add_argument(
        "--output_dir",
        default="/scratch/${USER}/data/ssv2_vjepa21_features".replace("${USER}", os.environ.get("USER", "user")),
        help="Output directory for synthetic features",
    )
    parser.add_argument("--n_clips", type=int, default=400, help="Total clips to generate")
    parser.add_argument("--n_classes", type=int, default=174, help="Number of action classes")
    parser.add_argument("--d", type=int, default=1024, help="Feature dimension (must match codebase: 1024)")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.8, help="Spatial smoothing sigma")
    parser.add_argument("--n_components", type=int, default=64, help="Subspace rank per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.json"
    if index_path.exists() and not args.overwrite:
        print(f"[synth] Index already exists at {index_path}. Use --overwrite to regenerate.")
        sys.exit(0)

    rng = np.random.default_rng(args.seed)

    n_train = int(args.n_clips * args.train_frac)
    n_val   = int(args.n_clips * args.val_frac)
    n_test  = args.n_clips - n_train - n_val

    splits = (
        [("train", i) for i in range(n_train)] +
        [("val",   i) for i in range(n_val)] +
        [("test",  i) for i in range(n_test)]
    )

    print(f"[synth] Generating {args.n_clips} clips → {out_dir}")
    print(f"[synth] Split: train={n_train}, val={n_val}, test={n_test}")
    print(f"[synth] Feature shape: [196, {args.d}], n_classes={args.n_classes}")
    print(f"[synth] Spatial sigma={args.sigma}, subspace_rank={args.n_components}")

    index = {"train": [], "val": [], "test": []}

    from tqdm import tqdm
    for idx, (split, _) in tqdm(enumerate(splits), total=len(splits), desc="Generating"):
        clip_id = f"synth_{idx:06d}"
        label = rng.integers(0, args.n_classes)

        f_c, f_t = generate_clip_features(
            label=int(label),
            n_classes=args.n_classes,
            d=args.d,
            rng=rng,
            sigma=args.sigma,
            n_components=args.n_components,
        )

        # Save in format expected by SSv2FeatureDataset
        torch.save(torch.from_numpy(f_c), out_dir / f"{clip_id}_fc.pt")
        torch.save(torch.from_numpy(f_t), out_dir / f"{clip_id}_ft.pt")

        index[split].append({"clip_id": clip_id, "label": int(label)})

    # Save index
    with open(index_path, "w") as fp:
        json.dump(index, fp, indent=2)

    print(f"\n[synth] ✓ Saved {args.n_clips} clip pairs to {out_dir}")
    print(f"[synth] ✓ Index written to {index_path}")

    # ── Quick quality check ────────────────────────────────────────────────────
    print("\n[synth] Running quick quality checks...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.metrics import effective_rank, neighbor_corr

    # Load 32 train samples
    n_check = min(32, n_train)
    f_list = []
    for entry in index["train"][:n_check]:
        clip_id = entry["clip_id"]
        f_c = torch.load(out_dir / f"{clip_id}_fc.pt", weights_only=True)
        f_list.append(f_c)

    f_all = torch.stack(f_list)  # [n_check, 196, 1024]
    erank = effective_rank(f_all.reshape(-1, args.d))
    ncorr = neighbor_corr(f_all, grid_size=14)

    print(f"[synth] raw_erank = {erank:.2f}  (required > 20)")
    print(f"[synth] raw_ncorr = {ncorr:.3f} (required > 0.30)")

    if erank > 20 and ncorr > 0.3:
        print("[synth] ✓ PASS: synthetic features pass verify_baseline.py requirements")
    else:
        print("[synth] ✗ WARN: features may not meet requirements. Adjust --sigma or --n_components.")
        if erank <= 20:
            print(f"[synth]   erank={erank:.1f} ≤ 20 → increase --n_components (current: {args.n_components})")
        if ncorr <= 0.3:
            print(f"[synth]   ncorr={ncorr:.3f} ≤ 0.3 → increase --sigma (current: {args.sigma})")


if __name__ == "__main__":
    main()
