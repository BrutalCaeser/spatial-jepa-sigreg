"""
Linear probe evaluation on frozen adapter representations.

Definition 4.5 (foundations.md):
    Input:  z_pool = mean(A_theta(f), dim=1) in R^{B x d}  (pooled adapter output)
    Target: SSv2 clip class in {0, ..., 173}

    Probe: LogisticRegression trained on held-out test split.
    NO gradient flows back to adapter or predictor.
    Measures how much task-relevant information is linearly accessible.

Split (build_spec.md Section 4.2):
    Test set: 2K clips, split 50/50 → 1K probe train, 1K probe eval.

Usage (in-training):
    top1, top5 = run_linear_probe(adapter, data_loader, device)

Usage (post-hoc):
    python -m evaluation.linear_probe \
        --checkpoint outputs/condition_E/checkpoint_step050000.pt \
        --feature_dir /scratch/data/ssv2_vjepa21_features \
        --condition E
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_pooled_features(
    adapter: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract spatially-pooled adapter representations.

    Definition 4.5 (foundations.md):
        z_pool = mean(A_theta(f_c), dim=1)  in R^{B x d}

    NO gradients flow to adapter.

    Args:
        adapter:     Trained PatchAdapter (in eval mode during extraction).
        data_loader: DataLoader returning (f_c, f_t, label) batches.
        device:      Compute device.
        max_samples: Stop after this many samples (for probe train/eval split).

    Returns:
        features: np.ndarray of shape [N_total, d].
        labels:   np.ndarray of shape [N_total] with integer class indices.
    """
    adapter.eval()
    all_features = []
    all_labels   = []
    n_collected  = 0

    with torch.no_grad():
        for f_c, f_t, label in data_loader:
            f_c   = f_c.to(device)    # [B, N, D]
            label = label             # [B]  keep on CPU for numpy

            z_c     = adapter(f_c)         # [B, N, d]
            z_pool  = z_c.mean(dim=1)      # [B, d]  — spatial pooling
            z_pool  = z_pool.cpu().float().numpy()

            all_features.append(z_pool)
            all_labels.append(label.numpy())
            n_collected += z_pool.shape[0]

            if max_samples is not None and n_collected >= max_samples:
                break

    features = np.concatenate(all_features, axis=0)
    labels   = np.concatenate(all_labels,   axis=0)

    if max_samples is not None:
        features = features[:max_samples]
        labels   = labels[:max_samples]

    return features, labels


def run_linear_probe(
    adapter: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_iter: int = 1000,
    C: float = 1.0,
) -> Tuple[float, float]:
    """Train and evaluate a logistic regression linear probe.

    Uses the first half of data_loader for probe training and the second half
    for probe evaluation, emulating the 50/50 test-set split from build_spec.md.

    Args:
        adapter:     Trained PatchAdapter.
        data_loader: DataLoader over the test set (NOT train/val).
        device:      Compute device.
        max_iter:    Max sklearn solver iterations.
        C:           Regularization strength for LogisticRegression.

    Returns:
        (top1_accuracy, top5_accuracy) as floats in [0, 1].
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Extract all features.
    all_features, all_labels = extract_pooled_features(adapter, data_loader, device)
    N = len(all_labels)

    if N < 10:
        print("[probe] WARNING: fewer than 10 samples, probe unreliable.")
        return 0.0, 0.0

    # 50/50 split.
    n_train = N // 2
    X_train, y_train = all_features[:n_train], all_labels[:n_train]
    X_eval,  y_eval  = all_features[n_train:], all_labels[n_train:]

    # Standardize features (probe-specific normalization, not adapter outputs).
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval  = scaler.transform(X_eval)

    # Train logistic regression.
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Top-1 accuracy.
    top1 = clf.score(X_eval, y_eval)

    # Top-5 accuracy.
    probs = clf.predict_proba(X_eval)            # [N_eval, n_classes]
    top5_preds = np.argsort(probs, axis=1)[:, -5:]   # [N_eval, 5]
    top5 = float(np.mean([
        y_eval[i] in top5_preds[i]
        for i in range(len(y_eval))
    ]))

    return float(top1), float(top5)


# ---------------------------------------------------------------------------
# Post-hoc CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Linear probe evaluation for GAP 1")
    parser.add_argument("--checkpoint",   required=True, help="Adapter checkpoint .pt file")
    parser.add_argument("--feature_dir",  required=True, help="Pre-extracted SSv2 features dir")
    parser.add_argument("--condition",    default="unknown", help="Condition name for logging")
    parser.add_argument("--split",        default="test", choices=["val", "test"])
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--d",            type=int, default=256, help="Adapter output dim")
    parser.add_argument("--D_in",         type=int, default=1024)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity",  default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load adapter.
    from models.adapter import PatchAdapter
    adapter = PatchAdapter(D_in=args.D_in, D_out=args.d).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "adapter" in ckpt:
        adapter.load_state_dict(ckpt["adapter"])
    else:
        adapter.load_state_dict(ckpt)
    print(f"[probe] Loaded adapter from {args.checkpoint}")

    # Load data.
    from data.ssv2_dataset import SSv2FeatureDataset
    dataset    = SSv2FeatureDataset(args.feature_dir, split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Run probe.
    top1, top5 = run_linear_probe(adapter, dataloader, device)
    print(f"[probe] Condition {args.condition} | {args.split} | top1={top1:.4f} top5={top5:.4f}")

    # Log to W&B if requested.
    if args.wandb_project is not None:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"probe_condition_{args.condition}",
                tags=["probe", f"condition_{args.condition}", "gap1"],
            )
            wandb.log({
                "eval/probe_top1": top1,
                "eval/probe_top5": top5,
                "condition": args.condition,
            })
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
