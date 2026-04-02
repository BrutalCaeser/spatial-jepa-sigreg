"""
Figure 1: Eigenspectrum (singular value spectrum) per condition.

Shows the distribution of singular value energy across components.
Collapsed representations (Condition A) have all energy in component 1.
Well-regularized representations (E, D3) have more uniform energy.

Rule 7 (CLAUDE.md): loads values from saved checkpoints, not hardcoded.

Usage:
    python analysis/plot_eigenspectrum.py \
        --checkpoint_dir /scratch/$USER/outputs/gap1 \
        --feature_dir /scratch/$USER/data/ssv2_vjepa21_features \
        --output analysis/outputs/eigenspectrum.pdf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CONDITION_ORDER = ["A", "B", "C", "D3", "E", "F"]


def load_adapter_representations(checkpoint_path: str, feature_dir: str, n_samples: int = 256):
    """Load adapter checkpoint and extract val set representations."""
    import torch
    from models.adapter import PatchAdapter
    from data.ssv2_dataset import SSv2FeatureDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("cfg", {})
    d    = cfg.get("d", 256)
    D_in = cfg.get("D_in", 1024)

    adapter = PatchAdapter(D_in=D_in, D_out=d).to(device)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()

    dataset = SSv2FeatureDataset(feature_dir, split="val")
    n = min(n_samples, len(dataset))

    z_list = []
    with torch.no_grad():
        for i in range(n):
            f_c, _, _ = dataset[i]
            f_c = f_c.unsqueeze(0).to(device)  # [1, N, D]
            z   = adapter(f_c).squeeze(0)       # [N, d]
            z_list.append(z.cpu())

    z_all = torch.cat(z_list, dim=0)   # [n*N, d]
    return z_all


def compute_sv_spectrum(Z):
    """Compute normalized singular value energy distribution."""
    import torch
    sv = torch.linalg.svdvals(Z.float())
    energy = sv ** 2
    total  = energy.sum()
    return (energy / total).numpy() if total > 0 else energy.numpy()


def make_plot(spectra: dict, output_path: str):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
    axes = axes.flatten()

    colors = {
        "A": "red", "B": "orange", "C": "green",
        "D3": "cyan", "E": "blue", "F": "pink",
    }

    for idx, condition in enumerate(CONDITION_ORDER):
        if condition not in spectra:
            continue
        ax = axes[idx]
        p  = spectra[condition]

        top_k = min(50, len(p))
        x = np.arange(1, top_k + 1)

        ax.bar(x, p[:top_k], color=colors.get(condition, "gray"), alpha=0.8, edgecolor="none")
        ax.set_title(f"Condition {condition}", fontsize=11,
                     fontweight="bold" if condition == "E" else "normal")
        ax.set_xlabel("Singular Value Index", fontsize=9)
        ax.set_ylabel("Normalized Energy", fontsize=9)
        ax.set_xlim(0, top_k + 1)
        ax.grid(axis="y", alpha=0.3)

        # Annotate effective rank.
        erank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
        ax.text(0.65, 0.85, f"erank={erank:.1f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Hide unused axes.
    for i in range(len(CONDITION_ORDER), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Singular Value Spectrum per Condition\n"
        "(Condition A should show all energy in first component = collapsed)",
        fontsize=12,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[plot] Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot eigenspectrum per condition")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing condition_X/ subdirs with checkpoints")
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--output", default="analysis/outputs/eigenspectrum.pdf")
    parser.add_argument("--n_samples", type=int, default=256)
    args = parser.parse_args()

    spectra = {}
    for condition in CONDITION_ORDER:
        ckpt_dir = Path(args.checkpoint_dir) / condition
        # Find latest checkpoint.
        ckpts = sorted(ckpt_dir.glob("checkpoint_step*.pt")) if ckpt_dir.exists() else []
        if not ckpts:
            print(f"[eigen] No checkpoint found for Condition {condition}, skipping.")
            continue

        ckpt_path = str(ckpts[-1])
        print(f"[eigen] Condition {condition}: {ckpt_path}")

        Z = load_adapter_representations(ckpt_path, args.feature_dir, args.n_samples)
        spectra[condition] = compute_sv_spectrum(Z)

    if not spectra:
        print("[eigen] No spectra computed. Check checkpoint_dir.")
        return

    make_plot(spectra, args.output)


if __name__ == "__main__":
    main()
