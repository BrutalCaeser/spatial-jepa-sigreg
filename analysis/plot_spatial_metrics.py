"""
Figure 3: Spatial metrics comparison across conditions.

Shows neighbor_corr and token_diversity for all conditions as a grouped bar chart.
Visualizes Claim 2: SIGReg axis affects spatial structure.
    Expected ordering: D3 > D2 > D1 for neighbor_corr.

Rule 7 (CLAUDE.md): reads values from W&B, never hardcoded.

Usage:
    python analysis/plot_spatial_metrics.py \
        --entity YOUR_WANDB_ENTITY \
        --output analysis/outputs/spatial_metrics.pdf
"""

import argparse
from pathlib import Path

CONDITION_ORDER = ["A", "B", "C", "D1", "D2", "D3", "E", "F"]


def fetch_spatial_metrics(entity: str, project: str) -> dict:
    import wandb
    api = wandb.Api()

    metrics = {}
    for condition in CONDITION_ORDER:
        runs = api.runs(
            f"{entity}/{project}",
            filters={"tags": {"$in": [f"condition_{condition}"]}},
        )
        if not runs:
            continue

        run = sorted(runs, key=lambda r: r.created_at)[-1]
        hist = run.history(samples=1000, pandas=True)

        def get_final(key):
            if key in hist.columns:
                series = hist[key].dropna()
                return float(series.iloc[-1]) if len(series) > 0 else None
            return None

        metrics[condition] = {
            "ncorr_adapter": get_final("eval/ncorr_adapter"),
            "tokdiv_adapter": get_final("eval/tokdiv_adapter"),
        }

    return metrics


def make_plot(metrics: dict, output_path: str):
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = [c for c in CONDITION_ORDER if c in metrics]
    ncorr  = [metrics[c].get("ncorr_adapter") for c in conditions]
    tokdiv = [metrics[c].get("tokdiv_adapter") for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width/2, ncorr,  width, label="Neighbor Corr (spatial coherence)",
                   color="#1f77b4", alpha=0.8, edgecolor="black")
    bars2 = ax.bar(x + width/2, tokdiv, width, label="Token Diversity (within-sample diversity)",
                   color="#ff7f0e", alpha=0.8, edgecolor="black")

    # Highlight Condition E (proposed method).
    e_idx = conditions.index("E") if "E" in conditions else None
    if e_idx is not None:
        for bar in [bars1[e_idx], bars2[e_idx]]:
            bar.set_edgecolor("red")
            bar.set_linewidth(2.5)

    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title(
        "Spatial Metrics Across Conditions\n"
        "(Higher ncorr = more spatial structure preserved; "
        "Higher diversity = less collapse)",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Annotate prediction: D3 > D2 > D1 for ncorr.
    ax.annotate(
        "Predicted: D3 > D2 > D1",
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        ha="center", fontsize=9, color="gray",
        style="italic",
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[plot] Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot spatial metrics comparison")
    parser.add_argument("--entity",  required=True)
    parser.add_argument("--project", default="gap1-sigreg-spatial")
    parser.add_argument("--output",  default="analysis/outputs/spatial_metrics.pdf")
    args = parser.parse_args()

    print("[plot] Fetching spatial metrics from W&B...")
    metrics = fetch_spatial_metrics(args.entity, args.project)
    make_plot(metrics, args.output)


if __name__ == "__main__":
    main()
