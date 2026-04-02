"""
Central figure: effective rank vs linear probe accuracy scatter plot.

Figure 2 in the paper (build_spec.md Section 11):
    X-axis: erank (collapse prevention measure)
    Y-axis: probe_top1 (task-relevant information)
    Points: one per condition (A–F), labeled and color-coded

This is the primary visualization of the paper's Claim 3:
    E (SIGReg + L_info) achieves high erank AND high probe accuracy.

Rule 7 (CLAUDE.md): reads values from W&B API, never hardcoded.

Usage:
    python analysis/plot_erank_vs_probe.py \
        --entity YOUR_WANDB_ENTITY \
        --output analysis/outputs/erank_vs_probe.pdf
"""

import argparse
from pathlib import Path


CONDITION_COLORS = {
    "A":  "#d62728",   # red — collapse baseline
    "B":  "#ff7f0e",   # orange — stop-grad only
    "C":  "#2ca02c",   # green — stop-grad + global SIGReg
    "D1": "#9467bd",   # purple — per-token SIGReg
    "D2": "#8c564b",   # brown — per-channel SIGReg
    "D3": "#17becf",   # cyan — global SIGReg, no stop-grad
    "E":  "#1f77b4",   # blue — proposed method (highlight)
    "F":  "#e377c2",   # pink — L_info only ablation
}

CONDITION_MARKERS = {
    "A": "X", "B": "s", "C": "D", "D1": "v",
    "D2": "^", "D3": "o", "E": "*", "F": "P",
}


def fetch_final_metrics(entity: str, project: str):
    """Fetch final erank and probe_top1 for all conditions from W&B."""
    import wandb
    api = wandb.Api()

    metrics = {}
    for condition in ["A", "B", "C", "D1", "D2", "D3", "E", "F"]:
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
            "erank":     get_final("eval/erank"),
            "probe_top1": get_final("eval/probe_top1"),
        }

    return metrics


def make_plot(metrics: dict, output_path: str):
    """Generate the erank vs probe scatter plot."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(7, 5))

    for condition, vals in metrics.items():
        erank = vals.get("erank")
        probe = vals.get("probe_top1")
        if erank is None or probe is None:
            continue

        color  = CONDITION_COLORS.get(condition, "gray")
        marker = CONDITION_MARKERS.get(condition, "o")
        size   = 250 if condition == "E" else 100

        ax.scatter(erank, probe * 100, c=color, marker=marker, s=size,
                   zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(
            f"  {condition}",
            (erank, probe * 100),
            fontsize=9,
            ha="left",
            va="center",
        )

    ax.set_xlabel("Effective Rank (erank)", fontsize=12)
    ax.set_ylabel("Linear Probe Top-1 Accuracy (%)", fontsize=12)
    ax.set_title(
        "Collapse Prevention vs. Information Preservation\n"
        "(★ = Condition E: proposed SIGReg + L_info_dense method)",
        fontsize=11,
    )

    # Add reference lines.
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="erank=1 (collapse)")
    ax.grid(True, alpha=0.3)

    # Legend patches.
    patches = [
        mpatches.Patch(color=CONDITION_COLORS[c], label=f"Cond. {c}")
        for c in ["A", "B", "C", "D1", "D2", "D3", "E", "F"]
        if c in metrics
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[plot] Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot erank vs probe accuracy")
    parser.add_argument("--entity",  required=True)
    parser.add_argument("--project", default="gap1-sigreg-spatial")
    parser.add_argument("--output",  default="analysis/outputs/erank_vs_probe.pdf")
    args = parser.parse_args()

    print("[plot] Fetching metrics from W&B...")
    metrics = fetch_final_metrics(args.entity, args.project)
    print(f"[plot] Found metrics for conditions: {list(metrics.keys())}")

    make_plot(metrics, args.output)


if __name__ == "__main__":
    main()
