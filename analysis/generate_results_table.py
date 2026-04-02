"""
Generate the main results table from W&B logs.

Rule 7 (CLAUDE.md): Every number in the paper comes from W&B logs, never from memory.
This script reads all metric values directly from the W&B API.

Usage:
    python analysis/generate_results_table.py \
        --entity brutalcaesar-northeastern-university \
        --project gap1-sigreg-spatial \
        --output_dir analysis/outputs

Output:
    results_table.csv    — all conditions × metrics
    results_table.tex    — LaTeX table for the paper

Table format (build_spec.md Section 11 — Table 1):
    Condition | erank | probe_acc | neighbor_corr | xcov_trace | Collapsed?
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


CONDITION_ORDER = ["A", "B", "C", "D1", "D2", "D3", "E", "F"]

# W&B metric keys to extract from run history (final values).
METRIC_KEYS = {
    "eval/erank":        "erank",
    "eval/probe_top1":   "probe_top1",
    "eval/ncorr_adapter": "ncorr_adapter",
    "eval/xcov_trace":   "xcov_trace",
    "eval/xcov_trace_dense": "xcov_trace_dense",
    "eval/infonce_mi":   "infonce_mi",
    "eval/tokdiv_adapter": "tokdiv_adapter",
}


def fetch_run_metrics(entity: str, project: str, condition: str) -> dict:
    """Fetch final metric values for a condition from W&B."""
    try:
        import wandb
        api = wandb.Api()
    except ImportError:
        raise ImportError("wandb not installed. pip install wandb")

    # Find runs matching this condition.
    runs = api.runs(
        f"{entity}/{project}",
        filters={"tags": {"$in": [f"condition_{condition}"]}},
    )

    if not runs:
        print(f"[table] WARNING: No W&B run found for condition_{condition}")
        return {"condition": condition}

    # Use most recent run.
    run = sorted(runs, key=lambda r: r.created_at)[-1]
    print(f"[table] Condition {condition}: run {run.name} (id={run.id})")

    history = run.history(samples=1000, pandas=True)
    metrics = {"condition": condition}

    for wandb_key, col_name in METRIC_KEYS.items():
        if wandb_key in history.columns:
            # Get the last non-NaN value.
            series = history[wandb_key].dropna()
            if len(series) > 0:
                metrics[col_name] = float(series.iloc[-1])
            else:
                metrics[col_name] = float("nan")
        else:
            metrics[col_name] = float("nan")

    return metrics


def infer_collapse(row: pd.Series) -> str:
    """Infer whether a condition collapsed based on erank."""
    erank = row.get("erank", float("nan"))
    if pd.isna(erank):
        return "?"
    if erank < 3.0:
        return "Yes"
    elif erank > 10.0:
        return "No"
    else:
        return "Partial"


def to_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table string."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Results for all GAP 1 experimental conditions. "
        r"All values from W&B logs (Rule 7, CLAUDE.md). "
        r"erank: effective rank; ncorr: neighbor token correlation; "
        r"xcov\_dense: dense cross-covariance trace; probe: top-1 linear probe accuracy.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Cond. & erank & probe & ncorr & xcov\_dense & infonce & Collapsed? \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        def fmt(val, decimals=2):
            if pd.isna(val):
                return "--"
            return f"{val:.{decimals}f}"

        line = (
            f"{row['condition']} & "
            f"{fmt(row.get('erank'))} & "
            f"{fmt(row.get('probe_top1', float('nan')), 3)} & "
            f"{fmt(row.get('ncorr_adapter'))} & "
            f"{fmt(row.get('xcov_trace_dense'))} & "
            f"{fmt(row.get('infonce_mi'))} & "
            f"{row.get('collapsed', '?')} \\\\"
        )
        lines.append(line)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate results table from W&B")
    parser.add_argument("--entity",      required=True, help="W&B entity")
    parser.add_argument("--project",     default="gap1-sigreg-spatial")
    parser.add_argument("--output_dir",  default="analysis/outputs")
    parser.add_argument("--conditions",  nargs="+", default=CONDITION_ORDER,
                        help="Conditions to include")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch metrics for each condition.
    rows = []
    for condition in args.conditions:
        row = fetch_run_metrics(args.entity, args.project, condition)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("condition").reindex(
        [c for c in CONDITION_ORDER if c in args.conditions]
    ).reset_index()

    # Add collapse indicator.
    df["collapsed"] = df.apply(infer_collapse, axis=1)

    # Save CSV.
    csv_path = output_dir / "results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"[table] Saved: {csv_path}")

    # Save LaTeX.
    latex_str = to_latex_table(df)
    tex_path  = output_dir / "results_table.tex"
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"[table] Saved: {tex_path}")

    # Print to console.
    print("\n" + "=" * 60)
    print("RESULTS TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
