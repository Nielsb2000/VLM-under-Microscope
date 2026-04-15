"""
Plot skills vs baseline accuracy comparison for a given task.

Reads the accuracy CSVs from eval_summary/{vqa,vtqa}/{task}_acc.csv,
identifies which rows are skills vs baseline, and produces a grouped
bar chart saved as {out_dir}/{task}_skills_comparison.png.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves PNG without needing a display
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def classify_row(model_name: str) -> str:
    """Return 'skills' or 'baseline' based on model name string."""
    return "skills" if "_skills_" in model_name or model_name.endswith("_skills") else "baseline"


def load_mode_csv(eval_summary_dir: str, mode: str, task: str) -> pd.DataFrame:
    csv_path = os.path.join(eval_summary_dir, mode, f"{task}_acc.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["variant"] = df["Model Name"].apply(classify_row)
    df["mode"] = mode.upper()
    return df


def best_accuracy(df: pd.DataFrame, variant: str) -> float:
    """Return the best accuracy for a given variant (most recent run if multiple)."""
    rows = df[df["variant"] == variant]
    if rows.empty:
        return 0.0
    return rows["Acc"].max()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    modes = ["vqa", "vtqa"]
    results = {}  # mode -> {variant -> accuracy}

    for mode in modes:
        df = load_mode_csv(args.eval_summary_dir, mode, args.task)
        if df.empty:
            print(f"[WARN] No data for {mode}/{args.task}, skipping.")
            results[mode] = {"baseline": 0.0, "skills": 0.0}
            continue
        results[mode] = {
            "baseline": best_accuracy(df, "baseline"),
            "skills": best_accuracy(df, "skills"),
        }
        print(f"{mode.upper()} — baseline: {results[mode]['baseline']:.1%}  skills: {results[mode]['skills']:.1%}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    x = np.arange(len(modes))
    width = 0.32

    baseline_accs = [results[m]["baseline"] * 100 for m in modes]
    skills_accs   = [results[m]["skills"] * 100   for m in modes]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars_base  = ax.bar(x - width / 2, baseline_accs, width, label="GPT-5.2 Baseline",
                        color="#555555", edgecolor="white", linewidth=0.8)
    bars_skill = ax.bar(x + width / 2, skills_accs,   width, label="GPT-5.2 + Skills",
                        color="#56B4E9", edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar in bars_base:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars_skill:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Delta annotation between bars
    for i, mode in enumerate(modes):
        delta = results[mode]["skills"] - results[mode]["baseline"]
        sign = "+" if delta >= 0 else ""
        y_pos = max(baseline_accs[i], skills_accs[i]) + 5.5
        color = "#009E73" if delta >= 0 else "#D55E00"
        ax.text(i, y_pos, f"Δ {sign}{delta * 100:.1f}%",
                ha="center", va="bottom", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes], fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(f"Task: {args.task.capitalize()} — Skills vs Baseline\n"
                 f"(GPT-5.2, first_k={args.first_k})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f"{args.task}_skills_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_summary_dir", default="eval_summary",
                        help="Directory containing {mode}/{task}_acc.csv files")
    parser.add_argument("--task", default="mazenav",
                        choices=["mazenav", "spatialgrid", "spatialmap", "spatialreal"])
    parser.add_argument("--out_dir", default="eval_summary/result_vis",
                        help="Where to save the PNG")
    parser.add_argument("--first_k", type=int, default=10,
                        help="Used only for the plot title (informational)")
    args = parser.parse_args()
    main(args)
