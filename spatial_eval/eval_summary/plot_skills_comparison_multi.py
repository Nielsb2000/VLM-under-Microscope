"""
Cross-round skills vs baseline comparison with mean ± std error bars.

Reads {mode}/{task}_acc.csv from each eval_dir (one per round), collects
baseline and skills accuracy per round, then plots mean ± std for each
task in a grouped bar chart with error bars.

Usage:
  uv run python eval_summary/plot_skills_comparison_multi.py \
    --eval_dirs eval_summary_final eval_summary_final_r2 eval_summary_final_r3 \
    --tasks mazenav spatialgrid spatialmap \
    --out_dir eval_summary_final_stats
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def classify_row(model_name: str) -> str:
    return "skills" if "_skills_" in model_name or model_name.endswith("_skills") else "baseline"


def best_acc_for_variant(csv_path: str, variant: str) -> float | None:
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    df["variant"] = df["Model Name"].apply(classify_row)
    rows = df[df["variant"] == variant]
    return float(rows["Acc"].max()) if not rows.empty else None


def collect_round_accs(eval_dirs: list[str], task: str, mode: str) -> dict[str, list[float]]:
    """Return {'baseline': [acc_r1, acc_r2, ...], 'skills': [...]} across all rounds."""
    result = {"baseline": [], "skills": []}
    for d in eval_dirs:
        csv_path = os.path.join(d, mode, f"{task}_acc.csv")
        for variant in ("baseline", "skills"):
            acc = best_acc_for_variant(csv_path, variant)
            if acc is not None:
                result[variant].append(acc * 100)
    return result


def plot_task(task: str, eval_dirs: list[str], out_dir: str, n_rounds: int):
    modes = ["vqa", "vtqa"]
    x = np.arange(len(modes))
    width = 0.32

    baseline_means, baseline_stds = [], []
    skills_means, skills_stds = [], []

    for mode in modes:
        accs = collect_round_accs(eval_dirs, task, mode)
        b = accs["baseline"]
        s = accs["skills"]
        baseline_means.append(np.mean(b) if b else 0.0)
        baseline_stds.append(np.std(b, ddof=1) if len(b) > 1 else 0.0)
        skills_means.append(np.mean(s) if s else 0.0)
        skills_stds.append(np.std(s, ddof=1) if len(s) > 1 else 0.0)

        print(f"  {task} {mode.upper()} baseline: {b}  → {baseline_means[-1]:.1f}% ± {baseline_stds[-1]:.1f}%")
        print(f"  {task} {mode.upper()} skills:   {s}  → {skills_means[-1]:.1f}% ± {skills_stds[-1]:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 5))

    bars_base = ax.bar(x - width / 2, baseline_means, width,
                       yerr=baseline_stds, capsize=5,
                       label=f"GPT-5.2 Baseline",
                       color="#5B8DB8", edgecolor="white", linewidth=0.8,
                       error_kw=dict(elinewidth=1.5, ecolor="#3a6a96"))
    bars_skill = ax.bar(x + width / 2, skills_means, width,
                        yerr=skills_stds, capsize=5,
                        label=f"GPT-5.2 + Skills",
                        color="#F28C38", edgecolor="white", linewidth=0.8,
                        error_kw=dict(elinewidth=1.5, ecolor="#c06820"))

    # Value labels
    for bar, mean, std in zip(bars_base, baseline_means, baseline_stds):
        h = bar.get_height()
        label = f"{mean:.1f}%" if std == 0 else f"{mean:.1f}%\n±{std:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, h + (std or 0) + 1.2,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, mean, std in zip(bars_skill, skills_means, skills_stds):
        h = bar.get_height()
        label = f"{mean:.1f}%" if std == 0 else f"{mean:.1f}%\n±{std:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, h + (std or 0) + 1.2,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Delta annotation
    for i, mode in enumerate(modes):
        delta = skills_means[i] - baseline_means[i]
        sign = "+" if delta >= 0 else ""
        y_pos = max(baseline_means[i] + baseline_stds[i],
                    skills_means[i] + skills_stds[i]) + 6.5
        color = "#2ca02c" if delta >= 0 else "#d62728"
        ax.text(i, y_pos, f"Δ {sign}{delta:.1f}%",
                ha="center", va="bottom", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes], fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    round_label = f"{n_rounds} rounds × 100 samples"
    ax.set_title(f"Task: {task.capitalize()} — Skills vs Baseline\n"
                 f"(GPT-5.2, {round_label}, mean ± std)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{task}_skills_comparison_multi.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    n = len(args.eval_dirs)
    print(f"\nAggregating {n} rounds: {args.eval_dirs}\n")
    for task in args.tasks:
        print(f"── {task} ──")
        plot_task(task, args.eval_dirs, args.out_dir, n)
    print(f"\nDone. Plots saved to {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dirs", nargs="+", required=True,
                        help="One eval_summary dir per round, in order")
    parser.add_argument("--tasks", nargs="+",
                        default=["mazenav", "spatialgrid", "spatialmap"])
    parser.add_argument("--out_dir", default="eval_summary_final_stats")
    args = parser.parse_args()
    main(args)
