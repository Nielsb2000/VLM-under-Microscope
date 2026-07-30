"""
All models x all blur levels accuracy bar chart (with dotted separators between blur groups).
Reads from Results/dashboard_data/ via load_results_df().

Usage (from project root):
    uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_all_conditions.py \
        --image-type color --reasoning-mode medium --skills-mode no_skills
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from json_results_to_df import load_results_df

OUT_BASE = os.path.join(os.path.dirname(__file__), "..", "Results", "res_vis")
BLUR_LEVELS = ["no_blur", "med_blur", "heavy_blur"]
MODEL_ORDER = ["gpt-4o", "gpt-5.1", "gpt-5.2", "gpt-5.5"]

MODEL_COLORS = {
    "gpt-4o":  "#56B4E9",
    "gpt-5.1": "#E69F00",
    "gpt-5.2": "#009E73",
    "gpt-5.5": "#CC79A7",
}


def model_sort_key(model: str) -> tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(MODEL_ORDER), model)


def models_in_order(models) -> list[str]:
    return sorted(set(models), key=model_sort_key)


def main():
    parser = argparse.ArgumentParser(
        description="Overall model accuracy across all blur levels in one chart.")
    parser.add_argument("--image-type", default="color",
                        choices=["color", "greyscale", "inverted_greyscale"])
    parser.add_argument("--reasoning-mode", default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("--skills-mode", default="no_skills",
                        choices=["skills", "no_skills"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to specific models (default: all found, ordered as gpt-4o, gpt-5.1, gpt-5.2, gpt-5.5).")
    args = parser.parse_args()

    df = load_results_df()
    if df.empty:
        print("No data found in Results/dashboard_data/")
        return

    df["Correct"] = df["Correct"].astype(float)
    models = models_in_order(args.models if args.models else df["Model"].unique())

    accuracies = []
    for blur in BLUR_LEVELS:
        base_mask = (
            (df["image_type"] == args.image_type)
            & (df["blur_level"] == blur)
            & (df["skills_mode"] == args.skills_mode)
            & (df["Model"].isin(models))
        )
        if "reasoning_mode" in df.columns:
            base_mask &= (df["reasoning_mode"] == args.reasoning_mode) | (df["Model"] == "gpt-4o")
        sub = df[base_mask].drop_duplicates(subset=["image_num", "question_num", "Model"])
        for model in models:
            valid = sub[sub["Model"] == model]["Correct"].dropna()
            acc = valid.mean() if len(valid) > 0 else np.nan
            accuracies.append((blur, model, acc))

    if not any(not np.isnan(acc) for _, _, acc in accuracies):
        print("No accuracy data found.")
        return

    labels = [f"{model}\n({blur})" for blur, model, _ in accuracies]
    values = [0.0 if np.isnan(acc) else acc * 100 for _, _, acc in accuracies]
    bar_colors = [MODEL_COLORS.get(model, "#0072B2") for _, model, _ in accuracies]
    unique_models = models_in_order(model for _, model, _ in accuracies)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 6))
    bars = ax.bar(labels, values, color=bar_colors, alpha=0.85)
    ax.set_title(
        f"MS Paint Accuracy - All Blur Levels\n"
        f"({args.image_type}, reasoning: {args.reasoning_mode}, {args.skills_mode})",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[Patch(color=MODEL_COLORS.get(m, "#0072B2"), label=m) for m in unique_models],
        title="Model", fontsize=10, title_fontsize=11,
    )
    for bar, (_, _, acc) in zip(bars, accuracies):
        if not np.isnan(acc):
            ax.text(bar.get_x() + bar.get_width() / 2, acc * 100 + 1.5,
                    f"{acc * 100:.1f}%", ha="center", fontsize=9, fontweight="bold")

    for b in range(1, len(BLUR_LEVELS)):
        ax.axvline(x=b * len(models) - 0.5, color="red", linestyle=":", linewidth=2)

    fig.tight_layout()
    out_dir = os.path.normpath(
        os.path.join(OUT_BASE, f"{args.image_type}_{args.reasoning_mode}_{args.skills_mode}")
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "accuracy_all_blur.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
