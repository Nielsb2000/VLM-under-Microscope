"""
Heavy blur accuracy comparison between two reasoning modes (e.g. medium vs high).
Useful for showing the effect of increased reasoning effort on the hardest blur condition.
Reads from Results/dashboard_data/ via load_results_df().

Usage (from project root):
    uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heavy_blur_high.py \
        --image-type color --mode-a medium --mode-b high --skills-mode no_skills
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
        description="Compare heavy blur accuracy between two reasoning modes.")
    parser.add_argument("--image-type", default="color",
                        choices=["color", "greyscale", "inverted_greyscale"])
    parser.add_argument("--mode-a", default="medium", choices=["low", "medium", "high"],
                        help="First reasoning mode (left group).")
    parser.add_argument("--mode-b", default="high", choices=["low", "medium", "high"],
                        help="Second reasoning mode (right group).")
    parser.add_argument("--skills-mode", default="no_skills",
                        choices=["skills", "no_skills"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to compare (default: all found, ordered as gpt-4o, gpt-5.1, gpt-5.2, gpt-5.5).")
    args = parser.parse_args()

    df = load_results_df()
    if df.empty:
        print("No data found in Results/dashboard_data/")
        return

    df["Correct"] = df["Correct"].astype(float)
    requested_models = models_in_order(args.models if args.models else df["Model"].unique())

    results = []
    for mode in [args.mode_a, args.mode_b]:
        base_mask = (
            (df["image_type"] == args.image_type)
            & (df["blur_level"] == "heavy_blur")
            & (df["skills_mode"] == args.skills_mode)
            & (df["Model"].isin(requested_models))
        )
        if "reasoning_mode" in df.columns:
            base_mask &= (df["reasoning_mode"] == mode) | (df["Model"] == "gpt-4o")
        sub = df[base_mask].drop_duplicates(subset=["image_num", "question_num", "Model"])
        for model in requested_models:
            valid = sub[sub["Model"] == model]["Correct"].dropna()
            acc = valid.mean() if len(valid) > 0 else np.nan
            results.append((model, mode, acc))

    if not results or not any(not np.isnan(acc) for _, _, acc in results):
        print("No data found.")
        return

    labels = [f"{model}\n({mode})" for model, mode, _ in results]
    values = [0.0 if np.isnan(acc) else acc * 100 for _, _, acc in results]
    bar_colors = [MODEL_COLORS.get(model, "#0072B2") for model, _, _ in results]
    unique_models = models_in_order(m for m, _, _ in results)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    bars = ax.bar(labels, values, color=bar_colors, alpha=0.85)
    ax.set_title(
        f"Heavy Blur: {args.mode_a} vs {args.mode_b} Reasoning\n"
        f"({args.image_type}, {args.skills_mode})",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[Patch(color=MODEL_COLORS.get(m, "#0072B2"), label=m) for m in unique_models],
        title="Model", fontsize=10,
    )
    for bar, (_, _, acc) in zip(bars, results):
        if not np.isnan(acc):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.axvline(x=len(requested_models) - 0.5, color="red", linestyle=":", linewidth=2)
    fig.tight_layout()

    out_dir = os.path.normpath(os.path.join(
        OUT_BASE,
        f"{args.image_type}_heavy_blur_{args.mode_a}_vs_{args.mode_b}_{args.skills_mode}",
    ))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "heavy_blur_reasoning_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
