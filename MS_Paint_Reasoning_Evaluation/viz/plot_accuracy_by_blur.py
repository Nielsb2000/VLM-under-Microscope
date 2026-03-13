"""
Bar chart comparing model accuracy across blur levels for a fixed image_type + reasoning_mode.
Reads from Results/dashboard_data/ via load_results_df().

Usage (from project root):
    uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_by_blur.py \\
        --image-type color --blur-levels no_blur med_blur heavy_blur \\
        --reasoning-mode medium --skills-mode no_skills
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from json_results_to_df import load_results_df

OUT_BASE = os.path.join(os.path.dirname(__file__), "..", "Results", "res_vis")


def main():
    parser = argparse.ArgumentParser(
        description="Model accuracy comparison across blur levels.")
    parser.add_argument("--image-type", default="color",
                        choices=["color", "greyscale", "inverted_greyscale"])
    parser.add_argument("--blur-levels", nargs="+",
                        default=["no_blur", "med_blur", "heavy_blur"],
                        choices=["no_blur", "med_blur", "heavy_blur"])
    parser.add_argument("--reasoning-mode", default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("--skills-mode", default="no_skills",
                        choices=["skills", "no_skills"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to specific models (default: all found).")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    df = load_results_df()
    if df.empty:
        print("No data found in Results/dashboard_data/")
        return

    df["Correct"] = df["Correct"].astype(float)

    accuracies = []
    for blur in args.blur_levels:
        base_mask = (
            (df["image_type"] == args.image_type)
            & (df["blur_level"] == blur)
            & (df["skills_mode"] == args.skills_mode)
        )
        if "reasoning_mode" in df.columns:
            base_mask &= (df["reasoning_mode"] == args.reasoning_mode) | (df["Model"] == "gpt-4o")
        sub = df[base_mask].drop_duplicates(subset=["image_num", "question_num", "Model"])
        models = args.models if args.models else sorted(sub["Model"].unique())
        for model in models:
            valid = sub[sub["Model"] == model]["Correct"].dropna()
            acc = valid.mean() if len(valid) > 0 else np.nan
            accuracies.append((blur, model, acc))

    if not accuracies:
        print("No accuracy data found. Check that Results/dashboard_data/ is populated.")
        return

    unique_models = list(dict.fromkeys(m for _, m, _ in accuracies))
    cmap = plt.get_cmap("tab10")
    model_colors = {m: cmap(i % 10) for i, m in enumerate(unique_models)}

    labels = [f"{model}\n({blur})" for blur, model, _ in accuracies]
    values = [acc for _, _, acc in accuracies]
    bar_colors = [model_colors[model] for _, model, _ in accuracies]

    plt.figure(figsize=(max(12, len(labels) * 1.3), 7))
    bars = plt.bar(labels, values, color=bar_colors, alpha=0.85)
    plt.title(
        f"Model Accuracy by Blur Level\n({args.image_type}, reasoning={args.reasoning_mode}, "
        f"{args.skills_mode})",
        fontsize=14,
    )
    plt.ylabel("Accuracy", fontsize=13)
    plt.ylim(0, 1.15)
    plt.xticks(fontsize=11, rotation=25, ha="right")
    plt.legend(
        handles=[Patch(color=model_colors[m], label=m) for m in unique_models],
        title="Model", fontsize=11, title_fontsize=12,
    )
    for i, (bar, (_, _, acc)) in enumerate(zip(bars, accuracies)):
        if not np.isnan(acc):
            plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                     f"{acc * 100:.1f}%", ha="center", fontsize=11)

    # Separator lines between blur level groups
    for b in range(1, len(args.blur_levels)):
        plt.axvline(x=b * len(unique_models) - 0.5, color="red", linestyle=":", linewidth=2)

    plt.tight_layout()
    tag = f"{args.image_type}_{args.reasoning_mode}_{args.skills_mode}"
    out_dir = args.output_dir or os.path.normpath(os.path.join(OUT_BASE, tag))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "accuracy_by_blur.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
