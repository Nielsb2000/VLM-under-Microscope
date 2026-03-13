"""
All models × all blur levels accuracy bar chart (with dotted separators between blur groups).
Reads from Results/dashboard_data/ via load_results_df().

Usage (from project root):
    uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_all_conditions.py \\
        --image-type color --reasoning-mode medium --skills-mode no_skills
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from json_results_to_df import load_results_df

OUT_BASE = os.path.join(os.path.dirname(__file__), "..", "Results", "res_vis")
BLUR_LEVELS = ["no_blur", "med_blur", "heavy_blur"]


def main():
    parser = argparse.ArgumentParser(
        description="Overall model accuracy across all blur levels in one chart.")
    parser.add_argument("--image-type", default="color",
                        choices=["color", "greyscale", "inverted_greyscale"])
    parser.add_argument("--reasoning-mode", default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("--skills-mode", default="no_skills",
                        choices=["skills", "no_skills"])
    args = parser.parse_args()

    df = load_results_df()
    if df.empty:
        print("No data found in Results/dashboard_data/")
        return

    df["Correct"] = df["Correct"].astype(float)
    models = sorted(df["Model"].unique())

    accuracies = []
    for blur in BLUR_LEVELS:
        base_mask = (
            (df["image_type"] == args.image_type)
            & (df["blur_level"] == blur)
            & (df["skills_mode"] == args.skills_mode)
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
    values = [0.0 if np.isnan(acc) else acc for _, _, acc in accuracies]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"] * len(BLUR_LEVELS)

    plt.figure(figsize=(max(10, len(labels) * 0.9), 6))
    bars = plt.bar(labels, values, color=colors[:len(labels)], alpha=0.85)
    plt.title(
        f"Overall Model Accuracy — All Blur Levels\n"
        f"({args.image_type}, reasoning={args.reasoning_mode}, {args.skills_mode})",
    )
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    for bar, (_, _, acc) in zip(bars, accuracies):
        if not np.isnan(acc):
            plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                     f"{acc * 100:.1f}%", ha="center", fontsize=9)

    for b in range(1, len(BLUR_LEVELS)):
        plt.axvline(x=b * len(models) - 0.5, color="red", linestyle=":", linewidth=2)

    plt.tight_layout()
    out_dir = os.path.normpath(
        os.path.join(OUT_BASE, f"{args.image_type}_{args.reasoning_mode}_{args.skills_mode}")
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "accuracy_all_blur.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
