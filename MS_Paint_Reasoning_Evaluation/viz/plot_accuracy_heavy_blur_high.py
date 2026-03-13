"""
Heavy blur accuracy comparison between two reasoning modes (e.g. medium vs high).
Useful for showing the effect of increased reasoning effort on the hardest blur condition.
Reads from Results/dashboard_data/ via load_results_df().

Usage (from project root):
    uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heavy_blur_high.py \\
        --image-type color --mode-a medium --mode-b high --skills-mode no_skills
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from json_results_to_df import load_results_df

OUT_BASE = os.path.join(os.path.dirname(__file__), "..", "Results", "res_vis")


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
    parser.add_argument("--models", nargs="+", default=["gpt-5.1", "gpt-5.2"],
                        help="Models to compare (default: gpt-5.1 gpt-5.2).")
    args = parser.parse_args()

    df = load_results_df()
    if df.empty:
        print("No data found in Results/dashboard_data/")
        return

    df["Correct"] = df["Correct"].astype(float)

    results = []
    for mode in [args.mode_a, args.mode_b]:
        base_mask = (
            (df["image_type"] == args.image_type)
            & (df["blur_level"] == "heavy_blur")
            & (df["skills_mode"] == args.skills_mode)
        )
        if "reasoning_mode" in df.columns:
            base_mask &= df["reasoning_mode"] == mode
        sub = df[base_mask].drop_duplicates(subset=["image_num", "question_num", "Model"])
        for model in args.models:
            valid = sub[sub["Model"] == model]["Correct"].dropna()
            acc = valid.mean() if len(valid) > 0 else np.nan
            results.append((model, mode, acc))

    if not results:
        print("No data found.")
        return

    labels = [f"{model}\n({mode})" for model, mode, _ in results]
    values = [0.0 if np.isnan(acc) else acc for _, _, acc in results]
    # Alternate colors per model across both mode groups
    model_colors = {"gpt-5.1": "#ff7f0e", "gpt-5.2": "#2ca02c"}
    bar_colors = [model_colors.get(model, "#1f77b4") for model, _, _ in results]

    plt.figure(figsize=(max(8, len(labels) * 1.5), 6))
    bars = plt.bar(labels, values, color=bar_colors, alpha=0.85)
    plt.title(
        f"Heavy Blur: {args.mode_a} vs {args.mode_b} Reasoning\n"
        f"({args.image_type}, {args.skills_mode})",
    )
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                 f"{h * 100:.1f}%", ha="center")
    # Separator between mode groups
    plt.axvline(x=len(args.models) - 0.5, color="red", linestyle=":", linewidth=2)
    plt.tight_layout()

    out_dir = os.path.normpath(os.path.join(
        OUT_BASE,
        f"{args.image_type}_heavy_blur_{args.mode_a}_vs_{args.mode_b}_{args.skills_mode}",
    ))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "heavy_blur_reasoning_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
