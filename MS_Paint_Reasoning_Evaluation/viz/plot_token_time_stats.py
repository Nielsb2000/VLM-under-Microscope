"""
Token usage and elapsed time statistics for a specific run configuration.
Reads Input Tokens, Output Tokens, and Elapsed Time from Results/dashboard_data/ via load_results_df().

Usage (from project root):
    uv run python MS_Paint_Reasoning_Evaluation/viz/plot_token_time_stats.py \\
        --image-type color --blur-level heavy_blur --reasoning-mode medium --skills-mode no_skills
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from json_results_to_df import load_results_df

OUT_BASE = os.path.join(os.path.dirname(__file__), "..", "Results", "res_vis")

# Pricing per token in USD (Feb 2026)
PRICING = {
    "gpt-4o":  {"input": 5.00 / 1e6, "output": 15.00 / 1e6},
    "gpt-5.1": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-5.2": {"input": 12.00 / 1e6, "output": 36.00 / 1e6},
}
USD_TO_EUR = 0.92


def main():
    parser = argparse.ArgumentParser(
        description="Token usage and elapsed time stats for a specific run configuration.")
    parser.add_argument("--image-type", default="color",
                        choices=["color", "greyscale", "inverted_greyscale"])
    parser.add_argument("--blur-level", default="no_blur",
                        choices=["no_blur", "med_blur", "heavy_blur"])
    parser.add_argument("--reasoning-mode", default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("--skills-mode", default="no_skills",
                        choices=["skills", "no_skills"])
    args = parser.parse_args()

    df = load_results_df()
    if df.empty:
        print("No data found in Results/dashboard_data/")
        return

    base_mask = (
        (df["image_type"] == args.image_type)
        & (df["blur_level"] == args.blur_level)
        & (df["skills_mode"] == args.skills_mode)
    )
    if "reasoning_mode" in df.columns:
        base_mask &= (df["reasoning_mode"] == args.reasoning_mode) | (df["Model"] == "gpt-4o")
    sub = df[base_mask].drop_duplicates(subset=["image_num", "question_num", "Model"])
    sub = sub.dropna(subset=["Input Tokens", "Output Tokens", "Elapsed Time"])

    if sub.empty:
        print(
            f"No token/time data for: image_type={args.image_type}, blur={args.blur_level}, "
            f"reasoning={args.reasoning_mode}, skills={args.skills_mode}\n"
            "Token data is only available if results were generated with token recording enabled."
        )
        return

    sub = sub.copy()
    sub["imgq"] = sub.apply(
        lambda r: f"img{int(r['image_num'])}/q{int(r['question_num'])}", axis=1
    )
    imgq_labels = sorted(sub["imgq"].unique())
    models = sorted(sub["Model"].unique())

    n = len(imgq_labels)
    n_m = len(models)
    imgq_idx = {lbl: i for i, lbl in enumerate(imgq_labels)}

    input_mat = np.zeros((n, n_m))
    output_mat = np.zeros((n, n_m))
    total_mat = np.zeros((n, n_m))
    time_mat = np.zeros((n, n_m))
    cost_mat = np.zeros((n, n_m))

    for _, row in sub.iterrows():
        r = imgq_idx[row["imgq"]]
        c = models.index(row["Model"])
        it = float(row["Input Tokens"] or 0)
        ot = float(row["Output Tokens"] or 0)
        input_mat[r, c] = it
        output_mat[r, c] = ot
        total_mat[r, c] = it + ot
        time_mat[r, c] = float(row["Elapsed Time"] or 0)
        pr = PRICING.get(row["Model"], {"input": 0, "output": 0})
        cost_mat[r, c] = (it * pr["input"] + ot * pr["output"]) * USD_TO_EUR

    bar_width = 0.25
    x = np.arange(n)
    token_types = [("Input", input_mat, "#1f77b4"),
                   ("Output", output_mat, "#ff7f0e"),
                   ("Total", total_mat, "#2ca02c")]

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n * 1.2), 12), sharex=True)

    for i, model in enumerate(models):
        for t, (label, mat, color) in enumerate(token_types):
            offset = (i * len(token_types) + t) * bar_width / len(token_types)
            bars = axes[0].bar(
                x + offset, mat[:, i],
                width=bar_width / len(token_types),
                label=f"{model} {label}", color=color, alpha=0.8,
            )
            if label == "Total":
                for j, bar in enumerate(bars):
                    cost = cost_mat[j, i]
                    if mat[j, i] > 0:
                        axes[0].text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.02,
                            f"€{cost:.3f}",
                            ha="center", va="bottom", fontsize=7, rotation=90,
                        )
    axes[0].set_ylabel("Tokens")
    axes[0].set_title(
        f"Token Usage — {args.image_type}, {args.blur_level}, "
        f"reasoning={args.reasoning_mode}, {args.skills_mode}\n"
        "(Cost in EUR above total bars)"
    )
    axes[0].legend(fontsize=8)

    for i, model in enumerate(models):
        axes[1].bar(x + i * bar_width, time_mat[:, i], width=bar_width, label=model)
    axes[1].set_ylabel("Elapsed Time (s)")
    axes[1].set_title("Elapsed Time per Image/Question")
    axes[1].set_xticks(x + bar_width)
    axes[1].set_xticklabels(imgq_labels, rotation=90, fontsize=8)
    axes[1].legend()

    plt.tight_layout()

    tag = f"{args.image_type}_{args.blur_level}_{args.reasoning_mode}_{args.skills_mode}"
    out_dir = os.path.normpath(os.path.join(OUT_BASE, tag))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "token_time_stats.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
