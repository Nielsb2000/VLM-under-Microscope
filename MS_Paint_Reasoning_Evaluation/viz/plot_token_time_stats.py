"""
Token usage and elapsed time statistics for a specific run configuration.
Reads Input Tokens, Output Tokens, and Elapsed Time from Results/dashboard_data/ via load_results_df().

Usage (from project root):
    uv run python MS_Paint_Reasoning_Evaluation/viz/plot_token_time_stats.py \
        --image-type color --blur-level heavy_blur --reasoning-mode medium --skills-mode no_skills
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from json_results_to_df import load_results_df

OUT_BASE = os.path.join(os.path.dirname(__file__), "..", "Results", "res_vis")
MODEL_ORDER = ["gpt-4o", "gpt-5.1", "gpt-5.2", "gpt-5.5"]

MODEL_COLORS = {
    "gpt-4o":  "#56B4E9",
    "gpt-5.1": "#E69F00",
    "gpt-5.2": "#009E73",
    "gpt-5.5": "#CC79A7",
}

# Pricing per token in USD. Leave models out when pricing is unavailable/uncertain;
# those models will still appear in token/time charts, but no cost label is shown.
PRICING = {
    "gpt-4o":  {"input": 5.00 / 1e6, "output": 15.00 / 1e6},
    "gpt-5.1": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-5.2": {"input": 12.00 / 1e6, "output": 36.00 / 1e6},
    # "gpt-5.5": intentionally omitted unless you want to pin a verified price.
}
USD_TO_EUR = 0.92


def model_sort_key(model: str) -> tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(MODEL_ORDER), model)


def models_in_order(models) -> list[str]:
    return sorted(set(models), key=model_sort_key)


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
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to specific models (default: all found, ordered as gpt-4o, gpt-5.1, gpt-5.2, gpt-5.5).")
    args = parser.parse_args()

    df = load_results_df()
    if df.empty:
        print("No data found in Results/dashboard_data/")
        return

    requested_models = models_in_order(args.models if args.models else df["Model"].unique())
    base_mask = (
        (df["image_type"] == args.image_type)
        & (df["blur_level"] == args.blur_level)
        & (df["skills_mode"] == args.skills_mode)
        & (df["Model"].isin(requested_models))
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
    models = models_in_order(sub["Model"].unique())

    n = len(imgq_labels)
    n_m = len(models)
    imgq_idx = {lbl: i for i, lbl in enumerate(imgq_labels)}

    input_mat = np.zeros((n, n_m))
    output_mat = np.zeros((n, n_m))
    total_mat = np.zeros((n, n_m))
    time_mat = np.zeros((n, n_m))
    cost_mat = np.full((n, n_m), np.nan)

    for _, row in sub.iterrows():
        r = imgq_idx[row["imgq"]]
        c = models.index(row["Model"])
        it = float(row["Input Tokens"] or 0)
        ot = float(row["Output Tokens"] or 0)
        input_mat[r, c] = it
        output_mat[r, c] = ot
        total_mat[r, c] = it + ot
        time_mat[r, c] = float(row["Elapsed Time"] or 0)
        pr = PRICING.get(row["Model"])
        if pr is not None:
            cost_mat[r, c] = (it * pr["input"] + ot * pr["output"]) * USD_TO_EUR

    bar_width = min(0.85 / max(n_m, 1), 0.25)
    x = np.arange(n)
    token_types = [("Input",  input_mat,  "#56B4E9"),
                   ("Output", output_mat, "#E69F00"),
                   ("Total",  total_mat,  "#009E73")]

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n * 1.2), 12), sharex=True)

    sub_bar_width = bar_width / len(token_types)
    group_width = n_m * bar_width
    for i, model in enumerate(models):
        for t, (label, mat, color) in enumerate(token_types):
            offset = -group_width / 2 + i * bar_width + t * sub_bar_width + sub_bar_width / 2
            bars = axes[0].bar(
                x + offset, mat[:, i],
                width=sub_bar_width,
                label=f"{model} {label}", color=color, alpha=0.8,
            )
            if label == "Total":
                for j, bar in enumerate(bars):
                    cost = cost_mat[j, i]
                    if mat[j, i] > 0 and not np.isnan(cost):
                        axes[0].text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.02,
                            f"EUR {cost:.3f}",
                            ha="center", va="bottom", fontsize=7, rotation=90,
                        )
    axes[0].set_ylabel("Tokens")
    axes[0].set_title(
        f"Token Usage - {args.image_type}, {args.blur_level}, "
        f"reasoning={args.reasoning_mode}, {args.skills_mode}\n"
        "(Cost labels shown only for models with configured pricing)"
    )
    axes[0].legend(fontsize=8, ncol=max(1, min(3, n_m)))

    for i, model in enumerate(models):
        offset = -group_width / 2 + i * bar_width + bar_width / 2
        axes[1].bar(x + offset, time_mat[:, i], width=bar_width,
                    label=model, color=MODEL_COLORS.get(model, "#0072B2"))
    axes[1].set_ylabel("Elapsed Time (s)")
    axes[1].set_title("Elapsed Time per Image/Question")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(imgq_labels, rotation=90, fontsize=8)
    axes[1].legend()

    plt.tight_layout()

    tag = f"{args.image_type}_{args.blur_level}_{args.reasoning_mode}_{args.skills_mode}"
    out_dir = os.path.normpath(os.path.join(OUT_BASE, tag))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "token_time_stats.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
