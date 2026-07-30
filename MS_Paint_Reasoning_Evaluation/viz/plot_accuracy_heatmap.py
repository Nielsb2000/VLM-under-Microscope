"""
Per-run heatmap + accuracy bar chart for a specific (image_type, blur_level, reasoning_mode, skills_mode).
Reads from Results/dashboard_data/ via load_results_df().

Usage (from project root):
    uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heatmap.py \
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
    parser = argparse.ArgumentParser(description="Per-run accuracy heatmap and bar chart.")
    parser.add_argument("--image-type", default="color",
                        choices=["color", "greyscale", "inverted_greyscale"])
    parser.add_argument("--blur-level", default="no_blur",
                        choices=["no_blur", "med_blur", "heavy_blur"])
    parser.add_argument("--reasoning-mode", default="medium",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort (gpt-4o is always included regardless of this setting).")
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
    requested_models = models_in_order(args.models if args.models else df["Model"].unique())

    base_mask = (
        (df["image_type"] == args.image_type)
        & (df["blur_level"] == args.blur_level)
        & (df["skills_mode"] == args.skills_mode)
        & (df["Model"].isin(requested_models))
    )
    if "reasoning_mode" in df.columns:
        reasoning_mask = (df["reasoning_mode"] == args.reasoning_mode) | (df["Model"] == "gpt-4o")
        mask = base_mask & reasoning_mask
    else:
        mask = base_mask
    sub = df[mask].drop_duplicates(subset=["image_num", "question_num", "Model"])

    if sub.empty:
        print(f"No data for: image_type={args.image_type}, blur={args.blur_level}, "
              f"reasoning={args.reasoning_mode}, skills={args.skills_mode}")
        return

    models = models_in_order(sub["Model"].unique())
    tag = f"{args.image_type}_{args.blur_level}_{args.reasoning_mode}_{args.skills_mode}"
    out_dir = os.path.normpath(os.path.join(OUT_BASE, tag))
    os.makedirs(out_dir, exist_ok=True)

    accuracies = {}
    for model in models:
        valid = sub[sub["Model"] == model]["Correct"].dropna()
        accuracies[model] = valid.mean() * 100 if len(valid) > 0 else 0.0

    bar_colors = [MODEL_COLORS.get(m, "#0072B2") for m in models]
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2), 5))
    ax.bar(list(accuracies.keys()), list(accuracies.values()),
           color=bar_colors, edgecolor="white", linewidth=1.2)
    ax.set_title(
        f"MS Paint Accuracy - {args.image_type}, {args.blur_level}\n"
        f"reasoning: {args.reasoning_mode}, {args.skills_mode}",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (m, acc) in enumerate(accuracies.items()):
        ax.text(i, acc + 1.5, f"{acc:.1f}%", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    bar_path = os.path.join(out_dir, "model_accuracy.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {bar_path}")

    imgs = sorted(sub["image_num"].unique())
    qs = sorted(sub["question_num"].unique())
    for model in models:
        mdf = sub[sub["Model"] == model]
        hmap = np.full((len(imgs), len(qs)), np.nan)
        for i, img in enumerate(imgs):
            for j, q in enumerate(qs):
                row = mdf[(mdf["image_num"] == img) & (mdf["question_num"] == q)]
                if not row.empty:
                    hmap[i, j] = row["Correct"].values[0]

        fig, ax = plt.subplots(figsize=(len(qs) + 2, len(imgs)))
        cmap = plt.cm.Greens.copy()
        cmap.set_bad(color="black")
        ax.imshow(np.ma.masked_invalid(hmap), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{model} Correctness\n({args.image_type}, {args.blur_level}, "
                     f"reasoning={args.reasoning_mode}, {args.skills_mode})")
        ax.set_xlabel("Question #")
        ax.set_ylabel("Image #")
        ax.set_xticks(np.arange(len(qs)))
        ax.set_xticklabels([str(q) for q in qs])
        ax.set_yticks(np.arange(len(imgs)))
        ax.set_yticklabels([str(img) for img in imgs])
        for i in range(len(imgs)):
            for j in range(len(qs)):
                v = hmap[i, j]
                if not np.isnan(v):
                    ax.text(j, i, str(int(v)), ha="center", va="center", color="black")
        ax.legend(
            handles=[
                Patch(facecolor="green", edgecolor="black", label="Correct (1)"),
                Patch(facecolor="white", edgecolor="black", label="Incorrect (0)"),
                Patch(facecolor="black", edgecolor="black", label="No answer"),
            ],
            loc="upper right", bbox_to_anchor=(1.15, 1),
        )
        plt.tight_layout()
        safe_model = model.replace("/", "_")
        hmap_path = os.path.join(out_dir, f"{safe_model}_heatmap.png")
        plt.savefig(hmap_path, dpi=150)
        plt.close()
        print(f"Saved: {hmap_path}")


if __name__ == "__main__":
    main()
