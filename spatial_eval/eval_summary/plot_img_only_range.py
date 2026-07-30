"""plot_img_only_range.py — Test 2: img-only range learning curve.

Shows how accuracy changes as more example images are provided in the skill:
  n10 (10 imgs) → n30 (30 imgs) → n50 (50 imgs) → n100 (100 imgs)

Produces:
  - One bar-chart PNG per task (mazenav, spatialgrid, spatialmap)
  - One combined 3-panel PNG

The compared variants also include the original img-only (3 separate images,
end-of-dataset) and baseline for reference.

Usage (run from project root):
  uv run python spatial_eval/eval_summary/plot_img_only_range.py \
      --date 20260316 20260317 \
      --out_dir spatial_eval/eval_summary/result_vis
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASKS = ["mazenav", "spatialgrid", "spatialmap"]
TASK_DISPLAY = {
    "mazenav":     "MazeNav",
    "spatialgrid": "Spatial Grid",
    "spatialmap":  "Spatial Map",
}

# Ordered variants for x-axis
VARIANTS = [
    ("baseline",            "Baseline\n(no skills)",    "#555555"),  # dark grey
    ("img-only",            "Img n-3",                  "#56B4E9"),  # sky blue
    ("img-only-annotated",  "Img Only\nn3 Annotated",  "#3A8FC0"),  # medium blue
    ("img-only-n10",        "Img-Only\n(n=10)",         "#0072B2"),  # deep blue
    ("img-only-n30",        "Img-Only\n(n=30)",         "#332288"),  # indigo
]

MODELS = [
    ("gpt-5.2", "GPT-5.2"),
    ("gpt-5.5", "GPT-5.5"),
]

_OUTPUTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_check_answer():
    _eval_dir = os.path.join(os.path.dirname(__file__), "..", "evals")
    if _eval_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(_eval_dir))
    from evaluation import _check_answer  # noqa: E402
    return _check_answer


def _file_accuracy(path, check_fn) -> float:
    correct = total = 0
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            c, _ = check_fn(row.get("answer", ""), row)
            correct += c
            total += 1
    return correct / total if total > 0 else 0.0


def _model_of(fname: str) -> str:
    """Return 'gpt-5.5' or 'gpt-5.2' based on filename."""
    if "gpt-5.5" in fname:
        return "gpt-5.5"
    return "gpt-5.2"


def _is_mc_file(fname: str) -> bool:
    return bool(re.search(r'_mc\d{2}s\d+_', fname))


def _classify(fname: str) -> str | None:
    # Order matters: check more specific patterns first
    for n in ("n30", "n10", "n3"):
        if f"_skills_img-only-{n}_" in fname:
            return f"img-only-{n}"
    if "_skills_img-only-annotated_" in fname:
        return "img-only-annotated"
    if "_skills_img-only_" in fname:
        return "img-only"
    if "_skills_" in fname:
        return None  # other skill variants — ignore
    return "baseline"


def collect_accuracies(
    jsonl_dir: str,
    dates: list[str] | None,
    check_fn,
) -> dict[str, dict[str, list[float]]]:
    """Collect accuracies split by model and variant.

    Returns {model_key: {variant_key: [acc, …]}}.
    """
    result: dict[str, dict[str, list[float]]] = {
        m: {v[0]: [] for v in VARIANTS} for m, _ in MODELS
    }
    variant_keys = {v[0] for v in VARIANTS}

    if not os.path.isdir(jsonl_dir):
        return result

    for fname in sorted(os.listdir(jsonl_dir)):
        if not fname.endswith(".jsonl"):
            continue
        if ".timing.json" in fname:
            continue
        # Removed MC tag requirement: include all .jsonl files
        if dates and not any(d in fname for d in dates):
            continue
        variant = _classify(fname)
        if variant is None or variant not in variant_keys:
            continue
        model = _model_of(fname)
        if model not in result:
            continue
        acc = _file_accuracy(os.path.join(jsonl_dir, fname), check_fn)
        result[model][variant].append(acc)

    return result


def _mean_sd(values: list[float]):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_task(
    task: str,
    dates: list[str] | None,
    out_dir: str,
    jsonl_root: str,
    check_fn,
) -> dict | None:
    jsonl_dir = os.path.join(jsonl_root, "MilaWang__SpatialEval", "vqa", task)
    accs_by_model = collect_accuracies(jsonl_dir, dates, check_fn)

    display = TASK_DISPLAY[task]
    keys   = [v[0] for v in VARIANTS]
    labels = [v[1] for v in VARIANTS]

    # Compute stats per (model, variant)
    stats: dict[str, dict[str, tuple]] = {}
    for model, model_label in MODELS:
        stats[model] = {}
        for k in keys:
            m, s = _mean_sd(accs_by_model[model][k])
            n = len(accs_by_model[model][k])
            stats[model][k] = (m, s, n)
            n_str = f"n={n}" if n else "no data"
            m_str = f"{m*100:.1f}%" if m is not None else "—"
            s_str = f"±{s*100:.1f}%" if s is not None else ""
            print(f"  [{model}] {task:14s} {k:17s}: {m_str} {s_str}  ({n_str})")

    if all(stats[m][k][0] is None for m, _ in MODELS for k in keys):
        print(f"  [WARN] No data found for {task} — skipping plot.")
        return None

    n_models = len(MODELS)
    bar_w = 0.38 / n_models
    offsets = np.linspace(-(n_models - 1) * bar_w / 2, (n_models - 1) * bar_w / 2, n_models)
    hatches = [None, "//"]
    x = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for mi, (model, _) in enumerate(MODELS):
        for i, (k, _, color) in enumerate(VARIANTS):
            m, s, n = stats[model][k]
            if m is None:
                continue
            ax.bar(x[i] + offsets[mi], m * 100, width=bar_w, color=color,
                   hatch=hatches[mi], edgecolor="white", linewidth=1.2, zorder=3)
            if s and s > 0:
                ax.errorbar(x[i] + offsets[mi], m * 100, yerr=s * 100, fmt="none",
                            ecolor="black", elinewidth=1.8, capsize=4, zorder=4)
            label_y = m * 100 + (s * 100 if s else 0) + 1.5
            ax.text(x[i] + offsets[mi], label_y, f"{m*100:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            if s and s > 0:
                ax.text(x[i] + offsets[mi], label_y + 4.5, f"±{s*100:.1f}%",
                        ha="center", va="bottom", fontsize=7.5, color="#555")



    n_runs = max((stats[m][k][2] for m, _ in MODELS for k in keys), default="?")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 130)
    ax.set_title(
        f"{display} — Image-Only Skill: Learning Curve (GPT-5.2 vs GPT-5.5, VQA, up to {n_runs} runs)\n"
        "Effect of increasing the number of in-skill example images  |  solid = GPT-5.2, hatched = GPT-5.5",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    import matplotlib.patches as mpatches
    variant_handles = [
        mpatches.Patch(color=c, label=l.replace("\n", " ")) for _, l, c in VARIANTS
    ]
    model_handles = [
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="white", label="GPT-5.2 (solid)"),
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="white", hatch="//", label="GPT-5.5 (hatched)"),
    ]
    ax.legend(handles=variant_handles + model_handles, fontsize=8.5, loc="upper right",
              framealpha=0.85, edgecolor="#ccc", ncol=2)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_img_only_range.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return stats


def plot_combined(all_results: dict, out_dir: str, n_runs: int):
    """3-panel figure, one panel per task."""
    # 18 cm wide (double-column journal), ~9 cm tall
    fig, axes = plt.subplots(1, 3, figsize=(18 / 2.54, 9 / 2.54), sharey=False)

    keys   = [v[0] for v in VARIANTS]
    labels = [v[1] for v in VARIANTS]
    x = np.arange(len(keys))

    n_models = len(MODELS)
    bar_w = 0.38 / n_models
    offsets = np.linspace(-(n_models - 1) * bar_w / 2, (n_models - 1) * bar_w / 2, n_models)
    hatches = [None, "//"]

    for ax, task in zip(axes, TASKS):
        task_stats = all_results.get(task)
        if task_stats is None:
            ax.set_title(TASK_DISPLAY[task], fontsize=7)
            continue

        for mi, (model, _) in enumerate(MODELS):
            for i, (k, _, color) in enumerate(VARIANTS):
                m, s, n = task_stats[model][k]
                if m is None:
                    continue
                ax.bar(x[i] + offsets[mi], m * 100, width=bar_w, color=color,
                       hatch=hatches[mi], edgecolor="white", linewidth=0.5, zorder=3)
                if s and s > 0:
                    ax.errorbar(x[i] + offsets[mi], m * 100, yerr=s * 100, fmt="none",
                                ecolor="black", elinewidth=0.8, capsize=2, zorder=4)



        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5, rotation=35, ha="right")
        ax.set_ylabel("Accuracy (%)", fontsize=6)
        ax.set_ylim(0, 100)
        ax.set_title(TASK_DISPLAY[task], fontsize=7, fontweight="bold", pad=3)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.4, zorder=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=5)

    import matplotlib.patches as mpatches
    variant_handles = [
        mpatches.Patch(color=c, label=l.replace("\n", " ")) for _, l, c in VARIANTS
    ]
    model_handles = [
        mpatches.Patch(facecolor="#888", edgecolor="white", label="GPT-5.2"),
        mpatches.Patch(facecolor="#888", edgecolor="white", hatch="//", label="GPT-5.5"),
    ]
    fig.legend(handles=variant_handles + model_handles, loc="lower center", ncol=4,
               fontsize=5.5, frameon=True, framealpha=0.9,
               edgecolor="#ccc", bbox_to_anchor=(0.5, -0.28))

    fig.suptitle("Image-only skill: accuracy vs. number of example images (mean ± SD)",
                 fontsize=7, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "all_tasks_img_only_range.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved combined → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot img-only range learning curve.")
    parser.add_argument("--date", nargs="+", default=None,
                        help="Filter files by YYYYMMDD date string(s).")
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "result_vis"),
                        help="Directory to save plots.")
    parser.add_argument("--jsonl_root", default=_OUTPUTS_ROOT,
                        help="Root folder containing inference outputs.")
    args = parser.parse_args()

    check_fn = _load_check_answer()

    print("=== Image-Only Range Test (Learning Curve) ===")
    all_results: dict[str, dict] = {}
    for task in TASKS:
        print(f"\n[{task}]")
        result = plot_task(task, args.date, args.out_dir, args.jsonl_root, check_fn)
        if result:
            all_results[task] = result

    if len(all_results) > 1:
        # Max run count across any (model, task, variant)
        n_runs = max(
            (task_stats[m][k][2] for task_stats in all_results.values()
             for m, _ in MODELS for k in [v[0] for v in VARIANTS]),
            default="?",
        )
        plot_combined(all_results, args.out_dir, n_runs)


if __name__ == "__main__":
    main()
