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
    ("baseline",       "Baseline\n(no skills)",    "#7f7f7f"),
    ("img-only",       "img-only\n(3 imgs)",        "#9ecae1"),
    ("img-only-n10",   "img-only\n(n=10)",          "#4292c6"),
    ("img-only-n30",   "img-only\n(n=30)",          "#2171b5"),
    ("img-only-n50",   "img-only\n(n=50)",          "#084594"),
    ("img-only-n100",  "img-only\n(n=100)",         "#08306b"),
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


def _is_mc_file(fname: str) -> bool:
    return bool(re.search(r'_mc\d{2}s\d+_', fname))


def _classify(fname: str) -> str | None:
    # Order matters: check more specific patterns first
    for n in ("n100", "n50", "n30", "n10"):
        if f"_skills_img-only-{n}_" in fname:
            return f"img-only-{n}"
    if "_skills_img-only_" in fname:
        return "img-only"
    if "_skills_" in fname:
        return None  # other skill variants — ignore
    return "baseline"


def collect_accuracies(
    jsonl_dir: str,
    dates: list[str] | None,
    check_fn,
) -> dict[str, list[float]]:
    """Collect MC accuracies split by variant."""
    result: dict[str, list[float]] = {v[0]: [] for v in VARIANTS}

    if not os.path.isdir(jsonl_dir):
        return result

    for fname in sorted(os.listdir(jsonl_dir)):
        if not fname.endswith(".jsonl"):
            continue
        if not _is_mc_file(fname):
            continue  # require MC tag for statistical estimates
        if dates and not any(d in fname for d in dates):
            continue
        variant = _classify(fname)
        if variant is None or variant not in result:
            continue
        acc = _file_accuracy(os.path.join(jsonl_dir, fname), check_fn)
        result[variant].append(acc)

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
) -> tuple | None:
    jsonl_dir = os.path.join(jsonl_root, "MilaWang__SpatialEval", "vqa", task)
    accs = collect_accuracies(jsonl_dir, dates, check_fn)

    display = TASK_DISPLAY[task]
    keys   = [v[0] for v in VARIANTS]
    labels = [v[1] for v in VARIANTS]
    colors = [v[2] for v in VARIANTS]

    means, sds, ns = [], [], []
    for k in keys:
        m, s = _mean_sd(accs[k])
        means.append(m)
        sds.append(s)
        ns.append(len(accs[k]))
        n_str = f"n={len(accs[k])}" if accs[k] else "no data"
        m_str = f"{m*100:.1f}%" if m is not None else "—"
        s_str = f"±{s*100:.1f}%" if s is not None else ""
        print(f"  {task:14s} {k:17s}: {m_str} {s_str}  ({n_str})")

    if all(m is None for m in means):
        print(f"  [WARN] No data found for {task} — skipping plot.")
        return None

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, (k, label, color) in enumerate(VARIANTS):
        m, s = means[i], sds[i]
        if m is None:
            continue
        ax.bar(x[i], m * 100, width=0.6, color=color,
               edgecolor="white", linewidth=1.2, zorder=3)
        if s and s > 0:
            ax.errorbar(x[i], m * 100, yerr=s * 100, fmt="none",
                        ecolor="black", elinewidth=1.8, capsize=5, zorder=4)
        label_y = m * 100 + (s * 100 if s else 0) + 1.5
        ax.text(x[i], label_y, f"{m*100:.1f}%",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        if s and s > 0:
            ax.text(x[i], label_y + 4.5, f"±{s*100:.1f}%",
                    ha="center", va="bottom", fontsize=8, color="#555")

    # Baseline reference line
    if means[0] is not None:
        ax.axhline(means[0] * 100, color="#7f7f7f", linestyle="--",
                   linewidth=1.2, alpha=0.7, zorder=2, label="Baseline")

    # Delta vs baseline annotations
    if means[0] is not None:
        for i, k in enumerate(keys):
            if k == "baseline" or means[i] is None:
                continue
            delta = means[i] - means[0]
            sign = "+" if delta >= 0 else ""
            col = "#1a7a1a" if delta >= 0 else "#d62728"
            bar_top = means[i] * 100 + (sds[i] * 100 if sds[i] else 0)
            ax.text(x[i], bar_top + 10,
                    f"Δ {sign}{delta*100:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color=col,
                    fontweight="bold")

    n_runs = max(ns) if ns else "?"
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 125)
    ax.set_title(
        f"{display} — Image-Only Skill: Learning Curve (GPT-5.2, VQA, {n_runs} MC runs × 10 imgs/q-type)\n"
        "Effect of increasing the number of in-skill example images on accuracy",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_img_only_range.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return means, sds


def plot_combined(all_results: dict, out_dir: str, n_runs: int):
    """3-panel figure, one panel per task."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5), sharey=False)

    keys   = [v[0] for v in VARIANTS]
    labels = [v[1] for v in VARIANTS]
    colors = [v[2] for v in VARIANTS]
    x = np.arange(len(keys))

    for ax, task in zip(axes, TASKS):
        result = all_results.get(task)
        if result is None:
            ax.set_title(f"{TASK_DISPLAY[task]}\n(no data)", fontsize=11)
            continue
        means, sds = result
        display = TASK_DISPLAY[task]

        for i, (k, _, color) in enumerate(VARIANTS):
            m, s = means[i], sds[i]
            if m is None:
                continue
            ax.bar(x[i], m * 100, width=0.6, color=color,
                   edgecolor="white", linewidth=1.2, zorder=3)
            if s and s > 0:
                ax.errorbar(x[i], m * 100, yerr=s * 100, fmt="none",
                            ecolor="black", elinewidth=1.8, capsize=5, zorder=4)
            label_y = m * 100 + (s * 100 if s else 0) + 1.5
            ax.text(x[i], label_y, f"{m*100:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        if means[0] is not None:
            ax.axhline(means[0] * 100, color="#7f7f7f", linestyle="--",
                       linewidth=1.0, alpha=0.7, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_ylim(0, 125)
        ax.set_title(f"{display}", fontsize=12, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Image-Only Skill: Learning Curve — effect of N example images on accuracy (GPT-5.2, VQA, {n_runs} MC runs)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "all_tasks_img_only_range.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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
    all_results: dict[str, tuple] = {}
    for task in TASKS:
        print(f"\n[{task}]")
        result = plot_task(task, args.date, args.out_dir, args.jsonl_root, check_fn)
        if result:
            all_results[task] = result

    if len(all_results) > 1:
        # Use the max number of MC runs found across any task/variant
        all_ns = [len(v) for r in all_results.values() for v in [r] if r]
        n_runs = max(all_ns, default="?")
        plot_combined(all_results, args.out_dir, n_runs)


if __name__ == "__main__":
    main()
