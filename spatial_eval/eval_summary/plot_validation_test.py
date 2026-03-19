"""plot_validation_test.py — Test 1: img-qa-val contamination validation.

Produces a 3-panel bar chart comparing:
  baseline   — no skills (MC runs)
  img-qa     — uncontaminated skill (MC runs)
  img-qa-val — SAME images in skill and test (single runs, expected ~100%)

Usage (run from project root):
  uv run python spatial_eval/eval_summary/plot_validation_test.py \
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

VARIANTS = [
    ("baseline",       "Baseline\n(no skills)",                   "#7f7f7f"),
    ("img-qa",         "Image + Q&A\n(uncontaminated)",            "#F28C38"),
    ("img-qa-val",     "Image + Q&A\n(same imgs — cheat)",         "#d62728"),
    ("img-qa-val-v2",  "Preload v2\n(tool-based lookup)",          "#1a7a1a"),
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
    if "_skills_img-qa-val-v2_" in fname:
        return "img-qa-val-v2"
    if "_skills_img-qa-val_" in fname:
        return "img-qa-val"
    if "_skills_img-qa_" in fname:
        return "img-qa"
    if "_skills_" in fname:
        return None  # other skill variants — ignore
    return "baseline"


def collect_accuracies(
    jsonl_dir: str,
    dates: list[str] | None,
    check_fn,
) -> dict[str, list[float]]:
    """Collect accuracies split by variant.

    For baseline and img-qa, only MC files (with _mc…_ tag) are included.
    For img-qa-val, non-MC files (single runs) are also included.
    """
    result: dict[str, list[float]] = {v[0]: [] for v in VARIANTS}

    if not os.path.isdir(jsonl_dir):
        return result

    for fname in sorted(os.listdir(jsonl_dir)):
        if not fname.endswith(".jsonl"):
            continue
        if dates and not any(d in fname for d in dates):
            continue
        variant = _classify(fname)
        if variant is None or variant not in result:
            continue
        # baseline and img-qa require MC tag; img-qa-val and img-qa-val-v2 accept both
        if variant not in ("img-qa-val", "img-qa-val-v2") and not _is_mc_file(fname):
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
        print(f"  {task:14s} {k:15s}: {m_str} {s_str}  ({n_str})")

    if all(m is None for m in means):
        print(f"  [WARN] No data found for {task} — skipping plot.")
        return None

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, (k, label, color) in enumerate(VARIANTS):
        m, s = means[i], sds[i]
        if m is None:
            continue
        ax.bar(x[i], m * 100, width=0.55, color=color,
               edgecolor="white", linewidth=1.2, zorder=3)
        if s and s > 0:
            ax.errorbar(x[i], m * 100, yerr=s * 100, fmt="none",
                        ecolor="black", elinewidth=1.8, capsize=5, zorder=4)
        label_y = m * 100 + (s * 100 if s else 0) + 1.5
        ax.text(x[i], label_y, f"{m*100:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        if s and s > 0:
            ax.text(x[i], label_y + 4.5, f"±{s*100:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color="#555")

    # Baseline reference line
    if means[0] is not None:
        ax.axhline(means[0] * 100, color="#7f7f7f", linestyle="--",
                   linewidth=1.2, alpha=0.6, zorder=2)

    # Delta vs baseline
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
                    ha="center", va="bottom", fontsize=9, color=col,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 130)
    ax.set_title(
        f"{display} — Contamination Validation (GPT-5.2, VQA, 10 imgs/q-type)\n"
        "Red bar: skill contains test images (expected ~100%)",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_validation_test.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return means, sds


def plot_combined(all_results: dict, out_dir: str):
    """3-panel figure comparing all tasks side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

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
            ax.bar(x[i], m * 100, width=0.55, color=color,
                   edgecolor="white", linewidth=1.2, zorder=3)
            if s and s > 0:
                ax.errorbar(x[i], m * 100, yerr=s * 100, fmt="none",
                            ecolor="black", elinewidth=1.8, capsize=5, zorder=4)
            label_y = m * 100 + (s * 100 if s else 0) + 1.5
            ax.text(x[i], label_y, f"{m*100:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        if means[0] is not None:
            ax.axhline(means[0] * 100, color="#7f7f7f", linestyle="--",
                       linewidth=1.0, alpha=0.6, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_ylim(0, 130)
        ax.set_title(f"{display}", fontsize=12, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Contamination Validation — img-qa-val vs img-qa vs Baseline (GPT-5.2, VQA)\n"
        "Red bar: skill contains exact test images (upper-bound check)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "all_tasks_validation_test.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved combined → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot contamination validation test results.")
    parser.add_argument("--date", nargs="+", default=None,
                        help="Filter files by YYYYMMDD date string(s).")
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "result_vis"),
                        help="Directory to save plots.")
    parser.add_argument("--jsonl_root", default=_OUTPUTS_ROOT,
                        help="Root folder containing inference outputs.")
    args = parser.parse_args()

    check_fn = _load_check_answer()

    print("=== Contamination Validation Test ===")
    all_results = {}
    for task in TASKS:
        print(f"\n[{task}]")
        result = plot_task(task, args.date, args.out_dir, args.jsonl_root, check_fn)
        if result:
            all_results[task] = result

    if len(all_results) > 1:
        plot_combined(all_results, args.out_dir)


if __name__ == "__main__":
    main()
