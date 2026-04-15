"""plot_img_only_tool.py — Img-Only-Tool scaling: accuracy vs N example images.

Compares:
  baseline          — bare no-skills MC runs
  img-only-tool-n3  — 3 example images via read_example tool (images only, no Q&A)
  img-only-tool-n10 — 10 example images via tool
  img-only-tool-n30 — 30 example images via tool

All conditions use offset_k=30 so no test sample overlaps with any example.
Each bar shows mean accuracy ± 1 SD across MC runs.  Δ annotations vs baseline.

Usage (from spatial_eval/):
    uv run python eval_summary/plot_img_only_tool.py \\
        --out_dir eval_summary/result_vis \\
        [--task mazenav] \\
        [--baseline_dates 20260323]
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
import matplotlib.patches as mpatches
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

# Okabe-Ito colorblind-safe palette; sequential blues = increasing N examples
CONDITIONS = [
    ("baseline",       "Baseline\n(no skills)",              "#555555"),  # dark grey
    ("img-only-n3",    "Img-Only tool\n(images only, n=3)",   "#56B4E9"),  # sky blue
    ("img-only-n10",   "Img-Only tool\n(images only, n=10)",  "#0072B2"),  # deep blue
    ("img-only-n30",   "Img-Only tool\n(images only, n=30)",  "#332288"),  # indigo
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
    """Map filename to condition key, or None to skip."""
    if "_skills_img-only-tool-n30_" in fname:
        return "img-only-n30"
    if "_skills_img-only-tool-n10_" in fname:
        return "img-only-n10"
    if "_skills_img-only-tool-n3_" in fname:
        return "img-only-n3"
    if "_skills_" in fname:
        return None   # other skill variants — skip
    return "baseline"


def collect_accuracies(
    jsonl_dir: str,
    baseline_dates: list[str] | None,
    check_fn,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {k: [] for k, _, _ in CONDITIONS}

    if not os.path.isdir(jsonl_dir):
        return result

    for fname in sorted(os.listdir(jsonl_dir)):
        if not fname.endswith(".jsonl"):
            continue

        cond = _classify(fname)
        if cond is None or cond not in result:
            continue

        if cond == "baseline":
            if not _is_mc_file(fname):
                continue
            if baseline_dates and not any(d in fname for d in baseline_dates):
                continue
        else:
            # img-only-tool conditions: require MC tag (these are MC runs)
            if not _is_mc_file(fname):
                continue

        acc = _file_accuracy(os.path.join(jsonl_dir, fname), check_fn)
        result[cond].append(acc)

    return result


def _mean_sd(values: list[float]):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


# ---------------------------------------------------------------------------
# Per-task plot
# ---------------------------------------------------------------------------
def plot_task(
    task: str,
    baseline_dates: list[str] | None,
    out_dir: str,
    jsonl_root: str,
    check_fn,
) -> tuple[list, list] | None:
    jsonl_dir = os.path.join(jsonl_root, "MilaWang__SpatialEval", "vqa", task)
    accs = collect_accuracies(jsonl_dir, baseline_dates, check_fn)

    display = TASK_DISPLAY[task]
    keys   = [c[0] for c in CONDITIONS]
    labels = [c[1] for c in CONDITIONS]
    colors = [c[2] for c in CONDITIONS]

    means, sds = [], []
    for k in keys:
        m, s = _mean_sd(accs[k])
        means.append(m)
        sds.append(s)
        n_str = f"n={len(accs[k])}" if accs[k] else "no data"
        m_str = f"{m*100:.1f}%" if m is not None else "—"
        s_str = f"±{s*100:.1f}%" if (s is not None and s > 0) else ""
        print(f"  {task:14s} {k:18s}: {m_str} {s_str}  ({n_str})")

    if all(m is None for m in means):
        print(f"  [WARN] No data found for {task} — skipping plot.")
        return None

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (k, label, color) in enumerate(CONDITIONS):
        m, s = means[i], sds[i]
        if m is None:
            ax.bar(x[i], 0, width=0.55, color=color, alpha=0.3,
                   edgecolor=color, linewidth=1.2, hatch="//", zorder=3)
            ax.text(x[i], 3, "no data", ha="center", va="bottom",
                    fontsize=8, color="#888", fontstyle="italic")
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

    # Delta annotations vs baseline
    if means[0] is not None:
        for i, (k, _, _) in enumerate(CONDITIONS):
            if k == "baseline" or means[i] is None:
                continue
            delta = means[i] - means[0]
            sign = "+" if delta >= 0 else ""
            col = "#009E73" if delta >= 0 else "#D55E00"
            bar_top = means[i] * 100 + (sds[i] * 100 if sds[i] else 0)
            ax.text(x[i], bar_top + 10,
                    f"Δ {sign}{delta*100:.1f}%",
                    ha="center", va="bottom", fontsize=9,
                    color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 130)
    ax.set_title(
        f"{display} — Img-Only Tool Scaling: Images via read_example Tool, No Q&A (GPT-5.2, VQA)\n"
        "Bars: mean ± 1 SD across MC runs  |  Δ vs baseline",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=color, label=label.replace("\n", " "))
        for _, label, color in CONDITIONS
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left",
              framealpha=0.85, edgecolor="#ccc")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_img_only_no_qa.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return means, sds


# ---------------------------------------------------------------------------
# Combined multi-task figure (only rendered when all tasks have data)
# ---------------------------------------------------------------------------
def plot_combined(all_results: dict, out_dir: str):
    tasks_with_data = [t for t in TASKS if t in all_results]
    if not tasks_with_data:
        return

    n_panels = len(tasks_with_data)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    keys   = [c[0] for c in CONDITIONS]
    labels = [c[1] for c in CONDITIONS]
    colors = [c[2] for c in CONDITIONS]
    x = np.arange(len(keys))

    for ax, task in zip(axes, tasks_with_data):
        means, sds = all_results[task]
        display = TASK_DISPLAY[task]

        for i, (k, _, color) in enumerate(CONDITIONS):
            m, s = means[i], sds[i]
            if m is None:
                ax.bar(x[i], 0, width=0.55, color=color, alpha=0.3,
                       edgecolor=color, linewidth=1.0, hatch="//", zorder=3)
                ax.text(x[i], 2, "n/a", ha="center", va="bottom",
                        fontsize=7, color="#888", fontstyle="italic")
                continue
            ax.bar(x[i], m * 100, width=0.55, color=color,
                   edgecolor="white", linewidth=1.2, zorder=3)
            if s and s > 0:
                ax.errorbar(x[i], m * 100, yerr=s * 100, fmt="none",
                            ecolor="black", elinewidth=1.6, capsize=4, zorder=4)
            ax.text(x[i], m * 100 + (s * 100 if s else 0) + 1.5,
                    f"{m*100:.1f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        if means[0] is not None:
            ax.axhline(means[0] * 100, color="#7f7f7f", linestyle="--",
                       linewidth=1.0, alpha=0.5, zorder=2)
            for i, (k, _, _) in enumerate(CONDITIONS):
                if k == "baseline" or means[i] is None:
                    continue
                delta = means[i] - means[0]
                sign = "+" if delta >= 0 else ""
                col = "#009E73" if delta >= 0 else "#D55E00"
                bar_top = means[i] * 100 + (sds[i] * 100 if sds[i] else 0)
                ax.text(x[i], bar_top + 5.5,
                        f"Δ {sign}{delta*100:.1f}%",
                        ha="center", va="bottom", fontsize=7.5,
                        color=col, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(0, 135)
        ax.set_title(display, fontsize=12, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=color, label=label.replace("\n", " "))
        for _, label, color in CONDITIONS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.85, edgecolor="#ccc",
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Img-Only Tool Scaling: Images via read_example Tool, No Q&A — All Tasks (GPT-5.2, VQA)\n"
        "Bars: mean ± 1 SD  |  Δ vs baseline",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "all_tasks_img_only_no_qa.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nCombined plot → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(args):
    check_fn = _load_check_answer()
    tasks = [args.task] if args.task else TASKS
    all_results = {}
    jsonl_root = _OUTPUTS_ROOT

    for task in tasks:
        print(f"\n=== {TASK_DISPLAY[task]} ===")
        result = plot_task(task, args.baseline_dates, args.out_dir, jsonl_root, check_fn)
        if result is not None:
            all_results[task] = result

    if len(all_results) > 1:
        plot_combined(all_results, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="eval_summary/result_vis")
    parser.add_argument("--task", default=None, choices=TASKS,
                        help="Single task to plot; omit for all available + combined.")
    parser.add_argument("--baseline_dates", default=None, nargs="+",
                        help="Date strings to filter bare MC baseline files "
                             "(e.g. 20260323). Omit to use all available MC files.")
    main(parser.parse_args())
