"""
Plot mean ± SD accuracy from Monte Carlo runs across skill variants and tasks.

For each (task, skill-variant) combination, locates all MC-iteration JSONL files
(filenames containing _mc[0-9]{2}s[0-9]+_), computes per-file accuracy, then
plots mean bars with SD error bars.

Also produces a combined multi-task figure.

Usage (from spatial_eval/):
    uv run python eval_summary/plot_mc_results.py \
        --eval_summary_dir eval_summary \
        --out_dir eval_summary/result_vis \
        [--date 20260316 20260317]  # optional: one or more YYYYMMDD dates
        [--task mazenav]            # optional: single task; omit for all 3
"""
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

# ── Config ─────────────────────────────────────────────────────────────────
TASKS = ["mazenav", "spatialgrid", "spatialmap"]
TASK_DISPLAY = {
    "mazenav":     "MazeNav",
    "spatialgrid": "Spatial Grid",
    "spatialmap":  "Spatial Map",
}

# Okabe-Ito colorblind-safe palette
VARIANTS = [
    ("baseline",    "Baseline\n(no skills)",       "#555555"),  # dark grey
    ("img-only",    "Image Only\nskill",            "#56B4E9"),  # sky blue
    ("img-qa",      "Img+Q&A\nskill (biased)",      "#E69F00"),  # orange
    ("img-context", "Img+Context\nskill (unbiased)","#009E73"),  # teal
]


def _classify(fname: str) -> str:
    if "_skills_img-only_" in fname:
        return "img-only"
    if "_skills_img-qa_" in fname:
        return "img-qa"
    if "_skills_img-context_" in fname:
        return "img-context"
    if "_skills_" in fname:
        return "full-skills"  # original full-skill runs — ignored
    return "baseline"


def _is_mc_file(fname: str) -> bool:
    """Return True if the filename contains an MC run tag (_mc00s42_…)."""
    return bool(re.search(r'_mc\d{2}s\d+_', fname))


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


def collect_mc_accuracies(jsonl_dir: str, dates: list[str] | None) -> dict[str, list[float]]:
    """
    Returns {variant_key: [acc_run0, acc_run1, …]} for all MC files in jsonl_dir.
    Non-MC files (no _mc…_ tag) are ignored.
    If dates is given, only files containing at least one of the YYYYMMDD strings are included.
    """
    check = _load_check_answer()
    result: dict[str, list[float]] = {v[0]: [] for v in VARIANTS}

    if not os.path.isdir(jsonl_dir):
        return result

    for fname in sorted(os.listdir(jsonl_dir)):
        if not fname.endswith(".jsonl"):
            continue

        if dates and not any(d in fname for d in dates):
            continue
        variant = _classify(fname)
        if variant not in result:
            continue
        acc = _file_accuracy(os.path.join(jsonl_dir, fname), check)
        result[variant].append(acc)

    return result


def _mean_sd(values: list[float]):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def plot_task(task: str, dates: list[str] | None, out_dir: str, jsonl_root: str):
    jsonl_dir = os.path.join(jsonl_root, "MilaWang__SpatialEval", "vqa", task)
    accs = collect_mc_accuracies(jsonl_dir, dates)

    display = TASK_DISPLAY[task]
    keys    = [v[0] for v in VARIANTS]
    labels  = [v[1] for v in VARIANTS]
    colors  = [v[2] for v in VARIANTS]

    means, sds, ns = [], [], []
    for k in keys:
        m, s = _mean_sd(accs[k])
        means.append(m)
        sds.append(s)
        ns.append(len(accs[k]))
        n_str = f"n={len(accs[k])}" if accs[k] else "no data"
        m_str = f"{m*100:.1f}%" if m is not None else "—"
        s_str = f"±{s*100:.1f}%" if s is not None else ""
        print(f"  {task:12s} {k:15s}: {m_str} {s_str}  ({n_str})")

    # skip plot if no MC data at all
    if all(m is None for m in means):
        print(f"  [WARN] No MC data found for {task} — skipping plot.")
        return None

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (k, label, color) in enumerate(VARIANTS):
        m, s = means[i], sds[i]
        if m is None:
            continue
        bar = ax.bar(x[i], m * 100, width=0.55, color=color,
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

    # baseline dashed reference
    if means[0] is not None:
        ax.axhline(means[0] * 100, color="#7f7f7f", linestyle="--",
                   linewidth=1.2, alpha=0.6, zorder=2)

    # delta annotations vs baseline
    if means[0] is not None:
        for i, k in enumerate(keys):
            if k == "baseline" or means[i] is None:
                continue
            delta = means[i] - means[0]
            sign = "+" if delta >= 0 else ""
            col = "#009E73" if delta >= 0 else "#D55E00"
            bar_top = means[i] * 100 + (sds[i] * 100 if sds[i] else 0)
            ax.text(x[i], bar_top + 10,
                    f"Δ {sign}{delta*100:.1f}%",
                    ha="center", va="bottom", fontsize=9, color=col,
                    fontweight="bold")

    n_runs = max(ns) if ns else "?"
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 120)
    ax.set_title(
        f"{display} — Image Skill Variants (GPT-5.2, VQA, {n_runs} MC runs)\n"
        "Bars: mean ± 1 SD across Monte Carlo subsets",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=color, label=label.replace("\n", " "))
        for _, label, color in VARIANTS
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left",
              framealpha=0.85, edgecolor="#ccc")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_mc_skill_variants.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return means, sds


def plot_combined(all_results: dict, out_dir: str, n_runs: int):
    """3-panel figure: one subplot per task."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    keys   = [v[0] for v in VARIANTS]
    labels = [v[1] for v in VARIANTS]
    colors = [v[2] for v in VARIANTS]
    x = np.arange(len(keys))

    for ax, task in zip(axes, TASKS):
        means, sds = all_results.get(task, ([None]*4, [None]*4))
        display = TASK_DISPLAY[task]

        for i, (k, _, color) in enumerate(VARIANTS):
            m, s = means[i], sds[i]
            if m is None:
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

        # Delta annotations vs baseline
        if means[0] is not None:
            for i, (k, _, _) in enumerate(VARIANTS):
                if k == "baseline" or means[i] is None:
                    continue
                delta = means[i] - means[0]
                sign = "+" if delta >= 0 else ""
                col = "#009E73" if delta >= 0 else "#D55E00"  # CB teal / vermillion
                bar_top = means[i] * 100 + (sds[i] * 100 if sds[i] else 0)
                ax.text(x[i], bar_top + 5.5,
                        f"Δ {sign}{delta*100:.1f}%",
                        ha="center", va="bottom", fontsize=8,
                        color=col, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(0, 130)
        ax.set_title(display, fontsize=12, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # Shared legend below the panels
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=color, label=label.replace("\n", " "))
        for _, label, color in VARIANTS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.85, edgecolor="#ccc",
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        f"Image Skill Variants — All Tasks (GPT-5.2, VQA, {n_runs} MC runs)\n"
        "Bars: mean ± 1 SD",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "all_tasks_mc_skill_variants.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nCombined plot → {out_path}")


def main(args):
    tasks = [args.task] if args.task else TASKS
    all_results = {}
    jsonl_root = os.path.join("outputs")

    dates = args.date if args.date else None  # list or None

    for task in tasks:
        print(f"\n=== {TASK_DISPLAY[task]} ===")
        result = plot_task(task, dates, args.out_dir, jsonl_root)
        if result is not None:
            all_results[task] = result

    if not args.task and len(all_results) > 1:
        # Determine the most common MC run count across all tasks
        n_runs = "?"
        for task in TASKS:
            jsonl_dir = os.path.join(jsonl_root, "MilaWang__SpatialEval", "vqa", task)
            accs = collect_mc_accuracies(jsonl_dir, dates)
            counts = [len(v) for v in accs.values() if v]
            if counts:
                n_runs = max(counts)
                break
        plot_combined(all_results, args.out_dir, n_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="eval_summary/result_vis")
    parser.add_argument("--task", default=None,
                        choices=TASKS,
                        help="Plot a single task; omit to plot all 3 + combined.")
    parser.add_argument("--date", default=None, nargs="+",
                        help="One or more YYYYMMDD dates to include (e.g. --date 20260316 20260317). "
                             "Omit to include all files regardless of date.")
    main(parser.parse_args())
