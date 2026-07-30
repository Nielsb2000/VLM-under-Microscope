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
TASKS = ["spatialmap", "mazenav", "spatialgrid"]
TASK_DISPLAY = {
    "mazenav":     "Maze-Nav",
    "spatialgrid": "Spatial-Grid",
    "spatialmap":  "Spatial-Map",
}

# Okabe-Ito colorblind-safe palette
# Keep labels single-line for the new all-variants figure, where the x-axis
# should spell out each skill-variant name completely.
VARIANTS = [
    ("baseline",    "Baseline",              "#555555"),  # dark grey
    ("bare_sam2",   "Baseline + SAM2",       "#F0E442"),  # yellow
    ("img-only",    "Image Examples (n=3)",  "#8ECAE6"),  # light blue
    ("img-qa",      "Image + Q&A",           "#E69F00"),  # orange
    ("img-context", "Image + Context",       "#009E73"),  # teal
]

# The original img-only condition is equivalent to the n=3 image-example
# condition, so n=3 is represented by the existing "img-only" key above.
IMG_RANGE_VARIANTS = [
    ("img-only-annotated", "Image Examples Annotated (n=3)", "#5DADE2"),
    ("img-only-range-n10", "Image Examples (n=10)",           "#0072B2"),
    ("img-only-range-n30", "Image Examples (n=30)",           "#003B73"),
]

# Logical x-axis order for the all-variants figure:
# baselines -> increasing image examples -> other image-based skill variants.
ALL_SKILL_VARIANTS = [
    ("baseline",              "Baseline",                         "#555555"),
    ("bare_sam2",             "Baseline + SAM2",                  "#F0E442"),
    ("img-only",              "Image Examples (n=3)",             "#8ECAE6"),
    ("img-only-annotated",    "Image Examples Annotated (n=3)",   "#5DADE2"),
    ("img-only-range-n10",    "Image Examples (n=10)",            "#0072B2"),
    ("img-only-range-n30",    "Image Examples (n=30)",            "#003B73"),
    ("img-qa",                "Image + Q&A",                      "#E69F00"),
    ("img-context",           "Image + Context",                  "#009E73"),
]

MODELS = [
    ("gpt-5.2", "GPT-5.2"),
    ("gpt-5.5", "GPT-5.5"),
]


def _model_of(fname: str) -> str:
    """Return 'gpt-5.5' or 'gpt-5.2' based on filename."""
    if "gpt-5.5" in fname:
        return "gpt-5.5"
    return "gpt-5.2"


def _classify(fname: str) -> str:
    """Classify a result filename into one of the plotted skill variants.

    The image-range filenames have changed a few times while experimenting, so
    this accepts common spellings such as img-only-range-n30,
    img_only_range_30, img-only-n30, image-only-range-n30, and top30.
    The original img-only condition is treated as Image Examples (n=3).
    """
    name = fname.lower()
    norm = re.sub(r"[^a-z0-9]+", "_", name).strip("_")

    if "bare_sam2" in norm:
        return "bare_sam2"

    is_img_only = (
        "img_only" in norm
        or "imgonly" in norm
        or "image_only" in norm
        or "imageonly" in norm
    )

    # Match img-only range/annotated variants before the generic img-only
    # condition. n=3 is deliberately folded into the generic img-only key,
    # because the original img-only run and Image Examples (n=3) are the same
    # condition. Annotated image-example runs remain separate.
    if is_img_only:
        if "annotated" in norm or "annotation" in norm or "annot" in norm:
            return "img-only-annotated"
        for n in (30, 10, 3):
            n_tag = rf"(?:^|_)(?:n_?{n}|range_?n_?{n}|range_?{n}|k_?{n}|top_?{n}|examples_?{n})(?:_|$)"
            if re.search(n_tag, norm):
                return "img-only" if n == 3 else f"img-only-range-n{n}"
        return "img-only"

    if "skills_img_qa" in norm:
        return "img-qa"
    if "skills_img_context" in norm:
        return "img-context"
    if "skills" in norm:
        return "full-skills"  # original full-skill runs ignored
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


def collect_mc_accuracies(
    jsonl_dir: str,
    dates: list[str] | None,
    variants: list[tuple[str, str, str]] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """
    Returns {model_key: {variant_key: [acc_run0, acc_run1, ...]}} for all files
    in jsonl_dir. If dates is given, only files containing at least one of the
    YYYYMMDD strings are included.
    """
    check = _load_check_answer()
    variants = variants or VARIANTS
    variant_keys = {v[0] for v in variants}
    result: dict[str, dict[str, list[float]]] = {
        m: {v[0]: [] for v in variants} for m, _ in MODELS
    }

    if not os.path.isdir(jsonl_dir):
        return result

    for fname in sorted(os.listdir(jsonl_dir)):
        if not fname.endswith(".jsonl"):
            continue
        if ".timing.json" in fname:
            continue
        if dates and not any(d in fname for d in dates):
            continue
        variant = _classify(fname)
        if variant not in variant_keys:
            continue
        model = _model_of(fname)
        if model not in result:
            continue
        acc = _file_accuracy(os.path.join(jsonl_dir, fname), check)
        result[model][variant].append(acc)

    return result


def _mean_sd(values: list[float]):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def plot_task(task: str, dates: list[str] | None, out_dir: str, jsonl_root: str):
    jsonl_dir = os.path.join(jsonl_root, "MilaWang__SpatialEval", "vqa", task)
    accs_by_model = collect_mc_accuracies(jsonl_dir, dates)

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
            print(f"  [{model}] {task:12s} {k:15s}: {m_str} {s_str}  ({n_str})")

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
                            ecolor="black", elinewidth=1.5, capsize=3, zorder=4)
            label_y = m * 100 + (s * 100 if s else 0) + 1.5
            ax.text(x[i] + offsets[mi], label_y, f"{m*100:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
            if s and s > 0:
                ax.text(x[i] + offsets[mi], label_y + 4.5, f"±{s*100:.1f}%",
                        ha="center", va="bottom", fontsize=7, color="#555")



    n_runs = max((stats[m][k][2] for m, _ in MODELS for k in keys), default="?")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 125)
    ax.set_title(
        f"{display} — Image Skill Variants (GPT-5.2 vs GPT-5.5, VQA, up to {n_runs} runs)\n"
        "Bars: mean ± 1 SD  |  solid = GPT-5.2, hatched = GPT-5.5",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    import matplotlib.patches as mpatches
    variant_handles = [
        mpatches.Patch(color=c, label=l.replace("\n", " "))
        for _, l, c in VARIANTS
    ]
    model_handles = [
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="white", label="GPT-5.2 (solid)"),
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="white", hatch="//", label="GPT-5.5 (hatched)"),
    ]
    ax.legend(handles=variant_handles + model_handles, fontsize=8.5, loc="upper right",
              framealpha=0.85, edgecolor="#ccc", ncol=2)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_mc_skill_variants.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")
    return stats


def collect_task_stats(
    task: str,
    dates: list[str] | None,
    jsonl_root: str,
    variants: list[tuple[str, str, str]],
):
    """Compute mean, SD, and run count for one task over a given variant list."""
    jsonl_dir = os.path.join(jsonl_root, "MilaWang__SpatialEval", "vqa", task)
    accs_by_model = collect_mc_accuracies(jsonl_dir, dates, variants=variants)
    keys = [v[0] for v in variants]

    stats: dict[str, dict[str, tuple]] = {}
    for model, _ in MODELS:
        stats[model] = {}
        for k in keys:
            m, s = _mean_sd(accs_by_model[model][k])
            n = len(accs_by_model[model][k])
            stats[model][k] = (m, s, n)

    if all(stats[m][k][0] is None for m, _ in MODELS for k in keys):
        return None
    return stats


def plot_all_skill_variants_combined(
    all_results: dict,
    out_dir: str,
    variants: list[tuple[str, str, str]] = ALL_SKILL_VARIANTS,
):
    """Combined figure with all skill variants, including image-range n=3/10/30."""
    fig, axes = plt.subplots(1, 3, figsize=(24 / 2.54, 10 / 2.54), sharey=True)

    keys = [v[0] for v in variants]
    labels = [v[1] for v in variants]
    x = np.arange(len(keys))

    n_models = len(MODELS)
    bar_w = 0.62 / n_models
    offsets = np.linspace(-(n_models - 1) * bar_w / 2, (n_models - 1) * bar_w / 2, n_models)
    hatches = [None, "//"]

    for ax, task in zip(axes, TASKS):
        task_stats = all_results.get(task)
        if task_stats is None:
            ax.set_title(TASK_DISPLAY[task], fontsize=7)
            continue

        for mi, (model, _) in enumerate(MODELS):
            for i, (k, _, color) in enumerate(variants):
                m, s, n = task_stats[model][k]
                if m is None:
                    continue
                ax.bar(
                    x[i] + offsets[mi],
                    m * 100,
                    width=bar_w,
                    color=color,
                    hatch=hatches[mi],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
                if s and s > 0:
                    ax.errorbar(
                        x[i] + offsets[mi],
                        m * 100,
                        yerr=s * 100,
                        fmt="none",
                        ecolor="black",
                        elinewidth=0.8,
                        capsize=2,
                        zorder=4,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.8, rotation=50, ha="right")
        ax.set_ylim(40, 100)
        ax.set_title(TASK_DISPLAY[task], fontsize=8, fontweight="bold", pad=3)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.4, zorder=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=6.5)

    # Show the y-axis values and label on every task panel, even though the
    # y-scale is shared across subplots.
    for ax in axes:
        ax.tick_params(axis="y", labelleft=True)
        ax.set_ylabel("Accuracy (% correct)", fontsize=10, fontweight="bold")

    import matplotlib.patches as mpatches
    model_handles = [
        mpatches.Patch(facecolor="#888", edgecolor="white", label=MODELS[0][1]),
        mpatches.Patch(facecolor="#888", edgecolor="white", hatch="//", label=MODELS[1][1]),
    ]
    fig.legend(
        handles=model_handles,
        loc="lower center",
        ncol=len(model_handles),
        fontsize=6.5,
        frameon=True,
        framealpha=0.9,
        edgecolor="#ccc",
        bbox_to_anchor=(0.5, -0.06),
        title="Model",
        title_fontsize=6.5,
    )

    plt.tight_layout()
    out_path = os.path.join(out_dir, "all_tasks_all_skill_variants.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nAll-skill-variants combined plot -> {out_path}")


def plot_combined(all_results: dict, out_dir: str, n_runs: int):
    """3-panel figure: one subplot per task (journal quality, 18 cm wide)."""
    # 18 cm wide (double-column journal), ~9 cm tall
    fig, axes = plt.subplots(1, 3, figsize=(18 / 2.54, 9 / 2.54), sharey=False)

    keys   = [v[0] for v in VARIANTS]
    labels = [v[1] for v in VARIANTS]
    x = np.arange(len(keys))

    n_models = len(MODELS)
    bar_w = 0.36 / n_models
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
        mpatches.Patch(color=c, label=l.replace("\n", " "))
        for _, l, c in VARIANTS
    ]
    model_handles = [
        mpatches.Patch(facecolor="#888", edgecolor="white", label="GPT-5.2"),
        mpatches.Patch(facecolor="#888", edgecolor="white", hatch="//", label="GPT-5.5"),
    ]
    fig.legend(handles=variant_handles + model_handles, loc="lower center",
               ncol=4, fontsize=5.5, frameon=True, framealpha=0.9,
               edgecolor="#ccc", bbox_to_anchor=(0.5, -0.28))

    fig.suptitle("Accuracy by skill variant across tasks (mean ± SD)",
                 fontsize=7, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "all_tasks_mc_skill_variants.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nCombined plot → {out_path}")


def main(args):
    tasks = [args.task] if args.task else TASKS
    all_results = {}
    jsonl_root = os.path.join(os.path.dirname(__file__), "..", "outputs")

    dates = args.date if args.date else None  # list or None

    for task in tasks:
        print(f"\n=== {TASK_DISPLAY[task]} ===")
        result = plot_task(task, dates, args.out_dir, jsonl_root)
        if result is not None:
            all_results[task] = result

    if not args.task and len(all_results) > 1:
        # Determine the most common run count across all tasks
        n_runs = "?"
        for task in TASKS:
            jsonl_dir = os.path.join(jsonl_root, "MilaWang__SpatialEval", "vqa", task)
            accs = collect_mc_accuracies(jsonl_dir, dates)
            counts = [len(v) for model_accs in accs.values() for v in model_accs.values() if v]
            if counts:
                n_runs = max(counts)
                break
        plot_combined(all_results, args.out_dir, n_runs)

        all_variant_results = {}
        for task in TASKS:
            result = collect_task_stats(task, dates, jsonl_root, ALL_SKILL_VARIANTS)
            if result is not None:
                all_variant_results[task] = result

        if len(all_variant_results) > 1:
            plot_all_skill_variants_combined(all_variant_results, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "result_vis"))
    parser.add_argument("--task", default=None,
                        choices=TASKS,
                        help="Plot a single task; omit to plot all 3 + combined.")
    parser.add_argument("--date", default=None, nargs="+",
                        help="One or more YYYYMMDD dates to include (e.g. --date 20260316 20260317). "
                             "Omit to include all files regardless of date.")
    main(parser.parse_args())
