"""
Plot accuracy comparison across image-skill variants for mazenav, spatialgrid,
or spatialmap.

Reads eval_summary/vqa/{task}_acc.csv, filters to the 30-question runs
(identified by having exactly the 4 expected variant suffixes or a given
date stamp), and produces a grouped bar chart.

Usage (from spatial_eval/):
    uv run python eval_summary/plot_image_skill_variants.py \
        --eval_summary_dir eval_summary \
        --out_dir eval_summary/result_vis \
        --task mazenav \           # mazenav | spatialgrid | spatialmap
        [--date 20260313]          # optional: restrict to this YYYYMMDD
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd



# ── Per-task display config ────────────────────────────────────────────────
TASK_CONFIG = {
    "mazenav": {
        "display": "MazeNav",
        "q_labels": {
            "0": "Right turns",
            "1": "Total turns",
            "2": "Directional\n(Yes/No)",
        },
    },
    "spatialgrid": {
        "display": "Spatial Grid",
        "q_labels": {
            "0": "Count animals",
            "1": "Top-left corner",
            "2": "Row 1, Col 2",
        },
    },
    "spatialmap": {
        "display": "Spatial Map",
        "q_labels": {
            "0": "Direction of X\nrel. to Y",
            "1": "Which object in\ngiven direction",
            "2": "Count objects in\ngiven direction",
        },
    },
}

# ── Label / colour config ──────────────────────────────────────────────────
# Okabe-Ito colorblind-safe palette
VARIANTS = [
    ("baseline",               "Baseline",                "#555555"),  # dark grey
    ("img-only",               "Image Only",              "#56B4E9"),  # sky blue
    ("img-only-annotated",     "Img Only\nAnnotated",     "#CC79A7"),  # pink
    ("img-qa",                 "Img+Q&A",                 "#E69F00"),  # orange
    ("img-context",            "Img+Context",             "#009E73"),  # teal
    ("img-annotated-context",  "Img Annotated\nContext",  "#D55E00"),  # vermillion
]


def classify(model_name: str) -> str:
    """Map a model-name string to one of the 4 variant keys.

    Uses exact trailing-underscore patterns so n-variants (img-only-n3, img-only-n10,
    etc.) and tool-variants are not accidentally classified as img-only.
    """
    if "_skills_img-only-annotated_" in model_name:
        return "img-only-annotated"
    if "_skills_img-only_" in model_name:
        return "img-only"
    if "_skills_img-annotated-context_" in model_name:
        return "img-annotated-context"
    if "_skills_img-qa_" in model_name:
        return "img-qa"
    if "_skills_img-context_" in model_name:
        return "img-context"
    if "_skills_" in model_name or model_name.endswith("_skills"):
        return "full-skills"   # n-variants, tool-variants, original 300-sample run
    return "baseline"


def pick_run(df: pd.DataFrame, variant: str, date: str | None) -> float | None:
    """Return accuracy (0-1) for the most recent matching run (or None)."""
    mask = df["variant"] == variant
    if date:
        mask &= df["Model Name"].str.contains(date)
    rows = df[mask]
    if rows.empty:
        return None
    # Sort by model name (which embeds the timestamp) and take the last
    rows = rows.sort_values("Model Name")
    return float(rows["Acc"].iloc[-1])


def main(args):
    task = args.task
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task '{task}'. Choices: {list(TASK_CONFIG)}")
    tcfg = TASK_CONFIG[task]
    display = tcfg["display"]

    csv_path = os.path.join(args.eval_summary_dir, "vqa", f"{task}_acc.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["variant"] = df["Model Name"].apply(classify)

    os.makedirs(args.out_dir, exist_ok=True)


    # ── Aggregate all runs per variant ───────────────────────────────────
    import statistics
    accs_mean = {}
    accs_sd = {}
    accs_n = {}
    for key, label, _ in VARIANTS:
        vals = df[df["variant"] == key]["Acc"].tolist()
        accs_n[key] = len(vals)
        if vals:
            accs_mean[key] = statistics.mean(vals)
            accs_sd[key] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        else:
            accs_mean[key] = 0.0
            accs_sd[key] = 0.0
        print(f"  {key:20s}: mean={accs_mean[key]*100:.1f}%  sd={accs_sd[key]*100:.2f}  n={accs_n[key]}")

    baseline_acc = accs_mean["baseline"]

    # ── Bar chart with error bars and n annotation ───────────────────────
    keys   = [v[0] for v in VARIANTS]
    labels = [v[1] for v in VARIANTS]
    colors = [v[2] for v in VARIANTS]
    means  = [accs_mean[k] * 100 for k in keys]
    sds    = [accs_sd[k] * 100 for k in keys]
    ns     = [accs_n[k] for k in keys]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(x, means, width=0.55, color=colors, edgecolor="white",
                  linewidth=1.2, zorder=3)

    # Value labels only (no n annotation)
    for bar, mean, sd in zip(bars, means, sds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 1.0,
                f"{mean:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    # Error bars (SD)
    ax.errorbar(x, means, yerr=sds, fmt="none", ecolor="black", elinewidth=1.8, capsize=5, zorder=4)
    # SD text annotation
    for i, (mean, sd, n) in enumerate(zip(means, sds, ns)):
        if n > 1:
            ax.text(x[i], mean + sd + 4.5, f"±{sd:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#555")

    # Delta vs baseline (skip the baseline bar itself)
    for i, (key, _, _) in enumerate(VARIANTS):
        if key == "baseline":
            continue
        delta = accs_mean[key] - baseline_acc
        sign = "+" if delta >= 0 else ""
        color = "#009E73" if delta >= 0 else "#D55E00"  # CB teal / vermillion
        bar_top = means[i] + (sds[i] if ns[i] > 1 else 0)
        ax.text(x[i], bar_top + 10, f"Δ {sign}{delta * 100:.1f}%",
                ha="center", va="bottom", fontsize=9,
                color=color, fontweight="bold")

    # Baseline reference line
    ax.axhline(baseline_acc * 100, color="#7f7f7f", linestyle="--",
               linewidth=1.2, alpha=0.6, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 120)
    title = (f"{display} — Image Skill Variants (GPT-5.2, mean ± SD)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    legend_handles = [
        mpatches.Patch(color=color, label=label.replace("\n", " "))
        for _, label, color in VARIANTS
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left",
              framealpha=0.85, edgecolor="#ccc")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f"{task}_image_skill_variants.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {out_path}")

    # ── Per-question-type breakdown ────────────────────────────────────────
    jsonl_dir = os.path.join("outputs", "MilaWang__SpatialEval", "vqa", task)
    if os.path.isdir(jsonl_dir):
        _plot_qtype_breakdown(jsonl_dir, args.date, args.out_dir, baseline_acc,
                              tcfg["q_labels"], task, display)


def _plot_qtype_breakdown(jsonl_dir: str, date: str | None, out_dir: str,
                           baseline_acc: float, q_labels: dict,
                           task: str, display: str):
    """Break down accuracy by question type (q0/q1/q2) for all variant runs.

    When `date` is None, aggregates across all runs per variant (mean accuracy).
    """
    import json as _json
    import re as _re
    import statistics as _stats
    import sys as _sys

    _eval_dir = os.path.join(os.path.dirname(__file__), "..", "evals")
    if _eval_dir not in _sys.path:
        _sys.path.insert(0, os.path.abspath(_eval_dir))
    from evaluation import _check_answer  # noqa: E402

    # Variant key → substring pattern (order: most specific first)
    variant_patterns = [
        ("img-only-annotated",    "_skills_img-only-annotated_"),
        ("img-only",              "_skills_img-only_"),
        ("img-annotated-context", "_skills_img-annotated-context_"),
        ("img-qa",                "_skills_img-qa_"),
        ("img-context",           "_skills_img-context_"),
        ("baseline",              None),   # anything without _skills_
    ]

    def _classify_file(fname: str) -> str | None:
        """Return variant key or None if file should be skipped."""
        if not fname.endswith(".jsonl"):
            return None
        for vkey, pat in variant_patterns:
            if pat is None:
                # baseline: must have no _skills_ tag and no n-variant tag
                if "_skills_" not in fname:
                    return vkey
            else:
                if pat in fname:
                    # Skip n-variants (img-only-n3/n10/n30) and tool-variants
                    if _re.search(r"_img-only-n\d+_", fname):
                        return None
                    return vkey
        return None

    # Collect per-run q-type results: variant -> qtype_str -> list of per-run accuracy
    # Structure: {vkey: {qidx: [[correct_per_item, ...], ...]}}  — inner list is one run
    from collections import defaultdict
    run_data: dict[str, dict[str, list[list[int]]]] = defaultdict(lambda: defaultdict(list))

    for fname in sorted(os.listdir(jsonl_dir)):
        if date and date not in fname:
            continue
        if ".timing.json" in fname:
            continue
        vkey = _classify_file(fname)
        if vkey is None:
            continue

        path = os.path.join(jsonl_dir, fname)
        qtypes: dict[str, list[int]] = defaultdict(list)
        with open(path) as f:
            for line in f:
                try:
                    row = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                qidx = str(row["id"].split(".")[-1])
                correct, _ = _check_answer(row.get("answer", ""), row)
                qtypes[qidx].append(correct)
        for qidx, scores in qtypes.items():
            run_data[vkey][qidx].append(scores)

    if not run_data:
        print("[INFO] No jsonl files found for q-type breakdown — skipping.")
        return

    # Compute mean accuracy per (variant, qtype) across all matched runs
    qtypes_sorted = sorted(q_labels.keys())
    # results: {vkey: {qidx: mean_acc%}}
    results: dict[str, dict[str, float]] = {}
    for vkey, qtype_runs in run_data.items():
        results[vkey] = {}
        for qidx in qtypes_sorted:
            runs_for_q = qtype_runs.get(qidx, [])
            if not runs_for_q:
                results[vkey][qidx] = 0.0
                continue
            per_run_accs = [sum(r) / len(r) for r in runs_for_q if r]
            results[vkey][qidx] = 100 * _stats.mean(per_run_accs) if per_run_accs else 0.0

    n_groups = len(qtypes_sorted)
    n_variants = len(VARIANTS)
    x = np.arange(n_groups)
    width = 0.13

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for vi, (vkey, vlabel, vcolor) in enumerate(VARIANTS):
        if vkey not in results:
            continue
        accs = [results[vkey].get(q, 0.0) for q in qtypes_sorted]
        offset = (vi - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width, label=vlabel.replace("\n", " "),
                      color=vcolor, edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", rotation=0)

    n_runs_label = f"date={date}" if date else "all runs (mean)"
    ax.set_xticks(x)
    ax.set_xticklabels([q_labels[q] for q in qtypes_sorted], fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 120)
    ax.set_title(f"{display} — Accuracy by Question Type × Skill Variant ({n_runs_label})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{task}_image_skill_variants_by_qtype.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Q-type breakdown saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_summary_dir", default="eval_summary")
    parser.add_argument("--out_dir", default="eval_summary/result_vis")
    parser.add_argument("--task", default="mazenav",
                        choices=list(TASK_CONFIG),
                        help="Task to plot: mazenav | spatialgrid | spatialmap")
    parser.add_argument("--date", default=None,
                        help="YYYYMMDD date string to select today's runs only "
                             "(e.g. 20260313). Recommended to avoid mixing runs.")
    main(parser.parse_args())
