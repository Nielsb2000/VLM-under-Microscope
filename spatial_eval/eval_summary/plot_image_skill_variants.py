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
VARIANTS = [
    ("baseline",     "Baseline\n(no skills)",    "#7f7f7f"),
    ("img-only",     "Image Only",                "#5B8DB8"),
    ("img-qa",       "Image + Q&A\n(biased)",     "#F28C38"),
    ("img-context",  "Image + Context\n(unbiased)","#2ca02c"),
]


def classify(model_name: str) -> str:
    """Map a model-name string to one of the 4 variant keys."""
    if "_skills_img-only" in model_name:
        return "img-only"
    if "_skills_img-qa" in model_name:
        return "img-qa"
    if "_skills_img-context" in model_name:
        return "img-context"
    if "_skills_" in model_name or model_name.endswith("_skills"):
        return "full-skills"   # original 300-sample run — not a focus here
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

    # ── Collect accuracies ─────────────────────────────────────────────────
    accs = {}
    for key, label, _ in VARIANTS:
        val = pick_run(df, key, args.date)
        if val is None:
            print(f"[WARN] No run found for variant='{key}'" +
                  (f" date='{args.date}'" if args.date else ""))
        accs[key] = val if val is not None else 0.0
        pct = accs[key] * 100
        print(f"  {key:20s}: {pct:.1f}%")

    baseline_acc = accs["baseline"]

    # ── Bar chart ─────────────────────────────────────────────────────────
    keys   = [v[0] for v in VARIANTS]
    labels = [v[1] for v in VARIANTS]
    colors = [v[2] for v in VARIANTS]
    values = [accs[k] * 100 for k in keys]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(x, values, width=0.55, color=colors, edgecolor="white",
                  linewidth=1.2, zorder=3)

    # Value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    # Delta vs baseline (skip the baseline bar itself)
    for i, (key, _, _) in enumerate(VARIANTS):
        if key == "baseline":
            continue
        delta = accs[key] - baseline_acc
        sign = "+" if delta >= 0 else ""
        color = "#1a7a1a" if delta >= 0 else "#d62728"
        top = values[i] + 5.5
        ax.text(x[i], top, f"Δ {sign}{delta * 100:.1f}%",
                ha="center", va="bottom", fontsize=10,
                color=color, fontweight="bold")

    # Baseline reference line
    ax.axhline(baseline_acc * 100, color="#7f7f7f", linestyle="--",
               linewidth=1.2, alpha=0.6, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    title = (f"{display} — Image Skill Variants (GPT-5.2, VQA, n=30)\n"
             "Skills: image-only  |  image + Q&A (biased)  |  image + context (unbiased)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f"{task}_image_skill_variants.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {out_path}")

    # ── Per-question-type breakdown ────────────────────────────────────────
    jsonl_dir = os.path.join("outputs", "MilaWang__SpatialEval", "vqa", task)
    if os.path.isdir(jsonl_dir) and args.date:
        _plot_qtype_breakdown(jsonl_dir, args.date, args.out_dir, baseline_acc,
                              tcfg["q_labels"], task, display)


def _plot_qtype_breakdown(jsonl_dir: str, date: str, out_dir: str,
                           baseline_acc: float, q_labels: dict,
                           task: str, display: str):
    """Break down accuracy by question type (q0/q1/q2) for the 4 variant runs."""
    import json as _json
    import re as _re
    import sys as _sys

    # Import the same _check_answer used by evaluation.py
    _eval_dir = os.path.join(os.path.dirname(__file__), "..", "evals")
    if _eval_dir not in _sys.path:
        _sys.path.insert(0, os.path.abspath(_eval_dir))
    from evaluation import _check_answer  # noqa: E402

    # Map variant key → jsonl filename suffix pattern
    variant_patterns = {
        "baseline":    rf"m-gpt-5\.2_bare_{date}",
        "img-only":    rf"m-gpt-5\.2_bare_skills_img-only_{date}",
        "img-qa":      rf"m-gpt-5\.2_bare_skills_img-qa_{date}",
        "img-context": rf"m-gpt-5\.2_bare_skills_img-context_{date}",
    }

    qtypes_sorted = sorted(q_labels.keys())
    results = {}  # variant -> qtype_str -> (correct, total)
    for file in sorted(os.listdir(jsonl_dir)):
        for vkey, pat in variant_patterns.items():
            if _re.search(pat, file):
                path = os.path.join(jsonl_dir, file)
                qtypes: dict[str, list[int]] = {}
                with open(path) as f:
                    for line in f:
                        try:
                            row = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        # id like {task}.{mode}.{img_idx}.{q_idx}
                        qidx = str(row["id"].split(".")[-1])
                        correct, _ = _check_answer(row.get("answer", ""), row)
                        qtypes.setdefault(qidx, []).append(correct)
                results[vkey] = {q: (sum(v), len(v)) for q, v in qtypes.items()}
                break

    if not results:
        print("[INFO] No jsonl files matched for q-type breakdown — skipping.")
        return

    qtypes_sorted = sorted(q_labels.keys())
    n_groups = len(qtypes_sorted)
    n_variants = len(VARIANTS)
    x = np.arange(n_groups)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for vi, (vkey, vlabel, vcolor) in enumerate(VARIANTS):
        if vkey not in results:
            continue
        accs = []
        for q in qtypes_sorted:
            corr, total = results[vkey].get(q, (0, 1))
            accs.append(100 * corr / total if total > 0 else 0)
        offset = (vi - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width, label=vlabel.replace("\n", " "),
                      color=vcolor, edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([q_labels[q] for q in qtypes_sorted], fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(f"{display} — Accuracy by Question Type × Skill Variant (n=10 per type)",
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
