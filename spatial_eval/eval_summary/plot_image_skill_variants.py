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
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd



# ── Per-task display config ────────────────────────────────────────────────
TASK_CONFIG = {
    "mazenav": {
        "display": "Maze-Nav",
        "q_labels": {
            "0": "Right\nturns",
            "1": "Total\nturns",
            "2": "Directional\n(Yes/No)",
        },
    },
    "spatialgrid": {
        "display": "Spatial-Grid",
        "q_labels": {
            "0": "Count\nanimals",
            "1": "Top-left\ncorner",
            "2": "Row 1,\nCol 2",
        },
    },
    "spatialmap": {
        "display": "Spatial-map",
        "q_labels": {
            "0": "Direction of X\nrel. to Y",
            "1": "Object in\ngiven direction",
            "2": "Count objects in\ngiven direction",
        },
    },
}

TASK_ORDER = ["spatialmap", "mazenav", "spatialgrid"]

# ── Label / colour config ──────────────────────────────────────────────────
# Okabe-Ito colorblind-safe palette
VARIANTS = [
    ("baseline",               "Baseline",                "#555555"),  # dark grey
    ("bare_sam2",              "Baseline\n+SAM2",          "#F0E442"),  # yellow
    ("img-only",               "Image Only",              "#56B4E9"),  # sky blue
    ("img-only-annotated",     "Img Only\nAnnotated",     "#3A8FC0"),  # medium blue
    ("img-qa",                 "Img+Q&A",                 "#E69F00"),  # orange
    ("img-context",            "Img+Context",             "#009E73"),  # teal
    ("img-annotated-context",  "Img Annotated\nContext",  "#00704F"),  # dark teal
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


def classify(model_name: str) -> str:
    """Map a model-name string to one of the variant keys."""
    if "_bare_sam2_" in model_name:
        return "bare_sam2"
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
    tasks = TASK_ORDER if args.task is None else [args.task]
    qtype_results_by_task = {}

    for task in tasks:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unknown task '{task}'. Choices: {list(TASK_CONFIG)}")
        print(f"\n=== {TASK_CONFIG[task]['display']} ===")
        qtype_results = _plot_task(args, task)
        if qtype_results:
            qtype_results_by_task[task] = qtype_results

    if args.task is None and qtype_results_by_task:
        _plot_combined_qtype_breakdowns(qtype_results_by_task, args.out_dir)


def _plot_task(args, task: str):
    tcfg = TASK_CONFIG[task]
    display = tcfg["display"]

    csv_path = os.path.join(args.eval_summary_dir, "vqa", f"{task}_acc.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["variant"] = df["Model Name"].apply(classify)
    df["model"] = df["Model Name"].apply(_model_of)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Aggregate all runs per (model, variant) ───────────────────────────
    import statistics
    keys = [v[0] for v in VARIANTS]
    labels = [v[1] for v in VARIANTS]
    colors_map = {v[0]: v[2] for v in VARIANTS}

    # stats[model][variant] = (mean_frac, sd_frac, n)
    stats: dict[str, dict[str, tuple]] = {}
    for model, model_label in MODELS:
        stats[model] = {}
        for key in keys:
            vals = df[(df["model"] == model) & (df["variant"] == key)]["Acc"].tolist()
            n = len(vals)
            mean = statistics.mean(vals) if vals else 0.0
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            stats[model][key] = (mean, sd, n)
            print(f"  [{model}] {key:22s}: mean={mean*100:.1f}%  sd={sd*100:.2f}  n={n}")

    gpt52_baseline = stats["gpt-5.2"]["baseline"][0]

    # ── Bar chart — grouped by model (solid = GPT-5.2, hatched = GPT-5.5) ─
    n_models = len(MODELS)
    bar_w = 0.38 / n_models
    offsets = np.linspace(-(n_models - 1) * bar_w / 2, (n_models - 1) * bar_w / 2, n_models)
    hatches = [None, "//"]
    x = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for mi, (model, _) in enumerate(MODELS):
        for i, (key, _, color) in enumerate(VARIANTS):
            mean, sd, n = stats[model][key]
            if n == 0:
                continue
            ax.bar(x[i] + offsets[mi], mean * 100, width=bar_w, color=color,
                   hatch=hatches[mi], edgecolor="white", linewidth=1.2, zorder=3)
            if n > 1:
                ax.errorbar(x[i] + offsets[mi], mean * 100, yerr=sd * 100, fmt="none",
                            ecolor="black", elinewidth=1.8, capsize=4, zorder=4)
            label_y = mean * 100 + (sd * 100 if n > 1 else 0) + 1.0
            ax.text(x[i] + offsets[mi], label_y, f"{mean*100:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            if n > 1:
                ax.text(x[i] + offsets[mi], label_y + 4.5, f"±{sd*100:.1f}%",
                        ha="center", va="bottom", fontsize=7.5, color="#555")

    # Baseline reference line (GPT-5.2)
    ax.axhline(gpt52_baseline * 100, color="#7f7f7f", linestyle="--",
               linewidth=1.2, alpha=0.6, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 125)
    ax.set_title(
        f"{display} — Image Skill Variants (GPT-5.2 vs GPT-5.5, mean ± SD)\n"
        "solid = GPT-5.2, hatched = GPT-5.5",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    variant_handles = [
        mpatches.Patch(color=colors_map[k], label=l.replace("\n", " "))
        for k, l, _ in VARIANTS
    ]
    model_handles = [
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="white", label="GPT-5.2 (solid)"),
        mpatches.Patch(facecolor="#aaaaaa", edgecolor="white", hatch="//", label="GPT-5.5 (hatched)"),
    ]
    ax.legend(handles=variant_handles + model_handles, fontsize=8.5, loc="upper right",
              framealpha=0.85, edgecolor="#ccc", ncol=2)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f"{task}_image_skill_variants.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {out_path}")

    # ── Per-question-type breakdown ────────────────────────────────────────
    jsonl_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "MilaWang__SpatialEval", "vqa", task)
    if os.path.isdir(jsonl_dir):
        return _plot_qtype_breakdown(jsonl_dir, args.date, args.out_dir, gpt52_baseline,
                                    tcfg["q_labels"], task, display)
    return None


def _plot_qtype_breakdown(jsonl_dir: str, date: str | None, out_dir: str,
                           baseline_acc: float, q_labels: dict,
                           task: str, display: str):
    """Break down accuracy by question type into baseline vs skill average.

    Produces one figure per model. For each question type, the plot shows only:
      1. Baseline (no skills)
      2. Average across available skill variants

    When `date` is None, aggregates across all matching runs.
    """
    import json as _json
    import re as _re
    import statistics as _stats
    import sys as _sys

    _eval_dir = os.path.join(os.path.dirname(__file__), "..", "evals")
    if _eval_dir not in _sys.path:
        _sys.path.insert(0, os.path.abspath(_eval_dir))
    from evaluation import _check_answer  # noqa: E402

    skill_variant_keys = {v[0] for v in VARIANTS if v[0] != "baseline"}

    # Variant key -> substring pattern (order: most specific first)
    variant_patterns = [
        ("bare_sam2",             "_bare_sam2_"),
        ("img-only-annotated",    "_skills_img-only-annotated_"),
        ("img-only",              "_skills_img-only_"),
        ("img-annotated-context", "_skills_img-annotated-context_"),
        ("img-qa",                "_skills_img-qa_"),
        ("img-context",           "_skills_img-context_"),
        ("baseline",              None),
    ]

    def _classify_file(fname: str) -> str | None:
        """Return variant key or None if file should be skipped."""
        if not fname.endswith(".jsonl"):
            return None
        for vkey, pat in variant_patterns:
            if pat is None:
                # Baseline: no skill/SAM2 tag and no n-variant tag.
                if "_skills_" not in fname and "_bare_sam2_" not in fname:
                    return vkey
            else:
                if pat in fname:
                    # Skip n-variants (img-only-n3/n10/n30) and tool-variants.
                    if _re.search(r"_img-only-n\d+_", fname):
                        return None
                    return vkey
        return None

    # Structure: {model: {variant_key: {qidx: [[correct_per_item, ...], ...]}}}
    from collections import defaultdict
    run_data_by_model: dict[str, dict] = {
        m: defaultdict(lambda: defaultdict(list)) for m, _ in MODELS
    }

    for fname in sorted(os.listdir(jsonl_dir)):
        if date and date not in fname:
            continue
        if ".timing.json" in fname:
            continue
        vkey = _classify_file(fname)
        if vkey is None:
            continue
        model = _model_of(fname)
        if model not in run_data_by_model:
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
            run_data_by_model[model][vkey][qidx].append(scores)

    if all(len(rd) == 0 for rd in run_data_by_model.values()):
        print("[INFO] No jsonl files found for q-type breakdown — skipping.")
        return {}

    qtypes_sorted = sorted(q_labels.keys())
    x = np.arange(len(qtypes_sorted))
    width = 0.34
    model_slug = {"gpt-5.2": "gpt52", "gpt-5.5": "gpt55"}

    # Two visible categories only.
    bar_specs = [
        ("baseline", "Baseline", "#555555"),
        ("skill_avg", "Skill variants average", "#56B4E9"),
    ]
    model_results: dict[str, dict] = {}

    for model, model_label in MODELS:
        run_data = run_data_by_model[model]
        if not run_data:
            print(f"[INFO] No q-type data for {model} — skipping.")
            continue

        # First compute per-variant mean accuracy per qtype across runs.
        per_variant: dict[str, dict[str, tuple[float, float]]] = {}
        for vkey, qtype_runs in run_data.items():
            per_variant[vkey] = {}
            for qidx in qtypes_sorted:
                runs_for_q = qtype_runs.get(qidx, [])
                per_run_accs = [sum(r) / len(r) for r in runs_for_q if r]
                mean_acc = 100 * _stats.mean(per_run_accs) if per_run_accs else 0.0
                sd_acc = 100 * _stats.stdev(per_run_accs) if len(per_run_accs) > 1 else 0.0
                per_variant[vkey][qidx] = (mean_acc, sd_acc)

        # Collapse all non-baseline skill variants into one average per qtype.
        results: dict[str, dict[str, tuple[float, float]]] = {
            "baseline": {},
            "skill_avg": {},
        }
        for qidx in qtypes_sorted:
            results["baseline"][qidx] = per_variant.get("baseline", {}).get(qidx, (0.0, 0.0))

            skill_means = [
                per_variant[vkey][qidx][0]
                for vkey in skill_variant_keys
                if vkey in per_variant and qidx in per_variant[vkey]
            ]
            if skill_means:
                skill_mean = _stats.mean(skill_means)
                skill_sd = _stats.stdev(skill_means) if len(skill_means) > 1 else 0.0
            else:
                skill_mean = 0.0
                skill_sd = 0.0
            results["skill_avg"][qidx] = (skill_mean, skill_sd)

        model_results[model] = {
            "model_label": model_label,
            "qtypes_sorted": qtypes_sorted,
            "q_labels": q_labels,
            "results": results,
            "bar_specs": bar_specs,
        }

        # 8.9 cm wide (single-column journal)
        fig, ax = plt.subplots(figsize=(8.9 / 2.54, 7 / 2.54))
        _draw_qtype_breakdown_ax(
            ax,
            results,
            qtypes_sorted,
            q_labels,
            bar_specs,
            title=display,
            x_label_fontsize=6.5,
            y_label_fontsize=10,
            tick_fontsize=6.5,
            show_legend=True,
            wrap_width=16,
        )

        plt.tight_layout()
        slug = model_slug.get(model, model.replace(".", "").replace("-", ""))
        out_path = os.path.join(out_dir, f"{task}_image_skill_variants_by_qtype_{slug.upper()}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Q-type breakdown saved -> {out_path}")

    return model_results



def _wrap_xtick_label(label: str, width: int | None) -> str:
    """Wrap each x tick label line so labels do not overlap in multi-panel figures."""
    if width is None or width <= 0:
        return label
    wrapped_lines = []
    for line in str(label).split("\n"):
        wrapped_lines.extend(textwrap.wrap(line, width=width) or [line])
    return "\n".join(wrapped_lines)

def _draw_qtype_breakdown_ax(
    ax,
    results: dict,
    qtypes_sorted: list[str],
    q_labels: dict,
    bar_specs: list[tuple[str, str, str]],
    title: str | None = None,
    x_label_fontsize: float = 6.5,
    y_label_fontsize: float = 10,
    tick_fontsize: float = 6.5,
    show_legend: bool = False,
    wrap_width: int | None = 14,
):
    """Draw baseline vs skill-average question-type bars on an existing axis."""
    x = np.arange(len(qtypes_sorted))
    width = 0.34

    for bi, (key, label, color) in enumerate(bar_specs):
        means = [results[key].get(q, (0.0, 0.0))[0] for q in qtypes_sorted]
        sds = [results[key].get(q, (0.0, 0.0))[1] for q in qtypes_sorted]
        offset = (bi - 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=sds,
            fmt="none",
            ecolor="black",
            elinewidth=0.7,
            capsize=2,
            zorder=4,
        )

    ax.set_xticks(x)
    xtick_labels = [_wrap_xtick_label(q_labels[q], wrap_width) for q in qtypes_sorted]
    ax.set_xticklabels(xtick_labels, fontsize=x_label_fontsize, ha="center", linespacing=0.92)
    ax.set_ylabel("Accuracy (% correct)", fontsize=y_label_fontsize, fontweight="bold")
    ax.set_ylim(0, 100)
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    if show_legend:
        ax.legend(
            fontsize=6.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=True,
            framealpha=0.9,
            edgecolor="#ccc",
        )


def _plot_combined_qtype_breakdowns(qtype_results_by_task: dict, out_dir: str):
    """Create one 3-panel qtype breakdown figure per model in map/maze/grid order."""
    model_slug = {"gpt-5.2": "GPT52", "gpt-5.5": "GPT55"}

    for model, model_label in MODELS:
        available_tasks = [
            task for task in TASK_ORDER
            if task in qtype_results_by_task and model in qtype_results_by_task[task]
        ]
        if not available_tasks:
            print(f"[INFO] No combined q-type data for {model} — skipping.")
            continue

        fig, axes = plt.subplots(1, len(available_tasks), figsize=(24 / 2.54, 8 / 2.54), sharey=True)
        if len(available_tasks) == 1:
            axes = [axes]

        legend_handles = None
        for ax, task in zip(axes, available_tasks):
            payload = qtype_results_by_task[task][model]
            _draw_qtype_breakdown_ax(
                ax,
                payload["results"],
                payload["qtypes_sorted"],
                payload["q_labels"],
                payload["bar_specs"],
                title=TASK_CONFIG[task]["display"],
                x_label_fontsize=5.6,
                y_label_fontsize=10,
                tick_fontsize=6.5,
                show_legend=False,
                wrap_width=12,
            )
            ax.tick_params(axis="y", labelleft=True)
            if legend_handles is None:
                legend_handles = [
                    mpatches.Patch(color=color, label=label)
                    for _, label, color in payload["bar_specs"]
                ]

        if legend_handles is not None:
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.06),
                ncol=2,
                fontsize=6.5,
                frameon=True,
                framealpha=0.9,
                edgecolor="#ccc",
            )

        plt.tight_layout(rect=[0, 0.12, 1, 1], w_pad=2.0)
        slug = model_slug.get(model, model.replace(".", "").replace("-", "").upper())
        out_path = os.path.join(out_dir, f"all_tasks_image_skill_variants_by_qtype_{slug}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Combined q-type breakdown saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_summary_dir", default=os.path.join(os.path.dirname(__file__), "..", "eval_summary"))
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "result_vis"))
    parser.add_argument("--task", default=None,
                        choices=list(TASK_CONFIG),
                        help="Task to plot: mazenav | spatialgrid | spatialmap. Omit to plot all tasks.")
    parser.add_argument("--date", default=None,
                        help="YYYYMMDD date string to select today's runs only "
                             "(e.g. 20260313). Recommended to avoid mixing runs.")
    main(parser.parse_args())