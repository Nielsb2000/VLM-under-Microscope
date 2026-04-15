"""
Compute accuracy from outputs_100 .jsonl files and generate comparison plots
for all three tasks (spatialgrid, spatialmap, mazenav) — VQA and VTQA modes,
baseline vs skills.

Each image has 3 questions, so 100 images = 300 lines per file.
Uses the official evaluate_model_accuracy() from evals/evaluation.py.

Usage:
  python spatial_eval/eval_summary/compute_and_plot.py
"""
import os
import sys
import csv
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use the official evaluator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evals.evaluation import evaluate_model_accuracy


TASKS  = ["spatialgrid", "spatialmap", "mazenav"]
MODES  = ["vqa", "vtqa"]

OUTPUTS_DIR      = "spatial_eval/outputs/MilaWang__SpatialEval"
EVAL_SUMMARY_DIR = "spatial_eval/eval_summary"
OUT_DIR          = "spatial_eval/eval_summary/result_vis"
FIRST_K_IMAGES   = 100  # informational only — files contain exactly 100 images × 3 q = 300 lines


# ── CSV helpers ───────────────────────────────────────────────────────────────

def read_csv(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(csv_path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Model Name", "Acc"])
        writer.writeheader()
        writer.writerows(rows)


def upsert_csv(csv_path: str, model_name: str, acc: float):
    """Add or update a row in the CSV."""
    rows = read_csv(csv_path)
    updated = False
    for row in rows:
        if row["Model Name"] == model_name:
            row["Acc"] = str(acc)
            updated = True
            break
    if not updated:
        rows.append({"Model Name": model_name, "Acc": str(acc)})
    write_csv(csv_path, rows)


def model_name_from_filename(fname: str) -> str:
    """Convert 'm-gpt-5.2_bare_20260308_140533.jsonl' → 'gpt-5.2_bare_20260308_140533'"""
    import re
    name = os.path.basename(fname)
    name = re.sub(r"^m-", "", name)
    name = re.sub(r"\.jsonl$", "", name)
    return name


# ── Plot ──────────────────────────────────────────────────────────────────────

def classify_variant(model_name: str) -> str:
    return "skills" if "_skills_" in model_name or model_name.endswith("_skills") else "baseline"


def load_csv_results(eval_summary_dir: str, mode: str, task: str) -> dict:
    csv_path = os.path.join(eval_summary_dir, mode, f"{task}_acc.csv")
    rows = read_csv(csv_path)
    results = {"baseline": 0.0, "skills": 0.0}
    for row in rows:
        variant = classify_variant(row["Model Name"])
        try:
            acc = float(row["Acc"])
        except (ValueError, KeyError):
            acc = 0.0
        # Keep the best (highest) accuracy per variant
        if acc > results[variant]:
            results[variant] = acc
    return results


def plot_task(task: str, eval_summary_dir: str, out_dir: str, first_k: int):
    os.makedirs(out_dir, exist_ok=True)
    modes = ["vqa", "vtqa"]
    results = {}
    for mode in modes:
        results[mode] = load_csv_results(eval_summary_dir, mode, task)
        print(f"  {mode.upper()} — baseline: {results[mode]['baseline']:.1%}  "
              f"skills: {results[mode]['skills']:.1%}")

    x     = np.arange(len(modes))
    width = 0.32

    baseline_accs = [results[m]["baseline"] * 100 for m in modes]
    skills_accs   = [results[m]["skills"] * 100   for m in modes]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars_base  = ax.bar(x - width / 2, baseline_accs, width,
                        label="GPT-5.2 Baseline", color="#555555",
                        edgecolor="white", linewidth=0.8)
    bars_skill = ax.bar(x + width / 2, skills_accs, width,
                        label="GPT-5.2 + Skills", color="#56B4E9",
                        edgecolor="white", linewidth=0.8)

    for bar in bars_base:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars_skill:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for i, mode in enumerate(modes):
        delta = results[mode]["skills"] - results[mode]["baseline"]
        sign  = "+" if delta >= 0 else ""
        y_pos = max(baseline_accs[i], skills_accs[i]) + 5.5
        color = "#009E73" if delta >= 0 else "#D55E00"
        ax.text(i, y_pos, f"Δ {sign}{delta * 100:.1f}%",
                ha="center", va="bottom", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes], fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(f"Task: {task.capitalize()} — Skills vs Baseline\n"
                 f"(GPT-5.2, {first_k} images × 3 questions)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{task}_skills_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tmp_dir = tempfile.mkdtemp()  # for official eval summary side-effect files

    print("=== Step 1: Computing accuracy from outputs_100 (official evaluator) ===\n")
    for mode in MODES:
        for task in TASKS:
            task_dir = os.path.join(OUTPUTS_DIR, mode, task)
            if not os.path.isdir(task_dir):
                print(f"[WARN] Directory not found: {task_dir}")
                continue
            files = [f for f in os.listdir(task_dir) if f.endswith(".jsonl")]
            if not files:
                print(f"[WARN] No .jsonl files in {task_dir}")
                continue
            csv_path = os.path.join(EVAL_SUMMARY_DIR, mode, f"{task}_acc.csv")
            for fname in sorted(files):
                jsonl_path = os.path.join(task_dir, fname)
                model_name = model_name_from_filename(fname)
                eval_summary_path = os.path.join(tmp_dir, f"{task}_{model_name}.jsonl")
                acc, n = evaluate_model_accuracy(jsonl_path, eval_summary_path, model_name)
                upsert_csv(csv_path, model_name, acc)
                variant = "skills" if "_skills_" in model_name else "baseline"
                print(f"  [{mode.upper()}] [{task}] {model_name} ({variant}): {acc:.1%} ({int(acc*n)}/{n})")

    print("\n=== Step 2: Generating plots ===\n")
    for task in TASKS:
        print(f"Task: {task}")
        plot_task(task, EVAL_SUMMARY_DIR, OUT_DIR, FIRST_K_IMAGES)
        print()

    print("Done! All plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
