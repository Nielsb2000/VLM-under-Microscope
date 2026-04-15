"""
reproduce_results.py
====================
End-to-end reproducibility script for the MS Paint Reasoning Evaluation experiment.

Runs the full experiment pipeline in order:
  1. Evaluate all configurations  (Eval_script.py)
  2. Generate all visualizations  (viz/plot_*.py)

Matches exactly the 54 result files that are the canonical results
(Results/dashboard_data/llm_results_*.json):
  - Image types  : color, greyscale, inverted_greyscale
  - Blur levels  : no_blur, med_blur, heavy_blur
  - Reasoning    : low, medium, high
  - Skills       : yes (DeepAgent), no (plain chat)
  - Models       : gpt-4o, gpt-5.1, gpt-5.2

Prerequisites:
  - .env at project root with OPENAI_API_KEY (and optionally OPENAI_BASE_URL)
  - `uv sync` run from project root

Usage (from project root):
    uv run python MS_Paint_Reasoning_Evaluation/reproduce_results.py

Flags:
    --skip-eval     Skip evaluation (use existing Results/); go straight to viz
    --smoke         Run evaluation on img1 only (fast sanity check, ~few API calls)
"""

import argparse
import subprocess
import sys
import os

# ---------------------------------------------------------------------------
# Configuration — mirrors exactly what produced the 54 dashboard_data JSONs
# ---------------------------------------------------------------------------
MODELS         = ["gpt-4o", "gpt-5.1", "gpt-5.2"]
IMAGE_TYPES    = ["color", "greyscale", "inverted_greyscale"]
BLUR_LEVELS    = ["no_blur", "med_blur", "heavy_blur"]
REASONING_EFFS = ["low", "medium", "high"]
SKILLS_MODES   = ["yes", "no"]

# ---------------------------------------------------------------------------
# Visualization combinations to generate
# For each viz script we enumerate every relevant (image_type, …) combination
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def run(cmd: list[str], label: str) -> None:
    """Run a subprocess command, print a clear header, abort on failure."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print('='*70)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed (exit {result.returncode}): {label}")
        sys.exit(result.returncode)


def step_evaluate(smoke: bool) -> None:
    """Run Eval_script.py once with the full configuration matrix."""
    script = os.path.join(ROOT, "evaluation", "Eval_script.py")
    cmd = [
        PYTHON, script,
        "--image-types", *IMAGE_TYPES,
        "--blur-levels", *BLUR_LEVELS,
        "--reasoning-effort", *REASONING_EFFS,
        "--skills", *SKILLS_MODES,
        "--models", *MODELS,
    ]
    if smoke:
        cmd.append("--smoke")
    run(cmd, "STEP 1 — Full evaluation matrix")


def step_visualize() -> None:
    """Generate all visualization PNGs from Results/dashboard_data/."""

    # -- plot_accuracy_all_conditions.py --
    # All models × all blur levels for each image_type × reasoning_mode × skills_mode
    script = os.path.join(ROOT, "viz", "plot_accuracy_all_conditions.py")
    for image_type in IMAGE_TYPES:
        for reasoning_mode in REASONING_EFFS:
            for skills_mode in ["skills", "no_skills"]:
                run(
                    [PYTHON, script,
                     "--image-type", image_type,
                     "--reasoning-mode", reasoning_mode,
                     "--skills-mode", skills_mode],
                    f"plot_accuracy_all_conditions | {image_type} | {reasoning_mode} | {skills_mode}",
                )

    # -- plot_accuracy_by_blur.py --
    # Accuracy across the blur progression for each image_type × reasoning_mode
    script = os.path.join(ROOT, "viz", "plot_accuracy_by_blur.py")
    for image_type in IMAGE_TYPES:
        for reasoning_mode in REASONING_EFFS:
            run(
                [PYTHON, script,
                 "--image-type", image_type,
                 "--blur-levels", *BLUR_LEVELS,
                 "--reasoning-mode", reasoning_mode],
                f"plot_accuracy_by_blur | {image_type} | {reasoning_mode}",
            )

    # -- plot_accuracy_heatmap.py --
    # Per-condition heatmap for each image_type × blur_level × reasoning_mode × skills_mode
    script = os.path.join(ROOT, "viz", "plot_accuracy_heatmap.py")
    for image_type in IMAGE_TYPES:
        for blur_level in BLUR_LEVELS:
            for reasoning_mode in REASONING_EFFS:
                for skills_mode in ["skills", "no_skills"]:
                    run(
                        [PYTHON, script,
                         "--image-type", image_type,
                         "--blur-level", blur_level,
                         "--reasoning-mode", reasoning_mode,
                         "--skills-mode", skills_mode],
                        f"plot_accuracy_heatmap | {image_type} | {blur_level} | {reasoning_mode} | {skills_mode}",
                    )

    # -- plot_accuracy_heavy_blur_high.py --
    # Compare medium vs high reasoning on heavy blur, for each image_type
    script = os.path.join(ROOT, "viz", "plot_accuracy_heavy_blur_high.py")
    for image_type in IMAGE_TYPES:
        run(
            [PYTHON, script,
             "--image-type", image_type,
             "--mode-a", "medium",
             "--mode-b", "high"],
            f"plot_accuracy_heavy_blur_high | {image_type} | medium vs high",
        )

    # -- plot_token_time_stats.py --
    # Token usage + cost for each image_type × blur_level × reasoning_mode
    script = os.path.join(ROOT, "viz", "plot_token_time_stats.py")
    for image_type in IMAGE_TYPES:
        for blur_level in BLUR_LEVELS:
            for reasoning_mode in REASONING_EFFS:
                run(
                    [PYTHON, script,
                     "--image-type", image_type,
                     "--blur-level", blur_level,
                     "--reasoning-mode", reasoning_mode],
                    f"plot_token_time_stats | {image_type} | {blur_level} | {reasoning_mode}",
                )

    print(f"\n{'='*70}")
    print("  All visualizations saved to Results/res_vis/")
    print('='*70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end reproducibility script for MS Paint Reasoning Evaluation."
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation step; generate visualizations from existing results only.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Evaluation smoke-test mode: run on img1 only (fast, few API calls).",
    )
    args = parser.parse_args()

    print("\n" + "#"*70)
    print("  MS Paint Reasoning Evaluation — Full Reproducibility Run")
    print("#"*70)

    if not args.skip_eval:
        step_evaluate(smoke=args.smoke)
    else:
        print("\n[--skip-eval] Skipping evaluation. Using existing Results/.")

    step_visualize()

    print("\n" + "#"*70)
    print("  Done. Results in:")
    print("    Results/dashboard_data/  — aggregated JSONs (canonical)")
    print("    Results/res_vis/         — all visualization PNGs")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
