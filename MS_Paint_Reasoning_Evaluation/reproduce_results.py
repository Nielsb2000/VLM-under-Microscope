"""
reproduce_results.py
====================
End-to-end reproducibility script for the MS Paint Reasoning Evaluation experiment.

Runs the full experiment pipeline in order:
  1. Evaluate all configurations (Eval_script.py)
  2. Generate all visualizations (viz/plot_*.py)

Flags:
    --skip-eval     Skip evaluation; generate plots only
    --smoke         Run evaluation on img1 only
    --max-workers   Number of concurrent evaluation jobs inside Eval_script.py
    --viz-workers   Number of concurrent visualization subprocesses
    --seed          Best-effort seed passed through evaluation and judge calls
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import sys

MODELS = ["gpt-4o", "gpt-5.1", "gpt-5.2", "gpt-5.5"]
IMAGE_TYPES = ["color", "greyscale", "inverted_greyscale"]
BLUR_LEVELS = ["no_blur", "med_blur", "heavy_blur"]
REASONING_EFFS = ["low", "medium", "high"]
SKILLS_MODES = ["yes", "no"]
VIZ_SKILLS_MODES = ["skills", "no_skills"]

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print("=" * 70)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {label}")


def step_evaluate(smoke: bool, max_workers: int, seed: int) -> None:
    script = os.path.join(ROOT, "evaluation", "Eval_script.py")
    cmd = [
        PYTHON,
        script,
        "--image-types",
        *IMAGE_TYPES,
        "--blur-levels",
        *BLUR_LEVELS,
        "--reasoning-effort",
        *REASONING_EFFS,
        "--skills",
        *SKILLS_MODES,
        "--models",
        *MODELS,
        "--max-workers",
        str(max_workers),
        "--seed",
        str(seed),
    ]
    if smoke:
        cmd.append("--smoke")
    run(cmd, "STEP 1 - Full evaluation matrix")


def build_visualization_commands() -> list[tuple[list[str], str]]:
    commands = []

    script = os.path.join(ROOT, "viz", "plot_accuracy_all_conditions.py")
    for image_type in IMAGE_TYPES:
        for reasoning_mode in REASONING_EFFS:
            for skills_mode in VIZ_SKILLS_MODES:
                commands.append(
                    (
                        [
                            PYTHON,
                            script,
                            "--image-type",
                            image_type,
                            "--reasoning-mode",
                            reasoning_mode,
                            "--skills-mode",
                            skills_mode,
                        ],
                        f"plot_accuracy_all_conditions | {image_type} | {reasoning_mode} | {skills_mode}",
                    )
                )

    script = os.path.join(ROOT, "viz", "plot_accuracy_by_blur.py")
    for image_type in IMAGE_TYPES:
        for reasoning_mode in REASONING_EFFS:
            for skills_mode in VIZ_SKILLS_MODES:
                commands.append(
                    (
                        [
                            PYTHON,
                            script,
                            "--image-type",
                            image_type,
                            "--blur-levels",
                            *BLUR_LEVELS,
                            "--reasoning-mode",
                            reasoning_mode,
                            "--skills-mode",
                            skills_mode,
                        ],
                        f"plot_accuracy_by_blur | {image_type} | {reasoning_mode} | {skills_mode}",
                    )
                )

    script = os.path.join(ROOT, "viz", "plot_accuracy_heatmap.py")
    for image_type in IMAGE_TYPES:
        for blur_level in BLUR_LEVELS:
            for reasoning_mode in REASONING_EFFS:
                for skills_mode in VIZ_SKILLS_MODES:
                    commands.append(
                        (
                            [
                                PYTHON,
                                script,
                                "--image-type",
                                image_type,
                                "--blur-level",
                                blur_level,
                                "--reasoning-mode",
                                reasoning_mode,
                                "--skills-mode",
                                skills_mode,
                            ],
                            f"plot_accuracy_heatmap | {image_type} | {blur_level} | {reasoning_mode} | {skills_mode}",
                        )
                    )

    script = os.path.join(ROOT, "viz", "plot_accuracy_heavy_blur_high.py")
    for image_type in IMAGE_TYPES:
        for skills_mode in VIZ_SKILLS_MODES:
            commands.append(
                (
                    [
                        PYTHON,
                        script,
                        "--image-type",
                        image_type,
                        "--mode-a",
                        "medium",
                        "--mode-b",
                        "high",
                        "--skills-mode",
                        skills_mode,
                    ],
                    f"plot_accuracy_heavy_blur_high | {image_type} | {skills_mode} | medium vs high",
                )
            )

    script = os.path.join(ROOT, "viz", "plot_token_time_stats.py")
    for image_type in IMAGE_TYPES:
        for blur_level in BLUR_LEVELS:
            for reasoning_mode in REASONING_EFFS:
                for skills_mode in VIZ_SKILLS_MODES:
                    commands.append(
                        (
                            [
                                PYTHON,
                                script,
                                "--image-type",
                                image_type,
                                "--blur-level",
                                blur_level,
                                "--reasoning-mode",
                                reasoning_mode,
                                "--skills-mode",
                                skills_mode,
                            ],
                            f"plot_token_time_stats | {image_type} | {blur_level} | {reasoning_mode} | {skills_mode}",
                        )
                    )


    script = os.path.join(ROOT, "viz", "plot_comparative_summary.py")
    commands.append(
        (
            [PYTHON, script],
            "plot_comparative_summary | compact comparative dashboard",
        )
    )

    return commands


def step_visualize(viz_workers: int) -> None:
    commands = build_visualization_commands()
    workers = max(1, viz_workers)
    print(f"\n[INFO] Running {len(commands)} visualization commands with viz_workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_label = {executor.submit(run, cmd, label): label for cmd, label in commands}
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                future.result()
            except Exception as exc:
                print(f"\n[ERROR] Visualization failed: {label}\n{exc}")
                raise

    print(f"\n{'=' * 70}")
    print("  All visualizations saved to Results/res_vis/")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end reproducibility script for MS Paint Reasoning Evaluation.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation; generate visualizations from existing results only.")
    parser.add_argument("--smoke", action="store_true", help="Evaluation smoke-test mode: run on img1 only.")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent eval+judge jobs inside Eval_script.py.")
    parser.add_argument("--viz-workers", type=int, default=4, help="Concurrent visualization subprocesses.")
    parser.add_argument("--seed", type=int, default=12345, help="Best-effort seed for VLM and judge calls.")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("  MS Paint Reasoning Evaluation - Full Reproducibility Run")
    print("#" * 70)
    print(f"  Seed: {args.seed}")
    print(f"  Eval workers: {args.max_workers}")
    print(f"  Viz workers: {args.viz_workers}")

    try:
        if not args.skip_eval:
            step_evaluate(smoke=args.smoke, max_workers=args.max_workers, seed=args.seed)
        else:
            print("\n[--skip-eval] Skipping evaluation. Using existing Results/.")

        step_visualize(viz_workers=args.viz_workers)
    except Exception as exc:
        print(f"\n[ERROR] Reproducibility run failed: {exc}")
        sys.exit(1)

    print("\n" + "#" * 70)
    print("  Done. Results in:")
    print("    Results/dashboard_data/  - aggregated JSONs (canonical)")
    print("    Results/res_vis/         - all visualization PNGs")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
