#!/usr/bin/env python3
"""
Generate img-only range skill directories for Test 2 (example-count ablation).

Creates 4 skill variants each with a different number of example images:
  img-only-n10   → images 0–9    → test with --offset_k 10  --first_k 10
  img-only-n30   → images 0–29   → test with --offset_k 30  --first_k 10
  img-only-n50   → images 0–49   → test with --offset_k 50  --first_k 10
  img-only-n100  → images 0–99   → test with --offset_k 100 --first_k 10

Running inference at offset_k = N guarantees zero overlap between skill images
and test images.

Run from spatial_eval/:
    uv run python scripts/generate_img_only_range_skills.py
"""
import sys
from pathlib import Path
from collections import defaultdict

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from datasets import load_dataset

TASKS = ["mazenav", "spatialgrid", "spatialmap"]
N_VALUES = [10, 30, 50, 100]

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_TASK_DISPLAY = {
    "mazenav":    "Maze Navigation",
    "spatialgrid": "Spatial Grid",
    "spatialmap":  "Spatial Map",
}

_TASK_KEYWORDS = {
    "mazenav":    "maze, turn-counting",
    "spatialgrid": "grid, animal positions",
    "spatialmap":  "map, directional relationships",
}


def _write_master_skill(n: int):
    """Each variant gets its own master-skill (identical routing logic)."""
    src = _MODELS_DIR / "skills_img_only" / "skills" / "master-skill" / "SKILL.md"
    variant_dir = _MODELS_DIR / f"skills_img_only_n{n}" / "skills" / "master-skill"
    variant_dir.mkdir(parents=True, exist_ok=True)
    dst = variant_dir / "SKILL.md"
    text = src.read_text()
    text = text.replace(
        "Routes spatial reasoning questions to image-only few-shot variant",
        f"Routes spatial reasoning questions to img-only-n{n} variant ({n} example images)",
    )
    dst.write_text(text)


def _write_task_skill(task: str, n: int, unique_images: dict):
    """Write SKILL.md listing all N image paths (no Q&A — image-only format)."""
    display = _TASK_DISPLAY[task]
    variant_dir = _MODELS_DIR / f"skills_img_only_n{n}" / "skills" / task
    assets_dir = variant_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Save images
    saved = []
    for img_idx in sorted(unique_images.keys()):
        pil_img = unique_images[img_idx]
        img_filename = f"{task}_{img_idx}.png"
        img_path = assets_dir / img_filename
        if not img_path.exists():
            pil_img.save(str(img_path))
        saved.append((img_idx, f"skills/{task}/assets/{img_filename}"))

    # Build SKILL.md
    lines = [
        "---",
        f"name: {task}",
        f"description: {display} image-only examples ({n} images, indices 0-{n-1}); "
        f"read each example image to familiarise yourself with the visual format before answering.",
        "---",
        "",
        f"# {display} — Image-Only Examples ({n} images)",
        "",
        f"Before answering, read each example image below to familiarise yourself with "
        f"the visual format of the {_TASK_KEYWORDS[task]} task.",
        "Call `read_file` on each path listed.",
        "",
    ]

    for img_idx, skill_path in saved:
        lines.append(f"- `{skill_path}` — example image (index {img_idx})")

    lines += [
        "",
        "After reading the images, answer the question provided.",
        "Give your final answer as: [Option Letter]. [Answer]",
        "Examples: `D. 0`, `A. Yes`, `C. 3`",
    ]

    skill_path = variant_dir / "SKILL.md"
    skill_path.write_text("\n".join(lines))
    return len(saved)


def main():
    print("Loading dataset (vqa split)…")
    ds = load_dataset("MilaWang/SpatialEval", "vqa", split="test")

    max_n = max(N_VALUES)

    # Single pass — collect ALL items per task, indexed by img_idx.
    # Image indices are NOT consecutive from 0 for all tasks
    # (spatialmap indices start at ~2000). We sort per-task and take the
    # first N elements, mirroring exactly how offset_k works in inference.
    print("Collecting items (single pass)…")
    task_all_items: dict[str, dict[int, object]] = {t: {} for t in TASKS}
    for item in ds:
        item_id = item["id"]
        for task in TASKS:
            if task not in item_id:
                continue
            img_idx = int(item_id.split(".")[2])
            q_type = item_id.split(".")[-1]
            # Store the PIL image once per (task, img_idx) — q_type=0 as source
            if q_type == "0" and img_idx not in task_all_items[task]:
                task_all_items[task][img_idx] = item["image"]
            break

    # Pre-compute sorted per-task index lists
    task_sorted_indices: dict[str, list[int]] = {
        task: sorted(task_all_items[task].keys()) for task in TASKS
    }

    for n in N_VALUES:
        print(f"\n=== N={n} ===")
        _models_n_dir = _MODELS_DIR / f"skills_img_only_n{n}"
        print(f"  Output: {_models_n_dir}")

        _write_master_skill(n)

        for task in TASKS:
            # Take first N from per-task sorted index list (positions 0..N-1)
            first_n = task_sorted_indices[task][:n]
            if len(first_n) < n:
                print(f"  [WARN] {task}: only {len(first_n)} images available (needed {n})")
            subset = {idx: task_all_items[task][idx] for idx in first_n}
            count = _write_task_skill(task, n, subset)
            print(f"  {task}: {count} images written (indices {first_n[0]}–{first_n[-1]})")

    print("\nDone.")
    print("\nTest commands (run from spatial_eval/):")
    for n in N_VALUES:
        print(f"\n  # img-only-n{n} (offset_k={n}):")
        for task in TASKS:
            print(
                f"  uv run python inference_vlm.py --model_path gpt-5.2 --mode vqa "
                f"--task {task} --first_k 10 --offset_k {n} --mc_runs 3 --mc_seed 42 "
                f"--use_skills --skills_variant img-only-n{n} --workers 8 --output_folder outputs"
            )


if __name__ == "__main__":
    main()
