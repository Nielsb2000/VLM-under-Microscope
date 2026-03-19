#!/usr/bin/env python3
"""
Generate the img-qa-val skill directories for Test 1 (validation test).

For each task (mazenav, spatialgrid, spatialmap):
  - Takes images 0–9 (first 10 unique image indices) from the dataset
  - For each image embeds ALL 3 question types (q0, q1, q2) with full Q&A
  - Creates skills_img_qa_val/skills/{task}/SKILL.md with 30 Q&A pairs total
  - Saves 10 image PNGs to assets/

The skill and the test run use the SAME images (deliberate contamination) to
validate that the skill mechanism can drive the model to near-100% accuracy.

Run from spatial_eval/:
    uv run python scripts/generate_img_qa_val_skill.py
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

# Ensure spatial_eval/ is on path when run from project root
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from datasets import load_dataset

TASKS = ["mazenav", "spatialgrid", "spatialmap"]
N_IMAGES = 10  # first 10 unique image indices (0–9)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_OUT_BASE = _MODELS_DIR / "skills_img_qa_val" / "skills"

_TASK_DISPLAY = {
    "mazenav":    "Maze Navigation",
    "spatialgrid": "Spatial Grid",
    "spatialmap":  "Spatial Map",
}


def _write_master_skill():
    """Copy master-skill from an existing variant — routing table is identical."""
    src = _MODELS_DIR / "skills_img_qa" / "skills" / "master-skill" / "SKILL.md"
    dst_dir = _OUT_BASE / "master-skill"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "SKILL.md"
    # Read and update description in frontmatter
    text = src.read_text()
    text = text.replace(
        "Routes spatial reasoning questions to image + Q&A few-shot variant",
        "Routes spatial reasoning questions to img-qa-val (validation) variant",
    )
    dst.write_text(text)
    print(f"  Copied master-skill → {dst}")


def _write_task_skill(task: str, items_by_img: dict):
    skill_dir = _OUT_BASE / task
    assets_dir = skill_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    display = _TASK_DISPLAY[task]
    lines = [
        "---",
        f"name: {task}",
        f"description: {display} validation examples — images 0-{N_IMAGES-1} with all "
        f"3 questions and answers embedded; read all examples then solve the question.",
        "---",
        "",
        f"# {display} — Validation Examples (Image + All Questions + Answers)",
        "",
        "These are SOLVED examples using the SAME images you will be tested on.",
        "Call `read_file` on each image path listed, study all Q&A pairs, then answer.",
        "",
    ]

    for img_idx in sorted(items_by_img.keys()):
        items = sorted(items_by_img[img_idx], key=lambda x: int(x["id"].split(".")[-1]))

        # Save image PNG (all items for same img_idx share the same image)
        img_filename = f"{task}_{img_idx}.png"
        img_path = assets_dir / img_filename
        if not img_path.exists():
            pil_img = items[0]["image"]
            pil_img.save(str(img_path))

        skill_img_path = f"skills/{task}/assets/{img_filename}"

        lines += [
            "---",
            "",
            f"## Image {img_idx}",
            "",
            f"Image: `{skill_img_path}`",
            "",
        ]

        for item in items:
            q_type = item["id"].split(".")[-1]
            lines += [
                f"### Question {q_type}",
                "",
                item["text"].strip(),
                "",
                f"**Answer: {item['oracle_full_answer']}**",
                "",
            ]

    lines += [
        "---",
        "",
        "Now apply what you've learned to the actual question.",
        "Give your final answer as: [Option Letter]. [Answer]",
        "Examples: `D. 0`, `A. Yes`, `C. 3`",
    ]

    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text("\n".join(lines))
    n_qa = sum(len(v) for v in items_by_img.values())
    print(f"  Written: {skill_path}  ({len(items_by_img)} images × {n_qa // len(items_by_img)} Q&A = {n_qa} pairs)")


def main():
    print("Loading dataset (vqa split)…")
    ds = load_dataset("MilaWang/SpatialEval", "vqa", split="test")

    _write_master_skill()

    # Single pass — collect ALL items for each task (no index filtering yet)
    # Image indices are NOT consecutive from 0 for all tasks
    # (e.g. spatialmap indices start at 2000)
    # We sort per-task and take the first N to match how offset_k works in inference.
    print("Collecting items (single pass)…")
    task_items: dict[str, dict[int, list]] = {t: defaultdict(list) for t in TASKS}
    for item in ds:
        item_id = item["id"]
        for task in TASKS:
            if task not in item_id:
                continue
            img_idx = int(item_id.split(".")[2])
            task_items[task][img_idx].append(item)
            break  # each item belongs to exactly one task

    for task in TASKS:
        print(f"\nGenerating {task}…")
        all_indices = sorted(task_items[task].keys())
        # Take the first N_IMAGES from the sorted index list (same slice as offset_k=0, first_k=N)
        first_n_indices = all_indices[:N_IMAGES]
        items_by_img = {idx: task_items[task][idx] for idx in first_n_indices}
        if len(items_by_img) < N_IMAGES:
            print(f"  [WARN] Only {len(items_by_img)} images found (expected {N_IMAGES})")
        _write_task_skill(task, items_by_img)

    print(f"\nDone. Output directory: {_OUT_BASE}")
    print("\nTo test (same 10 images as skill):")
    for task in TASKS:
        print(
            f"  uv run python inference_vlm.py --model_path gpt-5.2 --mode vqa "
            f"--task {task} --first_k 10 --offset_k 0 --mc_runs 3 --mc_seed 42 "
            f"--use_skills --skills_variant img-qa-val --workers 8 --output_folder outputs"
        )


if __name__ == "__main__":
    main()
