"""Regenerate malformed example_10–29.txt files for all three tasks.

The original files (added in commit 171722f) were generated with broken
formatting — only 1 option per question instead of 4, and a garbled answer
format.  This script reads the correct data directly from the HuggingFace
dataset and rewrites the .txt files in the canonical format matching
example_0–9 (created by generate_preload_examples.py).

RESUMABLE: already-correct files are skipped automatically.  If the script
is interrupted (WSL disconnect, etc.) just re-run it — it will pick up where
it left off.

Run from project root:
    uv run python spatial_eval/models/regenerate_examples_10_29.py

    # force overwrite all (ignore skip logic):
    uv run python spatial_eval/models/regenerate_examples_10_29.py --force

    # single task only:
    uv run python spatial_eval/models/regenerate_examples_10_29.py --task mazenav
"""

import argparse
import os
import re
import time

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "skills_img_qa_val_v2", "examples")

TASKS = ["mazenav", "spatialgrid", "spatialmap"]
EXAMPLE_RANGE = range(10, 30)

# spatialmap items in the HF dataset start at img_idx=2000 (example N → img 2000+N)
# mazenav and spatialgrid start at img_idx=0 (example N → img N)
TASK_IMG_BASE = {
    "mazenav":     0,
    "spatialgrid": 0,
    "spatialmap":  2000,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_malformed(path: str) -> bool:
    """Return True if the file looks like the broken format (only 1 option)."""
    try:
        text = open(path).read()
        # Good files have at least 3 pipe-separated options per question
        return "| B." not in text
    except OSError:
        return True  # missing → treat as malformed


def extract_question_and_options(text: str) -> tuple[str, str]:
    """Extract the question stem and pipe-separated options from dataset text."""
    m = re.search(
        r"Please answer the following (?:multiple-choice )?question based on the provided information\.\s*"
        r"(.+?)\s*Available options:\s*(.+)",
        text,
        re.DOTALL,
    )
    if not m:
        raise ValueError(f"Could not parse text:\n{text[:300]}")

    question = m.group(1).strip()
    opts_raw = m.group(2).strip()

    opt_parts = re.findall(r"([A-D])\.\s*([^\n]+)", opts_raw)
    if not opt_parts:
        raise ValueError(f"Could not parse options from:\n{opts_raw}")

    cleaned = [(letter, val.rstrip(".").strip()) for letter, val in opt_parts]
    options_str = " | ".join(f"{letter}. {val}" for letter, val in cleaned)
    return question, options_str


def make_example_txt(n: int, questions: list[tuple[str, str, str]]) -> str:
    """questions: list of (question_text, options_str, answer_full) triples."""
    lines = [f"# Example image {n}", ""]
    for i, (question, options, answer) in enumerate(questions):
        lines.append(f"Question {i + 1} (q{i}): {question}")
        lines.append(f"Options: {options}")
        lines.append(f"Answer: {answer}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Step 1: fetch only the rows we need via HF REST API (no dataset download)
# ---------------------------------------------------------------------------

HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=MilaWang%2FSpatialEval&config=vqa&split=test"
    "&offset={offset}&length={length}"
)
BATCH_SIZE = 100


def _build_target_ids(tasks: list[str]) -> set[str]:
    ids = set()
    for task in tasks:
        base = TASK_IMG_BASE[task]
        for img in range(10, 30):
            for q in range(3):
                ids.add(f"{task}.vqa.{base + img}.{q}")
    return ids


def load_lookup(tasks: list[str]) -> dict:
    """Fetch only the 180 rows we need from the HF REST API, in small batches."""
    target_ids = _build_target_ids(tasks)
    remaining = set(target_ids)
    lookup: dict[tuple[str, int, int], dict] = {}

    print(
        f"Step 1/2 — Fetching {len(target_ids)} rows from HF REST API "
        f"(batch={BATCH_SIZE}, no local download) …",
        flush=True,
    )

    offset = 0
    while remaining:
        url = HF_ROWS_URL.format(offset=offset, length=BATCH_SIZE)
        for attempt in range(6):
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                break
            except Exception as exc:
                wait = 15 * (attempt + 1)
                print(f"  [retry {attempt+1}/6] {exc} — waiting {wait}s …", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to fetch offset={offset} after 6 attempts")

        data = resp.json()
        rows = data.get("rows", [])
        if not rows:
            print(f"  No more rows at offset={offset}, stopping.", flush=True)
            break

        for entry in rows:
            row = entry["row"]
            row_id = row.get("id", "")
            if row_id in remaining:
                parts = row_id.split(".")
                task_name = parts[0].lower()
                img_idx_raw = int(parts[2])
                q_idx = int(parts[3])
                # normalise back to example index (strip the task base)
                example_idx = img_idx_raw - TASK_IMG_BASE.get(task_name, 0)
                lookup[(task_name, example_idx, q_idx)] = row
                remaining.discard(row_id)

        print(
            f"  offset={offset:>6}  found so far={len(lookup)}/{len(target_ids)}"
            f"  remaining={len(remaining)}",
            flush=True,
        )
        offset += BATCH_SIZE
        time.sleep(2)  # polite delay to avoid 429 rate limits

        if not remaining:
            print("  All target rows found — stopping early.", flush=True)
            break

    if remaining:
        print(f"  [WARN] {len(remaining)} items not found: {sorted(remaining)[:5]} …", flush=True)

    print(f"  Fetched {len(lookup)} items total.", flush=True)
    return lookup


# ---------------------------------------------------------------------------
# Step 2: write files
# ---------------------------------------------------------------------------

def regenerate_task(task: str, lookup: dict, force: bool) -> tuple[int, int]:
    """Write example_10–29.txt for one task. Returns (written, skipped)."""
    out_dir = os.path.join(BASE_DIR, task)
    os.makedirs(out_dir, exist_ok=True)

    written = skipped = 0
    for n in EXAMPLE_RANGE:
        txt_path = os.path.join(out_dir, f"example_{n}.txt")

        if not force and os.path.exists(txt_path) and not _is_malformed(txt_path):
            print(f"  [skip]  {task}/example_{n}.txt (already correct)", flush=True)
            skipped += 1
            continue

        questions = []
        for q_idx in range(3):
            key = (task, n, q_idx)
            if key not in lookup:
                print(f"  [WARN] dataset item not found: {key}", flush=True)
                continue
            item = lookup[key]
            question, options = extract_question_and_options(item["text"])
            answer = item["oracle_full_answer"].strip()
            questions.append((question, options, answer))

        if len(questions) != 3:
            print(f"  [ERROR] only {len(questions)}/3 questions found for {task}/example_{n}, skipping", flush=True)
            continue

        # Write atomically: temp file → rename
        tmp_path = txt_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(make_example_txt(n, questions))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, txt_path)

        print(f"  [wrote] {task}/example_{n}.txt", flush=True)
        written += 1

    return written, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Overwrite files even if they look correct already.")
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="Regenerate only one task.")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS

    lookup = load_lookup(tasks)

    print(f"\nStep 2/2 — Writing example files (force={args.force}) …", flush=True)
    total_written = total_skipped = 0
    for task in tasks:
        print(f"\n--- {task} ---", flush=True)
        w, s = regenerate_task(task, lookup, force=args.force)
        total_written += w
        total_skipped += s
        print(f"  {task}: {w} written, {s} skipped.", flush=True)

    print(f"\nDone. Total written={total_written}, skipped={total_skipped}.", flush=True)


if __name__ == "__main__":
    main()
