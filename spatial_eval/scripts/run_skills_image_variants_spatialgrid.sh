#!/usr/bin/env bash
# run_skills_image_variants_spatialgrid.sh
#
# Analyze the effect of image-only skill types on the spatialgrid task.
# Runs 4 conditions (baseline + 3 image skill variants) × VQA mode,
# 10 images × 3 question types = 30 questions per run.
#
# Test samples:  question-type groups 0-9  (first_k=10)
# Example images in skills: grid indices 497, 498, 499  (safe separation)
#
# Usage: cd spatial_eval && bash scripts/run_skills_image_variants_spatialgrid.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL="${MODEL:-gpt-5.2}"
TASK="spatialgrid"
MODE="vqa"
FIRST_K=10
OUTPUT_FOLDER="outputs"
EVAL_DIR="eval_summary"

echo "=== Spatial-Grid image-skill variants experiment ==="
echo "Model: $MODEL | Task: $TASK | Mode: $MODE | first_k: $FIRST_K"
echo

# 1. Baseline — no skills
echo "--- Run 1/4: baseline (no skills) ---"
uv run python inference_vlm.py \
  --model_path "$MODEL" \
  --mode "$MODE" \
  --task "$TASK" \
  --first_k "$FIRST_K" \
  --output_folder "$OUTPUT_FOLDER"

uv run python evals/evaluation.py \
  --mode "$MODE" --task "$TASK" \
  --output_folder "$OUTPUT_FOLDER" \
  --eval_summary_dir "$EVAL_DIR"

# 2. Image-only skill (paths to example images, no extra context)
echo "--- Run 2/4: img-only skill ---"
uv run python inference_vlm.py \
  --model_path "$MODEL" \
  --mode "$MODE" \
  --task "$TASK" \
  --first_k "$FIRST_K" \
  --use_skills \
  --skills_variant img-only \
  --output_folder "$OUTPUT_FOLDER"

uv run python evals/evaluation.py \
  --mode "$MODE" --task "$TASK" \
  --output_folder "$OUTPUT_FOLDER" \
  --eval_summary_dir "$EVAL_DIR"

# 3. Image + Q&A skill (biased few-shot — shows question + correct answer)
echo "--- Run 3/4: img-qa skill (biased) ---"
uv run python inference_vlm.py \
  --model_path "$MODEL" \
  --mode "$MODE" \
  --task "$TASK" \
  --first_k "$FIRST_K" \
  --use_skills \
  --skills_variant img-qa \
  --output_folder "$OUTPUT_FOLDER"

uv run python evals/evaluation.py \
  --mode "$MODE" --task "$TASK" \
  --output_folder "$OUTPUT_FOLDER" \
  --eval_summary_dir "$EVAL_DIR"

# 4. Image + context skill (domain explanation, unbiased)
echo "--- Run 4/4: img-context skill (unbiased) ---"
uv run python inference_vlm.py \
  --model_path "$MODEL" \
  --mode "$MODE" \
  --task "$TASK" \
  --first_k "$FIRST_K" \
  --use_skills \
  --skills_variant img-context \
  --output_folder "$OUTPUT_FOLDER"

uv run python evals/evaluation.py \
  --mode "$MODE" --task "$TASK" \
  --output_folder "$OUTPUT_FOLDER" \
  --eval_summary_dir "$EVAL_DIR"

echo
echo "=== All runs complete. Results in $OUTPUT_FOLDER and $EVAL_DIR ==="
echo "Output filenames:"
ls "$OUTPUT_FOLDER/MilaWang__SpatialEval/$MODE/$TASK/"m-*.jsonl 2>/dev/null | tail -8 || true
