#!/bin/bash
# Parallel 8-worker, 5-run, 100-sample deterministic evaluation for MazeNav (VQA)
# Variants: no skills, img-only, img-context, img-qa

set -e
cd "$(dirname "$0")/.."

MODEL="gpt-5.2"
MODE="vqa"
TASK="mazenav"
FIRST_K=100
RUNS=5
WORKERS=8
OUTPUT_FOLDER="outputs"

# Variant: no skills
uv run python -u inference_vlm.py \
  --model_path "$MODEL" --mode "$MODE" --task "$TASK" \
  --first_k "$FIRST_K" --runs "$RUNS" --workers "$WORKERS" \
  --output_folder "$OUTPUT_FOLDER" \
  2>&1 | tee scripts/log_mazenav_noskills.txt

# Variant: img-only skill
uv run python -u inference_vlm.py \
  --model_path "$MODEL" --mode "$MODE" --task "$TASK" \
  --first_k "$FIRST_K" --runs "$RUNS" --workers "$WORKERS" \
  --use_skills --skills_variant img-only \
  --output_folder "$OUTPUT_FOLDER" \
  2>&1 | tee scripts/log_mazenav_imgonly.txt

# Variant: img-context skill
uv run python -u inference_vlm.py \
  --model_path "$MODEL" --mode "$MODE" --task "$TASK" \
  --first_k "$FIRST_K" --runs "$RUNS" --workers "$WORKERS" \
  --use_skills --skills_variant img-context \
  --output_folder "$OUTPUT_FOLDER" \
  2>&1 | tee scripts/log_mazenav_imgcontext.txt

# Variant: img-qa skill
uv run python -u inference_vlm.py \
  --model_path "$MODEL" --mode "$MODE" --task "$TASK" \
  --first_k "$FIRST_K" --runs "$RUNS" --workers "$WORKERS" \
  --use_skills --skills_variant img-qa \
  --output_folder "$OUTPUT_FOLDER" \
  2>&1 | tee scripts/log_mazenav_imgqa.txt
echo "All runs completed. Logs in scripts/log_mazenav_*.txt"
echo "All inference runs completed. Logs in scripts/log_mazenav_*.txt"

# --- Evaluation: compute accuracy summaries ---
uv run python evals/evaluation.py \
  --mode "$MODE" --task "$TASK" \
  --output_folder "$OUTPUT_FOLDER" \
  --eval_summary_dir eval_summary

# --- Plotting: skill variant comparison (single-run, not MC) ---
uv run python eval_summary/plot_image_skill_variants.py \
  --eval_summary_dir eval_summary --task "$TASK" --out_dir eval_summary/result_vis

echo "Evaluation and plotting complete. Plots in eval_summary/result_vis/"
