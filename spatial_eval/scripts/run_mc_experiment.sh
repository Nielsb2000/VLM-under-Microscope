#!/usr/bin/env bash
# run_mc_experiment.sh
#
# Monte Carlo accuracy estimation across all 3 tasks × 4 skill configs.
# Each condition runs MC_RUNS iterations, each sampling FIRST_K random images
# per question type (= FIRST_K × 3 questions per iteration).
#
# 3 tasks × 4 configs × MC_RUNS iterations = 36 inference runs total.
#
# Example images in skills: highest dataset indices (safe separation from test pool).
#
# Usage (from spatial_eval/):
#   bash scripts/run_mc_experiment.sh
#   MC_RUNS=5 FIRST_K=20 bash scripts/run_mc_experiment.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL="${MODEL:-gpt-5.2}"
MODE="vqa"
FIRST_K="${FIRST_K:-10}"
MC_RUNS="${MC_RUNS:-3}"
MC_SEED="${MC_SEED:-42}"
OUTPUT_FOLDER="outputs"
EVAL_DIR="eval_summary"

echo "========================================================"
echo "  MC Experiment: tasks=mazenav,spatialgrid,spatialmap"
echo "  Model=$MODEL | mode=$MODE | first_k=$FIRST_K"
echo "  mc_runs=$MC_RUNS | mc_seed=$MC_SEED"
echo "========================================================"
echo

TASKS=("mazenav" "spatialgrid" "spatialmap")
CONFIGS=("baseline" "img-only" "img-qa" "img-context")
N_TASKS=${#TASKS[@]}
N_CONFIGS=${#CONFIGS[@]}
TOTAL=$(( N_TASKS * N_CONFIGS ))
COMBO=0

for TASK in "${TASKS[@]}"; do
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Task: $TASK"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for CONFIG in "${CONFIGS[@]}"; do
    COMBO=$(( COMBO + 1 ))
    echo
    echo "--- [$COMBO/$TOTAL] $TASK | $CONFIG ($MC_RUNS MC iterations) ---"

    if [ "$CONFIG" = "baseline" ]; then
      uv run python inference_vlm.py \
        --model_path "$MODEL" \
        --mode "$MODE" \
        --task "$TASK" \
        --first_k "$FIRST_K" \
        --mc_runs "$MC_RUNS" \
        --mc_seed "$MC_SEED" \
        --output_folder "$OUTPUT_FOLDER"
    else
      uv run python inference_vlm.py \
        --model_path "$MODEL" \
        --mode "$MODE" \
        --task "$TASK" \
        --first_k "$FIRST_K" \
        --mc_runs "$MC_RUNS" \
        --mc_seed "$MC_SEED" \
        --use_skills \
        --skills_variant "$CONFIG" \
        --output_folder "$OUTPUT_FOLDER"
    fi

    # Update the CSV after all MC iterations for this config
    uv run python evals/evaluation.py \
      --mode "$MODE" --task "$TASK" \
      --output_folder "$OUTPUT_FOLDER" \
      --eval_summary_dir "$EVAL_DIR"

    echo
  done
done

echo "========================================================"
echo "  All $TOTAL conditions complete."
echo "  Results in $OUTPUT_FOLDER and $EVAL_DIR"
echo "========================================================"
