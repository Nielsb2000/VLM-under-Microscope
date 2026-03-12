#!/bin/bash

# =============================================================================
# Main Experiment: GPT-5.2 Skills vs Baseline — All 3 Tasks, 100 samples
# =============================================================================
# 3 tasks × 2 modes × 2 variants = 12 inference runs
# Then 6 evaluation runs + 3 comparison plots (one per task)
#
# Writes to: outputs/  and  eval_summary/
#
# Usage (from spatial_eval/ directory):
#   bash scripts/run_experiment.sh
#
# See scripts/run_experiment_rounds.sh for multi-round statistical evaluation.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

OUT_FOLDER="outputs"
EVAL_DIR="eval_summary"
MODEL="gpt-5.2"
FIRST_K=100
MAX_TOKENS=1024
TASKS=("mazenav" "spatialgrid" "spatialmap")
MODES=("vqa" "vtqa")

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log() { echo -e "$(date +'%H:%M:%S') $*"; }

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY is not set.${NC}"; exit 1
fi

cd "${PROJECT_DIR}"

TOTAL=$(( ${#TASKS[@]} * ${#MODES[@]} * 2 ))
RUN=0

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Final Evaluation  (first_k=${FIRST_K}, model=${MODEL})${NC}"
echo -e "${BLUE}   Tasks: ${TASKS[*]}${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ── Phase 1: Inference ────────────────────────────────────────────────────────
for task in "${TASKS[@]}"; do
    for mode in "${MODES[@]}"; do
        RUN=$(( RUN + 1 ))
        log "${YELLOW}[$RUN/$TOTAL] BASELINE  task=${task} mode=${mode}${NC}"
        uv run python inference_vlm.py \
            --model_path "${MODEL}" \
            --task "${task}" \
            --mode "${mode}" \
            --first_k "${FIRST_K}" \
            --max_new_tokens "${MAX_TOKENS}" \
            --output_folder "${OUT_FOLDER}"

        RUN=$(( RUN + 1 ))
        log "${YELLOW}[$RUN/$TOTAL] SKILLS    task=${task} mode=${mode}${NC}"
        uv run python inference_vlm.py \
            --model_path "${MODEL}" \
            --task "${task}" \
            --mode "${mode}" \
            --first_k "${FIRST_K}" \
            --max_new_tokens "${MAX_TOKENS}" \
            --output_folder "${OUT_FOLDER}" \
            --use_skills
    done
done

# ── Phase 2: Evaluation ───────────────────────────────────────────────────────
echo ""
log "${YELLOW}Evaluating all runs...${NC}"

for task in "${TASKS[@]}"; do
    for mode in "${MODES[@]}"; do
        log "  eval: task=${task} mode=${mode}"
        uv run python evals/evaluation.py \
            --mode "${mode}" \
            --task "${task}" \
            --output_folder "${OUT_FOLDER}" \
            --dataset_id MilaWang/SpatialEval \
            --eval_summary_dir "${EVAL_DIR}"
    done
done

# ── Phase 3: Plots (one per task) ─────────────────────────────────────────────
echo ""
log "${YELLOW}Generating comparison plots...${NC}"

for task in "${TASKS[@]}"; do
    uv run python eval_summary/plot_skills_comparison.py \
        --eval_summary_dir "${EVAL_DIR}" \
        --task "${task}" \
        --out_dir "${EVAL_DIR}" \
        --first_k "${FIRST_K}"
done

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   Done! Results in ${PROJECT_DIR}/${EVAL_DIR}/             ${NC}"
echo -e "${GREEN}   Plots: mazenav_skills_comparison.png                   ${NC}"
echo -e "${GREEN}          spatialgrid_skills_comparison.png               ${NC}"
echo -e "${GREEN}          spatialmap_skills_comparison.png                ${NC}"
echo -e "${GREEN}   Accuracy CSVs: {vqa,vtqa}/{task}_acc.csv               ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
