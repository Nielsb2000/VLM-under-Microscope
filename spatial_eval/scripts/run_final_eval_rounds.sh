#!/bin/bash

# =============================================================================
# Final Evaluation Rounds 2 & 3 — non-overlapping 100-sample batches
# =============================================================================
# Round 1 (images 0-99)   → already run by run_final_eval.sh → outputs_final
# Round 2 (images 100-199) → outputs_final_r2
# Round 3 (images 200-299) → outputs_final_r3
#
# After all 3 rounds complete, run the stats script to get mean ± std.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="gpt-5.2"
TASKS=("mazenav" "spatialgrid" "spatialmap")
MODES=("vqa" "vtqa")
FIRST_K=100
MAX_TOKENS=1024

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "$(date +'%H:%M:%S') $*"; }

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY is not set.${NC}"; exit 1
fi

cd "${PROJECT_DIR}"

run_round() {
    local ROUND="$1"
    local OFFSET="$2"
    local OUT_FOLDER="outputs_final_r${ROUND}"
    local EVAL_DIR="eval_summary_final_r${ROUND}"
    local TOTAL=$(( ${#TASKS[@]} * ${#MODES[@]} * 2 ))
    local RUN=0

    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}   Round ${ROUND} / 3  (images ${OFFSET}–$((OFFSET+FIRST_K-1)))${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    for task in "${TASKS[@]}"; do
        for mode in "${MODES[@]}"; do
            RUN=$(( RUN + 1 ))
            log "${YELLOW}[${RUN}/${TOTAL}] BASELINE  task=${task} mode=${mode}${NC}"
            uv run python inference_vlm.py \
                --model_path "${MODEL}" --task "${task}" --mode "${mode}" \
                --first_k "${FIRST_K}" --offset_k "${OFFSET}" \
                --max_new_tokens "${MAX_TOKENS}" --output_folder "${OUT_FOLDER}"

            RUN=$(( RUN + 1 ))
            log "${YELLOW}[${RUN}/${TOTAL}] SKILLS    task=${task} mode=${mode}${NC}"
            uv run python inference_vlm.py \
                --model_path "${MODEL}" --task "${task}" --mode "${mode}" \
                --first_k "${FIRST_K}" --offset_k "${OFFSET}" \
                --max_new_tokens "${MAX_TOKENS}" --output_folder "${OUT_FOLDER}" \
                --use_skills
        done
    done

    log "${YELLOW}Evaluating round ${ROUND}...${NC}"
    for task in "${TASKS[@]}"; do
        for mode in "${MODES[@]}"; do
            uv run python evals/evaluation.py \
                --mode "${mode}" --task "${task}" \
                --output_folder "${OUT_FOLDER}" \
                --dataset_id MilaWang/SpatialEval \
                --eval_summary_dir "${EVAL_DIR}"
        done
    done

    log "${GREEN}Round ${ROUND} complete → ${EVAL_DIR}${NC}"
}

#run_round 2 100
#run_round 3 200

# ── Cross-round stats + plots ─────────────────────────────────────────────────
echo ""
log "${YELLOW}Computing cross-round statistics and plots...${NC}"
uv run python eval_summary/plot_skills_comparison_multi.py \
    --eval_dirs eval_summary_final eval_summary_final_r2 eval_summary_final_r3 \
    --tasks mazenav spatialgrid spatialmap \
    --out_dir eval_summary_final_stats

echo ""
echo -e "${GREEN}All done! Stats plots in eval_summary_final_stats/${NC}"
ye