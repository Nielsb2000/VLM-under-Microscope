#!/bin/bash

# =============================================================================
# GPT-5.5 Experiment: Baseline + SAM2 — All 3 Tasks, 100 samples, 5 runs
# =============================================================================
# Matches the exact gpt-5.2 protocol:
#   3 tasks × 2 configs (baseline, SAM2) × 5 runs = 30 JSONL files total
#
# Writes to: outputs/  and  eval_summary/  (canonical folders)
#
# Usage (from spatial_eval/ directory):
#   bash scripts/run_gpt55_experiment.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

OUT_FOLDER="outputs"
EVAL_DIR="eval_summary"
MODEL="gpt-5.5"
MODE="vqa"
FIRST_K=100
RUNS=5
SAM2_MODEL="facebook/sam2.1-hiera-base-plus"
WORKERS=4
TASKS=("mazenav" "spatialgrid" "spatialmap")

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log() { echo -e "$(date +'%H:%M:%S') $*"; }

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY is not set.${NC}"; exit 1
fi

cd "${PROJECT_DIR}"

TOTAL=$(( ${#TASKS[@]} * 2 ))
RUN=0

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   GPT-5.5 Experiment  (runs=${RUNS}, first_k=${FIRST_K}, mode=${MODE})${NC}"
echo -e "${BLUE}   Tasks  : ${TASKS[*]}${NC}"
echo -e "${BLUE}   Configs: baseline, SAM2${NC}"
echo -e "${BLUE}   Output : ${OUT_FOLDER}/${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ── Phase 1: Inference + incremental eval ────────────────────────────────────
for task in "${TASKS[@]}"; do
    echo ""
    echo -e "${BLUE}──────────────────── Task: ${task} ────────────────────${NC}"

    # ── Baseline ──
    RUN=$(( RUN + 1 ))
    log "${YELLOW}[$RUN/$TOTAL] BASELINE  task=${task}  (${RUNS} runs × first_k=${FIRST_K})${NC}"
    uv run python inference_vlm.py \
        --model_path "${MODEL}" \
        --task "${task}" \
        --mode "${MODE}" \
        --first_k "${FIRST_K}" \
        --runs "${RUNS}" \
        --workers "${WORKERS}" \
        --max_new_tokens "${MAX_TOKENS}" \
        --output_folder "${OUT_FOLDER}"

    uv run python evals/evaluation.py \
        --mode "${MODE}" --task "${task}" \
        --output_folder "${OUT_FOLDER}" \
        --eval_summary_dir "${EVAL_DIR}"

    # ── SAM2 ──
    RUN=$(( RUN + 1 ))
    log "${YELLOW}[$RUN/$TOTAL] SAM2      task=${task}  (${RUNS} runs × first_k=${FIRST_K})${NC}"
    uv run python inference_vlm.py \
        --model_path "${MODEL}" \
        --task "${task}" \
        --mode "${MODE}" \
        --first_k "${FIRST_K}" \
        --runs "${RUNS}" \
        --workers "${WORKERS}" \
        --max_new_tokens "${MAX_TOKENS}" \
        --output_folder "${OUT_FOLDER}" \
        --use_sam2 \
        --sam2_model_id "${SAM2_MODEL}"

    uv run python evals/evaluation.py \
        --mode "${MODE}" --task "${task}" \
        --output_folder "${OUT_FOLDER}" \
        --eval_summary_dir "${EVAL_DIR}"

    echo ""
done

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   GPT-5.5 experiment complete!${NC}"
echo -e "${GREEN}   Results : ${OUT_FOLDER}/MilaWang__SpatialEval/${MODE}/  (30 JSONL files)${NC}"
echo -e "${GREEN}   Eval    : ${EVAL_DIR}/${MODE}/  (per-task acc CSV + JSONL summaries)${NC}"
echo -e "${GREEN}   Overlays: ${OUT_FOLDER}/MilaWang__SpatialEval/${MODE}/{task}/sam2_overlays/${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
