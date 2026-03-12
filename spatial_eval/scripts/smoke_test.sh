#!/bin/bash

# =============================================================================
# Smoke Test: GPT-5.2 Skills vs Baseline — Mazenav (10 samples)
# =============================================================================
# Quick end-to-end sanity check: 4 inference jobs (2 modes × 2 variants),
# evaluation, and a comparison plot. Writes to isolated smoke-test folders
# so it never pollutes the canonical outputs/ or eval_summary/.
#
# Usage (from spatial_eval/ directory):
#   bash scripts/smoke_test.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"   # = spatial_eval/
OUT_FOLDER="outputs_smoke_test"
EVAL_SUMMARY="eval_summary_smoke_test"
MODEL="gpt-5.2"
FIRST_K=10
MAX_TOKENS=1024

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log() { echo -e "$(date +'%H:%M:%S') $*"; }

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY is not set.${NC}"; exit 1
fi

cd "${PROJECT_DIR}"

# ── Phase 1: Inference ────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Smoke Test — Mazenav  (first_k=${FIRST_K})${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

for mode in vqa vtqa; do
    log "${YELLOW}[BASELINE]${NC} mode=${mode}"
    uv run python inference_vlm.py \
        --model_path "${MODEL}" \
        --task mazenav \
        --mode "${mode}" \
        --first_k "${FIRST_K}" \
        --max_new_tokens "${MAX_TOKENS}" \
        --output_folder "${OUT_FOLDER}"

    log "${YELLOW}[SKILLS]${NC}   mode=${mode}"
    uv run python inference_vlm.py \
        --model_path "${MODEL}" \
        --task mazenav \
        --mode "${mode}" \
        --first_k "${FIRST_K}" \
        --max_new_tokens "${MAX_TOKENS}" \
        --output_folder "${OUT_FOLDER}" \
        --use_skills
done

# ── Phase 2: Evaluation ───────────────────────────────────────────────────────
echo ""
log "${YELLOW}Evaluating...${NC}"

for mode in vqa vtqa; do
    uv run python evals/evaluation.py \
        --mode "${mode}" \
        --task mazenav \
        --output_folder "${OUT_FOLDER}" \
        --dataset_id MilaWang/SpatialEval \
        --eval_summary_dir "${EVAL_SUMMARY}"
done

# ── Phase 3: Plot ─────────────────────────────────────────────────────────────
echo ""
log "${YELLOW}Generating comparison plot...${NC}"

uv run python eval_summary/plot_skills_comparison.py \
    --eval_summary_dir "${EVAL_SUMMARY}" \
    --task mazenav \
    --out_dir "${EVAL_SUMMARY}"

echo ""
echo -e "${GREEN}Smoke test complete!${NC}"
echo -e "  Outputs  : ${PROJECT_DIR}/${OUT_FOLDER}/"
echo -e "  Eval     : ${PROJECT_DIR}/${EVAL_SUMMARY}/"
echo -e "  Plot     : ${PROJECT_DIR}/${EVAL_SUMMARY}/mazenav_skills_comparison.png"
