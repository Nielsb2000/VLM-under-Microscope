#!/bin/bash

# =============================================================================
# Smoke Test: GPT-5.5 Baseline + SAM2 — Mazenav (5 samples)
# =============================================================================
# Quick end-to-end validation: 2 inference jobs (baseline, SAM2), then eval.
# Writes to isolated smoke-test folders so canonical outputs/ is never touched.
#
# Usage (from spatial_eval/ directory):
#   bash scripts/smoke_test_gpt55.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_FOLDER="outputs_smoke_test_gpt55"
EVAL_SUMMARY="eval_summary_smoke_test_gpt55"
MODEL="gpt-5.5"
FIRST_K=5
MAX_TOKENS=1024

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log() { echo -e "$(date +'%H:%M:%S') $*"; }

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY is not set.${NC}"; exit 1
fi

cd "${PROJECT_DIR}"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   GPT-5.5 Smoke Test — Mazenav  (first_k=${FIRST_K})${NC}"
echo -e "${BLUE}   Baseline + SAM2  →  ${OUT_FOLDER}/${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ── Phase 1: Inference ────────────────────────────────────────────────────────
log "${YELLOW}[1/3] BASELINE  model=${MODEL} task=mazenav mode=vqa${NC}"
uv run python inference_vlm.py \
    --model_path "${MODEL}" \
    --task mazenav \
    --mode vqa \
    --first_k "${FIRST_K}" \
    --max_new_tokens "${MAX_TOKENS}" \
    --output_folder "${OUT_FOLDER}"

log "${YELLOW}[2/3] SAM2      model=${MODEL} task=mazenav mode=vqa${NC}"
uv run python inference_vlm.py \
    --model_path "${MODEL}" \
    --task mazenav \
    --mode vqa \
    --first_k "${FIRST_K}" \
    --max_new_tokens "${MAX_TOKENS}" \
    --output_folder "${OUT_FOLDER}" \
    --use_sam2

# ── Phase 2: Evaluation ───────────────────────────────────────────────────────
log "${YELLOW}[3/3] Evaluating all outputs...${NC}"
uv run python evals/evaluation.py \
    --mode vqa --task mazenav \
    --output_folder "${OUT_FOLDER}" \
    --eval_summary_dir "${EVAL_SUMMARY}"

echo ""
echo -e "${GREEN}Smoke test complete.${NC}"
echo -e "  Outputs  : ${OUT_FOLDER}/MilaWang__SpatialEval/vqa/mazenav/"
echo -e "  Overlays : ${OUT_FOLDER}/MilaWang__SpatialEval/vqa/mazenav/sam2_overlays/  (SAM2 run)"
echo -e "  Eval     : ${EVAL_SUMMARY}/vqa/mazenav_acc.csv"
echo ""
echo -e "  Verify:"
echo -e "    Baseline JSONL  →  no sam2_* fields"
echo -e "    SAM2 JSONL      →  sam2_enabled=true, sam2_meta present"
echo -e "    Filenames       →  m-gpt-5.5_bare_*.jsonl  vs  m-gpt-5.5_bare_sam2_*.jsonl"
