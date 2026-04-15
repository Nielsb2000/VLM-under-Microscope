#!/bin/bash

# =============================================================================
# Img-Only-Tool Scaling: baseline vs img-only-tool-n3/n10/n30 — MazeNav
# =============================================================================
# Runs 3 MC rounds (30 questions each, offset 30) for 4 conditions to measure
# how providing N image-only examples via the read_example tool affects accuracy.
#
# Conditions:
#   baseline           — no skills, MC runs
#   img-only-tool-n3   — 3 example images via tool, test at offset 30
#   img-only-tool-n10  — 10 example images via tool, test at offset 30
#   img-only-tool-n30  — 30 example images via tool, test at offset 30
#
# Offset 30 ensures no example-test overlap for the largest variant (n30).
#
# Usage (from spatial_eval/ directory):
#   bash scripts/run_img_only_tool.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="gpt-5.2"
TASK="mazenav"
MODE="vqa"
FIRST_K=30
OFFSET_K=30
MC_RUNS=3
MAX_TOKENS=1024
OUT_FOLDER="outputs"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "$(date +'%H:%M:%S') $*"; }

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY is not set.${NC}"; exit 1
fi

cd "${PROJECT_DIR}"

echo ""
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Img-Only-Tool Scaling — ${TASK} (first_k=${FIRST_K}, offset=${OFFSET_K}, mc=${MC_RUNS}, parallel)${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
echo ""

PIDS=()
LOG_DIR="logs/img_only_tool_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "${LOG_DIR}"

run_condition() {
    local label="$1"; shift
    local log_file="${LOG_DIR}/${label}.log"
    log "${YELLOW}[${label}]${NC} starting → ${log_file}"
    uv run python inference_vlm.py "$@" > "${log_file}" 2>&1 &
    PIDS+=($!)
}

# ── Launch all 4 conditions in parallel, staggered 60s to avoid OOM ──────────
# (4 simultaneous dataset loads of 1.4 GB each causes OOM; staggering lets each
# process finish loading before the next one starts reading from disk)
run_condition "BASELINE" \
    --model_path "${MODEL}" --task "${TASK}" --mode "${MODE}" \
    --first_k "${FIRST_K}" --offset_k "${OFFSET_K}" --mc_runs "${MC_RUNS}" \
    --max_new_tokens "${MAX_TOKENS}" --output_folder "${OUT_FOLDER}"
sleep 60

run_condition "IMG-ONLY-N3" \
    --model_path "${MODEL}" --task "${TASK}" --mode "${MODE}" \
    --first_k "${FIRST_K}" --offset_k "${OFFSET_K}" --mc_runs "${MC_RUNS}" \
    --max_new_tokens "${MAX_TOKENS}" --output_folder "${OUT_FOLDER}" \
    --use_skills --skills_variant img-only-tool-n3
sleep 60

run_condition "IMG-ONLY-N10" \
    --model_path "${MODEL}" --task "${TASK}" --mode "${MODE}" \
    --first_k "${FIRST_K}" --offset_k "${OFFSET_K}" --mc_runs "${MC_RUNS}" \
    --max_new_tokens "${MAX_TOKENS}" --output_folder "${OUT_FOLDER}" \
    --use_skills --skills_variant img-only-tool-n10
sleep 60

run_condition "IMG-ONLY-N30" \
    --model_path "${MODEL}" --task "${TASK}" --mode "${MODE}" \
    --first_k "${FIRST_K}" --offset_k "${OFFSET_K}" --mc_runs "${MC_RUNS}" \
    --max_new_tokens "${MAX_TOKENS}" --output_folder "${OUT_FOLDER}" \
    --use_skills --skills_variant img-only-tool-n30

# ── Wait for all jobs and report status ──────────────────────────────────────
echo ""
log "Waiting for ${#PIDS[@]} parallel jobs…"
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        log "${RED}Job PID ${pid} failed — check logs in ${LOG_DIR}/${NC}"
        FAILED=1
    fi
done

echo ""
if [[ "${FAILED}" -eq 0 ]]; then
    echo -e "${GREEN}All inference runs complete. Logs in ${LOG_DIR}/${NC}"
else
    echo -e "${RED}One or more jobs failed. Check ${LOG_DIR}/ for details.${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Generate the plot with:${NC}"
echo "  uv run python eval_summary/plot_img_only_tool.py --task mazenav --out_dir eval_summary/result_vis"
