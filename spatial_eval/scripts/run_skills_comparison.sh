#!/bin/bash

# =============================================================================
# GPT-5.2 Skills vs Baseline Comparison — SpatialEval
# =============================================================================
# Runs inference for gpt-5.2 WITH and WITHOUT spatial reasoning skills
# across 2 tasks (spatialgrid, spatialmap) × 2 modes (vqa, vtqa)
# = 8 inference runs, then 4 evaluation runs
#
# Skills variant  : --use_skills  → DeepAgentGPT (reads task SKILL.md files)
# Baseline variant: (no flag)     → GPT4Vision (plain ChatCompletion)
#
# Output filenames:
#   m-gpt-5.2_bare_skills_{timestamp}.jsonl   ← skills
#   m-gpt-5.2_bare_{timestamp}.jsonl          ← baseline
# =============================================================================

set -e

# ── Colour codes ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"   # = spatial_eval/
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/skills_comparison_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="gpt-5.2"
TASKS=("spatialgrid" "spatialmap")
MODES=("vqa" "vtqa")
FIRST_K=100
MAX_NEW_TOKENS=1024

# ── Counters ──────────────────────────────────────────────────────────────────
TOTAL_INF_RUNS=$(( ${#TASKS[@]} * ${#MODES[@]} * 2 ))   # × 2 for skills/baseline
TOTAL_EVAL_RUNS=$(( ${#TASKS[@]} * ${#MODES[@]} ))
CURRENT_RUN=0
INF_SUCCESS=0
INF_FAIL=0

# ── Helpers ───────────────────────────────────────────────────────────────────
log() {
    local level="$1"; shift
    echo -e "$(date +'%Y-%m-%d %H:%M:%S') [${level}] $*" | tee -a "$LOG_FILE"
}

check_openai_key() {
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        log "ERROR" "OPENAI_API_KEY is not set. Export it before running this script."
        exit 1
    fi
    log "INFO" "OPENAI_API_KEY found."
}

run_inference() {
    local task="$1"
    local mode="$2"
    local use_skills="$3"   # "yes" or "no"

    CURRENT_RUN=$(( CURRENT_RUN + 1 ))
    local skills_label; [[ "$use_skills" == "yes" ]] && skills_label="WITH skills" || skills_label="BASELINE (no skills)"
    local skills_flag;  [[ "$use_skills" == "yes" ]] && skills_flag="--use_skills" || skills_flag=""

    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Run ${CURRENT_RUN}/${TOTAL_INF_RUNS} │ task=${YELLOW}${task}${BLUE} │ mode=${YELLOW}${mode}${BLUE} │ ${YELLOW}${skills_label}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    log "INFO" "Starting: model=${MODEL} task=${task} mode=${mode} skills=${use_skills}"

    local cmd="uv run python inference_vlm.py \
        --model_path \"${MODEL}\" \
        --task \"${task}\" \
        --mode \"${mode}\" \
        --first_k ${FIRST_K} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --output_folder outputs_100 \
        ${skills_flag}"

    if eval $cmd; then
        log "INFO" "SUCCESS: ${task}/${mode} ${skills_label}"
        INF_SUCCESS=$(( INF_SUCCESS + 1 ))
    else
        log "ERROR" "FAILED: ${task}/${mode} ${skills_label}"
        INF_FAIL=$(( INF_FAIL + 1 ))
    fi
}

run_evaluation() {
    local task="$1"
    local mode="$2"

    log "INFO" "Evaluating: task=${task} mode=${mode}"
    echo -e "${YELLOW}Evaluating ${task} / ${mode} ...${NC}"

    if uv run python evals/evaluation.py \
        --mode "${mode}" \
        --task "${task}" \
        --output_folder outputs_100/ \
        --dataset_id MilaWang/SpatialEval \
        --eval_summary_dir eval_summary; then
        log "INFO" "Eval OK: ${task}/${mode}"
    else
        log "WARN" "Eval FAILED: ${task}/${mode} (continuing)"
    fi
}

# =============================================================================
# Main
# =============================================================================

log "INFO" "======================================================="
log "INFO" "GPT-5.2 Skills vs Baseline — SpatialEval Comparison"
log "INFO" "======================================================="
log "INFO" "Model       : ${MODEL}"
log "INFO" "Tasks       : ${TASKS[*]}"
log "INFO" "Modes       : ${MODES[*]}"
log "INFO" "first_k     : ${FIRST_K}"
log "INFO" "Inference runs : ${TOTAL_INF_RUNS}  (${TOTAL_EVAL_RUNS} eval runs)"
log "INFO" "Log         : ${LOG_FILE}"
log "INFO" "======================================================="

check_openai_key

# Change into the spatial_eval project directory so relative paths work
cd "${PROJECT_DIR}" || { log "ERROR" "Cannot cd to ${PROJECT_DIR}"; exit 1; }

# ── Phase 1: Inference ────────────────────────────────────────────────────────
echo ""
log "INFO" "=== PHASE 1: INFERENCE ==="
echo ""

for task in "${TASKS[@]}"; do
    for mode in "${MODES[@]}"; do
        run_inference "$task" "$mode" "no"    # baseline first
        run_inference "$task" "$mode" "yes"   # then with skills
    done
done

echo ""
echo -e "${GREEN}Inference phase complete — ${INF_SUCCESS}/${TOTAL_INF_RUNS} succeeded, ${INF_FAIL} failed.${NC}"
log "INFO" "Inference done: success=${INF_SUCCESS} fail=${INF_FAIL}"

# ── Phase 2: Evaluation ───────────────────────────────────────────────────────
echo ""
log "INFO" "=== PHASE 2: EVALUATION ==="
echo ""

for task in "${TASKS[@]}"; do
    for mode in "${MODES[@]}"; do
        run_evaluation "$task" "$mode"
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Comparison run complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Inference:  ${INF_SUCCESS}/${TOTAL_INF_RUNS} succeeded  (${INF_FAIL} failed)"
echo ""
echo "  Outputs  :  ${PROJECT_DIR}/outputs/MilaWang__SpatialEval/"
echo "  Summaries:  ${PROJECT_DIR}/eval_summary/   (accuracy CSVs per task/mode)"
echo "  Log      :  ${LOG_FILE}"
echo ""
echo "To compare accuracy, check e.g.:"
echo "  cat ${PROJECT_DIR}/eval_summary/vqa/mazenav_acc.csv"
echo ""

log "INFO" "Done."

if [[ $INF_FAIL -gt 0 ]]; then
    exit 1
else
    exit 0
fi
