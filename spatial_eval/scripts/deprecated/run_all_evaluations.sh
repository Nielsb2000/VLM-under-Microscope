#!/bin/bash

# =============================================================================
# Multi-Model Spatial Evaluation - Batch Inference Script
# =============================================================================
# This script runs inference for 9 models across 4 tasks and 2 modes (vqa/vtqa)
# Processing 5 samples per task (--first_k 5)
# Total: 72 inference runs (9 models × 4 tasks × 2 modes)
# =============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/outputs"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/inference_${TIMESTAMP}.log"

# Create directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Inference parameters
TEMPERATURE=0.2
MAX_NEW_TOKENS=512
FIRST_K=100

# Define tasks and modes
TASKS=("spatialmap" "spatialgrid")
MODES=("vqa" "vtqa")

# Define API models (no CUDA needed)
API_MODELS=(
    #"gpt-4o"
    "gpt-5.1"
)

# Define local models (require CUDA)
LOCAL_MODELS=(
    #"liuhaotian/llava-v1.5-7b"
    # Mistral & Vicuna models disabled per user request
    # "liuhaotian/llava-v1.6-mistral-7b"
    # "liuhaotian/llava-v1.6-vicuna-7b"
    #"BAAI/bunny-phi-2-siglip"
    #"BAAI/bunny-v1_1-Llama-3-8b-v"
    #"BAAI/bunny-v1_0-3b"
)

# Counters
TOTAL_RUNS=$((${#API_MODELS[@]} * ${#TASKS[@]} * ${#MODES[@]} + ${#LOCAL_MODELS[@]} * ${#TASKS[@]} * ${#MODES[@]}))
CURRENT_RUN=0
SUCCESS_COUNT=0
FAILURE_COUNT=0

# Function to log messages
log() {
    local level="$1"
    shift
    local message="$@"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

# Function to display progress
show_progress() {
    local model="$1"
    local task="$2"
    local mode="$3"
    CURRENT_RUN=$((CURRENT_RUN + 1))
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Progress: ${CURRENT_RUN}/${TOTAL_RUNS} (Success: ${SUCCESS_COUNT}, Failed: ${FAILURE_COUNT})${NC}"
    echo -e "${BLUE}Model:  ${YELLOW}${model}${NC}"
    echo -e "${BLUE}Task:   ${YELLOW}${task}${NC}"
    echo -e "${BLUE}Mode:   ${YELLOW}${mode}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Function to run inference
run_inference() {
    local model_path="$1"
    local task="$2"
    local mode="$3"
    local device="$4"
    
    show_progress "$model_path" "$task" "$mode"
    
    log "INFO" "Running inference: model=${model_path}, task=${task}, mode=${mode}, device=${device}"
    
    # Build command
    local cmd="cd ${PROJECT_DIR} && uv run python inference_vlm.py \
        --model_path \"${model_path}\" \
        --task \"${task}\" \
        --mode \"${mode}\" \
        --first_k ${FIRST_K} \
        --temperature ${TEMPERATURE} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --output_folder \"${OUTPUT_DIR}\" \
        --device \"${device}\""
    
    # Execute with timeout (30 minutes per run)
    if timeout 1800 bash -c "$cmd" >> "$LOG_FILE" 2>&1; then
        log "SUCCESS" "✓ Completed: ${model_path} | ${task} | ${mode}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log "ERROR" "✗ Timeout (30min): ${model_path} | ${task} | ${mode}"
        else
            log "ERROR" "✗ Failed (exit code ${exit_code}): ${model_path} | ${task} | ${mode}"
        fi
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        return 1
    fi
}

# Function to check CUDA availability
check_cuda() {
    log "INFO" "Checking CUDA availability..."
    if cd "$PROJECT_DIR" && uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())" >> "$LOG_FILE" 2>&1; then
        log "SUCCESS" "CUDA check completed"
    else
        log "WARNING" "CUDA check failed - local models may not work"
    fi
}

# Function to check OpenAI API key
check_openai_key() {
    if [ -z "$OPENAI_API_KEY" ]; then
        log "WARNING" "OPENAI_API_KEY not set - API models (gpt-4o, gpt-5.1) will fail"
        return 1
    else
        log "SUCCESS" "OPENAI_API_KEY is set"
        return 0
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

log "INFO" "============================================"
log "INFO" "Starting Multi-Model Spatial Evaluation"
log "INFO" "============================================"
log "INFO" "Timestamp: ${TIMESTAMP}"
log "INFO" "Log file: ${LOG_FILE}"
log "INFO" "Output directory: ${OUTPUT_DIR}"
log "INFO" "Total inference runs: ${TOTAL_RUNS}"
log "INFO" "Tasks: ${TASKS[*]}"
log "INFO" "Modes: ${MODES[*]}"
log "INFO" "First K samples: ${FIRST_K}"
log "INFO" "============================================"

# Pre-flight checks
check_cuda
check_openai_key
HAS_OPENAI_KEY=$?


echo ""
log "INFO" "Starting API model inference runs..."
echo ""
for model in "${API_MODELS[@]}"; do
    if [ $HAS_OPENAI_KEY -ne 0 ]; then
        log "WARNING" "Skipping API model ${model} - no OPENAI_API_KEY"
        CURRENT_RUN=$((CURRENT_RUN + ${#TASKS[@]} * ${#MODES[@]}))
        FAILURE_COUNT=$((FAILURE_COUNT + ${#TASKS[@]} * ${#MODES[@]}))
        continue
    fi
    for task in "${TASKS[@]}"; do
        for mode in "${MODES[@]}"; do
            run_inference "$model" "$task" "$mode" "cpu" || true
        done
    done
done

echo ""
log "INFO" "Starting local model inference runs (CUDA)..."
echo ""

# Run local models (require CUDA)
for model in "${LOCAL_MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for mode in "${MODES[@]}"; do
            run_inference "$model" "$task" "$mode" "cuda" || true
        done
    done
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Inference Batch Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
log "INFO" "============================================"
log "INFO" "Inference Batch Complete"
log "INFO" "============================================"
log "INFO" "Total runs: ${TOTAL_RUNS}"
log "INFO" "Successful: ${SUCCESS_COUNT}"
log "INFO" "Failed: ${FAILURE_COUNT}"
log "INFO" "Success rate: $(awk "BEGIN {printf \"%.1f\", (${SUCCESS_COUNT}/${TOTAL_RUNS})*100}")%"
log "INFO" "============================================"
log "INFO" "Outputs saved to: ${OUTPUT_DIR}"
log "INFO" "Full log: ${LOG_FILE}"
log "INFO" "============================================"

# Print summary table
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "  Total runs:    ${TOTAL_RUNS}"
echo "  Successful:    ${GREEN}${SUCCESS_COUNT}${NC}"
echo "  Failed:        ${RED}${FAILURE_COUNT}${NC}"
echo "  Success rate:  $(awk "BEGIN {printf \"%.1f\", (${SUCCESS_COUNT}/${TOTAL_RUNS})*100}")%"
echo ""
echo "  Outputs:       ${OUTPUT_DIR}"
echo "  Log file:      ${LOG_FILE}"
echo ""

if [ $FAILURE_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠ Some inference runs failed. Check the log file for details.${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All inference runs completed successfully!${NC}"
    exit 0
fi
