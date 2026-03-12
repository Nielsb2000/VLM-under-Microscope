#!/bin/bash

# =============================================================================
# Multi-Model Spatial Evaluation - Evaluation Script
# =============================================================================
# This script evaluates all model outputs and generates accuracy summaries
# Runs evaluation for all 4 tasks across 2 modes (vqa/vtqa)
# Total: 8 evaluation runs (4 tasks × 2 modes)
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
EVAL_SUMMARY_DIR="${PROJECT_DIR}/eval_summary"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/evaluation_${TIMESTAMP}.log"

# Create directories
mkdir -p "$OUTPUT_DIR" "$EVAL_SUMMARY_DIR" "$LOG_DIR"

# Define tasks and modes
TASKS=("mazenav" "spatialmap" "spatialgrid" "spatialreal")
MODES=("vqa" "vtqa")

# Counters
TOTAL_EVALS=$((${#TASKS[@]} * ${#MODES[@]}))
CURRENT_EVAL=0
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
    local task="$1"
    local mode="$2"
    CURRENT_EVAL=$((CURRENT_EVAL + 1))
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Evaluation Progress: ${CURRENT_EVAL}/${TOTAL_EVALS} (Success: ${SUCCESS_COUNT}, Failed: ${FAILURE_COUNT})${NC}"
    echo -e "${BLUE}Task: ${YELLOW}${task}${NC}"
    echo -e "${BLUE}Mode: ${YELLOW}${mode}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Function to run evaluation
run_evaluation() {
    local task="$1"
    local mode="$2"
    
    show_progress "$task" "$mode"
    
    log "INFO" "Running evaluation: task=${task}, mode=${mode}"
    
    # Check if output directory exists
    local task_output_dir="${OUTPUT_DIR}/MilaWang__SpatialEval/${mode}/${task}"
    if [ ! -d "$task_output_dir" ]; then
        log "WARNING" "No outputs found for task=${task}, mode=${mode}. Skipping."
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        return 1
    fi
    
    # Count output files
    local output_count=$(find "$task_output_dir" -name "m-*_bare.jsonl" 2>/dev/null | wc -l)
    log "INFO" "Found ${output_count} output files in ${task_output_dir}"
    
    if [ $output_count -eq 0 ]; then
        log "WARNING" "No output files found for task=${task}, mode=${mode}. Skipping."
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        return 1
    fi
    
    # Build command
    local cmd="cd ${PROJECT_DIR} && uv run python evals/evaluation.py \
        --task \"${task}\" \
        --mode \"${mode}\" \
        --output_folder \"${OUTPUT_DIR}\" \
        --eval_summary_dir \"${EVAL_SUMMARY_DIR}\""
    
    # Execute with timeout (10 minutes per evaluation)
    if timeout 600 bash -c "$cmd" >> "$LOG_FILE" 2>&1; then
        log "SUCCESS" "✓ Evaluation completed: ${task} | ${mode}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # Check if summary files were created
        local summary_file="${EVAL_SUMMARY_DIR}/${mode}/${task}_acc.csv"
        if [ -f "$summary_file" ]; then
            log "SUCCESS" "✓ Summary file created: ${summary_file}"
        else
            log "WARNING" "Summary file not found: ${summary_file}"
        fi
        
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log "ERROR" "✗ Timeout (10min): ${task} | ${mode}"
        else
            log "ERROR" "✗ Failed (exit code ${exit_code}): ${task} | ${mode}"
        fi
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        return 1
    fi
}

# Function to generate summary report
generate_summary_report() {
    local report_file="${EVAL_SUMMARY_DIR}/evaluation_summary_${TIMESTAMP}.txt"
    
    log "INFO" "Generating summary report: ${report_file}"
    
    {
        echo "=========================================="
        echo "Multi-Model Spatial Evaluation Summary"
        echo "=========================================="
        echo "Generated: $(date)"
        echo ""
        echo "Tasks evaluated: ${TASKS[*]}"
        echo "Modes evaluated: ${MODES[*]}"
        echo ""
        echo "Total evaluations: ${TOTAL_EVALS}"
        echo "Successful: ${SUCCESS_COUNT}"
        echo "Failed: ${FAILURE_COUNT}"
        echo ""
        echo "=========================================="
        echo "Accuracy Summaries by Task and Mode"
        echo "=========================================="
        echo ""
        
        for mode in "${MODES[@]}"; do
            echo "----------------------------------------"
            echo "Mode: ${mode}"
            echo "----------------------------------------"
            for task in "${TASKS[@]}"; do
                local summary_file="${EVAL_SUMMARY_DIR}/${mode}/${task}_acc.csv"
                if [ -f "$summary_file" ]; then
                    echo ""
                    echo "Task: ${task}"
                    echo "File: ${summary_file}"
                    echo ""
                    cat "$summary_file"
                    echo ""
                else
                    echo ""
                    echo "Task: ${task}"
                    echo "Status: No summary file found"
                    echo ""
                fi
            done
        done
        
        echo "=========================================="
        echo "Detailed Evaluation Outputs"
        echo "=========================================="
        echo ""
        
        for mode in "${MODES[@]}"; do
            for task in "${TASKS[@]}"; do
                local summary_dir="${EVAL_SUMMARY_DIR}/${mode}"
                if [ -d "$summary_dir" ]; then
                    echo "Mode: ${mode}, Task: ${task}"
                    find "$summary_dir" -name "${task}*.jsonl" -exec echo "  - {}" \;
                    echo ""
                fi
            done
        done
        
    } > "$report_file"
    
    log "SUCCESS" "Summary report saved: ${report_file}"
    echo ""
    echo -e "${GREEN}Summary report saved to: ${report_file}${NC}"
}

# =============================================================================
# Main Execution
# =============================================================================

log "INFO" "============================================"
log "INFO" "Starting Multi-Model Evaluation"
log "INFO" "============================================"
log "INFO" "Timestamp: ${TIMESTAMP}"
log "INFO" "Log file: ${LOG_FILE}"
log "INFO" "Output directory: ${OUTPUT_DIR}"
log "INFO" "Evaluation summary directory: ${EVAL_SUMMARY_DIR}"
log "INFO" "Total evaluations: ${TOTAL_EVALS}"
log "INFO" "Tasks: ${TASKS[*]}"
log "INFO" "Modes: ${MODES[*]}"
log "INFO" "============================================"

echo ""
log "INFO" "Starting evaluations..."
echo ""

# Run evaluations for all task/mode combinations
for mode in "${MODES[@]}"; do
    for task in "${TASKS[@]}"; do
        run_evaluation "$task" "$mode" || true
    done
done

# Generate summary report
echo ""
log "INFO" "Generating summary report..."
generate_summary_report

# =============================================================================
# Summary
# =============================================================================

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Evaluation Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
log "INFO" "============================================"
log "INFO" "Evaluation Complete"
log "INFO" "============================================"
log "INFO" "Total evaluations: ${TOTAL_EVALS}"
log "INFO" "Successful: ${SUCCESS_COUNT}"
log "INFO" "Failed: ${FAILURE_COUNT}"
log "INFO" "Success rate: $(awk "BEGIN {printf \"%.1f\", (${SUCCESS_COUNT}/${TOTAL_EVALS})*100}")%"
log "INFO" "============================================"
log "INFO" "Summaries saved to: ${EVAL_SUMMARY_DIR}"
log "INFO" "Full log: ${LOG_FILE}"
log "INFO" "============================================"

# Print summary table
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "  Total evaluations: ${TOTAL_EVALS}"
echo "  Successful:        ${GREEN}${SUCCESS_COUNT}${NC}"
echo "  Failed:            ${RED}${FAILURE_COUNT}${NC}"
echo "  Success rate:      $(awk "BEGIN {printf \"%.1f\", (${SUCCESS_COUNT}/${TOTAL_EVALS})*100}")%"
echo ""
echo "  Summaries:         ${EVAL_SUMMARY_DIR}"
echo "  Log file:          ${LOG_FILE}"
echo ""

if [ $FAILURE_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠ Some evaluations failed. Check the log file for details.${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All evaluations completed successfully!${NC}"
    exit 0
fi
