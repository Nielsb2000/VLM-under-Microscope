#!/bin/bash

# =============================================================================
# Multi-Model Spatial Evaluation - Full Pipeline
# =============================================================================
# This master script orchestrates the complete evaluation pipeline:
# 1. Run inference for all models (72 runs: 9 models × 4 tasks × 2 modes)
# 2. Evaluate all results (8 evaluations: 4 tasks × 2 modes)
# 3. Generate comprehensive summary report
# =============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    local level="$1"
    shift
    local message="$@"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$MASTER_LOG"
}

# Function to display banner
show_banner() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                                    ║${NC}"
    echo -e "${CYAN}║         ${MAGENTA}Multi-Model Spatial Evaluation Pipeline${CYAN}                 ║${NC}"
    echo -e "${CYAN}║                                                                    ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Models to evaluate:${NC}"
    echo "  • API Models: gpt-4o, gpt-5.1"
    echo "  • LLaVA Models: llava-v1.5-7b, llava-v1.6-mistral-7b, llava-v1.6-vicuna-7b"
    echo "  • Bunny Models: bunny-phi-2, bunny-phi-1.5, bunny-stablelm-2, bunny-v1_0-3b"
    echo ""
    echo -e "${YELLOW}Tasks:${NC} mazenav, spatialmap, spatialgrid, spatialreal"
    echo -e "${YELLOW}Modes:${NC} vqa, vtqa"
    echo -e "${YELLOW}Samples per task:${NC} 5 (--first_k 5)"
    echo ""
    echo -e "${YELLOW}Total operations:${NC}"
    echo "  • Inference runs: 72 (9 models × 4 tasks × 2 modes)"
    echo "  • Evaluations: 8 (4 tasks × 2 modes)"
    echo ""
    echo -e "${YELLOW}Master log:${NC} ${MASTER_LOG}"
    echo ""
}

# Function to check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    local errors=0
    
    # Check Python
    if ! command -v uv &> /dev/null; then
        log "ERROR" "uv not found in PATH"
        errors=$((errors + 1))
    else
        log "SUCCESS" "uv found: $(uv --version 2>&1)"
    fi
    
    # Check required Python packages
    local required_packages=("torch" "transformers" "langchain_openai" "PIL")
    for package in "${required_packages[@]}"; do
        if cd "$PROJECT_DIR" && uv run python -c "import ${package}" 2>/dev/null; then
            log "SUCCESS" "Python package '${package}' is available"
        else
            log "WARNING" "Python package '${package}' not found - some models may fail"
        fi
    done
    
    # Check CUDA
    if cd "$PROJECT_DIR" && uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local cuda_version=$(cd "$PROJECT_DIR" && uv run python -c "import torch; print(torch.version.cuda)")
        log "SUCCESS" "CUDA is available (version: ${cuda_version})"
    else
        log "WARNING" "CUDA not available - local models will fail"
        errors=$((errors + 1))
    fi
    
    # Check OpenAI API key
    if [ -z "$OPENAI_API_KEY" ]; then
        log "WARNING" "OPENAI_API_KEY not set - API models (gpt-4o, gpt-5.1) will fail"
    else
        log "SUCCESS" "OPENAI_API_KEY is set"
    fi
    
    # Check disk space (need ~40GB for models + outputs)
    local available_space=$(df -BG "$PROJECT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 50 ]; then
        log "WARNING" "Low disk space: ${available_space}GB available (recommend 50GB+)"
    else
        log "SUCCESS" "Sufficient disk space: ${available_space}GB available"
    fi
    
    # Check if inference script exists
    if [ ! -f "${SCRIPT_DIR}/run_all_evaluations.sh" ]; then
        log "ERROR" "Inference script not found: ${SCRIPT_DIR}/run_all_evaluations.sh"
        errors=$((errors + 1))
    fi
    
    # Check if evaluation script exists
    if [ ! -f "${SCRIPT_DIR}/evaluate_all_results.sh" ]; then
        log "ERROR" "Evaluation script not found: ${SCRIPT_DIR}/evaluate_all_results.sh"
        errors=$((errors + 1))
    fi
    
    if [ $errors -gt 0 ]; then
        log "ERROR" "Prerequisites check failed with ${errors} error(s)"
        return 1
    else
        log "SUCCESS" "All critical prerequisites satisfied"
        return 0
    fi
}

# Function to run with timing
run_timed() {
    local description="$1"
    local command="$2"
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Starting: ${YELLOW}${description}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    log "INFO" "Starting: ${description}"
    local start_time=$(date +%s)
    
    if eval "$command"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}✓ Completed: ${description}${NC}"
        echo -e "${GREEN}Duration: ${hours}h ${minutes}m ${seconds}s${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        
        log "SUCCESS" "Completed: ${description} (${hours}h ${minutes}m ${seconds}s)"
        return 0
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        echo ""
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}✗ Failed: ${description}${NC}"
        echo -e "${RED}Exit code: ${exit_code}${NC}"
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        
        log "ERROR" "Failed: ${description} (exit code: ${exit_code}, duration: ${duration}s)"
        return $exit_code
    fi
}

# Function to generate final report
generate_final_report() {
    local report_file="${PROJECT_DIR}/EVALUATION_REPORT_${TIMESTAMP}.md"
    
    log "INFO" "Generating final report: ${report_file}"
    
    {
        echo "# Multi-Model Spatial Evaluation Report"
        echo ""
        echo "**Generated:** $(date)"
        echo "**Pipeline Start:** $(head -n 1 "$MASTER_LOG" | cut -d' ' -f1-2)"
        echo "**Pipeline End:** $(date +"%Y-%m-%d %H:%M:%S")"
        echo ""
        echo "---"
        echo ""
        echo "## Overview"
        echo ""
        echo "This report summarizes the results of running spatial reasoning evaluations across multiple vision-language models."
        echo ""
        echo "### Models Evaluated"
        echo ""
        echo "**API Models (2):**"
        echo "- gpt-4o"
        echo "- gpt-5.1"
        echo ""
        echo "**LLaVA Models (3):**"
        echo "- liuhaotian/llava-v1.5-7b"
        echo "- liuhaotian/llava-v1.6-mistral-7b"
        echo "- liuhaotian/llava-v1.6-vicuna-7b"
        echo ""
        echo "**Bunny Models (4):**"
        echo "- BAAI/bunny-phi-2"
        echo "- BAAI/bunny-phi-1.5"
        echo "- BAAI/bunny-stablelm-2"
        echo "- BAAI/bunny-v1_0-3b"
        echo ""
        echo "### Tasks"
        echo ""
        echo "- **mazenav**: Maze navigation spatial reasoning"
        echo "- **spatialmap**: Map-based spatial relationships"
        echo "- **spatialgrid**: Grid-based spatial reasoning"
        echo "- **spatialreal**: Real-world spatial understanding"
        echo ""
        echo "### Modes"
        echo ""
        echo "- **vqa**: Vision Question Answering (image + question)"
        echo "- **vtqa**: Vision-Text Question Answering (image + text representation + question)"
        echo ""
        echo "### Configuration"
        echo ""
        echo "- Samples per task: 5 (\`--first_k 5\`)"
        echo "- Temperature: 0.2"
        echo "- Max new tokens: 512"
        echo ""
        echo "---"
        echo ""
        echo "## Pipeline Execution"
        echo ""
        echo "### Phase 1: Inference"
        echo ""
        echo "Total inference runs: **72** (9 models × 4 tasks × 2 modes)"
        echo ""
        
        if [ -f "${LOG_DIR}/inference_"*".log" ]; then
            local inference_log=$(ls -t "${LOG_DIR}/inference_"*".log" 2>/dev/null | head -n 1)
            if [ -n "$inference_log" ]; then
                local success_count=$(grep -c "SUCCESS.*Completed:" "$inference_log" 2>/dev/null || echo "0")
                local failure_count=$(grep -c "ERROR.*Failed:" "$inference_log" 2>/dev/null || echo "0")
                echo "- Successful: ${success_count}"
                echo "- Failed: ${failure_count}"
                echo "- Success rate: $(awk "BEGIN {printf \"%.1f%%\", (${success_count}/72)*100}")"
            fi
        fi
        
        echo ""
        echo "### Phase 2: Evaluation"
        echo ""
        echo "Total evaluations: **8** (4 tasks × 2 modes)"
        echo ""
        
        if [ -f "${LOG_DIR}/evaluation_"*".log" ]; then
            local eval_log=$(ls -t "${LOG_DIR}/evaluation_"*".log" 2>/dev/null | head -n 1)
            if [ -n "$eval_log" ]; then
                local success_count=$(grep -c "SUCCESS.*Evaluation completed:" "$eval_log" 2>/dev/null || echo "0")
                local failure_count=$(grep -c "ERROR.*Failed:" "$eval_log" 2>/dev/null || echo "0")
                echo "- Successful: ${success_count}"
                echo "- Failed: ${failure_count}"
                echo "- Success rate: $(awk "BEGIN {printf \"%.1f%%\", (${success_count}/8)*100}")"
            fi
        fi
        
        echo ""
        echo "---"
        echo ""
        echo "## Results Location"
        echo ""
        echo "### Inference Outputs"
        echo ""
        echo "\`\`\`"
        echo "${PROJECT_DIR}/outputs/MilaWang__SpatialEval/"
        echo "├── vqa/"
        echo "│   ├── mazenav/"
        echo "│   ├── spatialmap/"
        echo "│   ├── spatialgrid/"
        echo "│   └── spatialreal/"
        echo "└── vtqa/"
        echo "    ├── mazenav/"
        echo "    ├── spatialmap/"
        echo "    ├── spatialgrid/"
        echo "    └── spatialreal/"
        echo "\`\`\`"
        echo ""
        echo "### Evaluation Summaries"
        echo ""
        echo "\`\`\`"
        echo "${PROJECT_DIR}/eval_summary/"
        echo "├── vqa/"
        echo "│   ├── mazenav_acc.csv"
        echo "│   ├── spatialmap_acc.csv"
        echo "│   ├── spatialgrid_acc.csv"
        echo "│   └── spatialreal_acc.csv"
        echo "└── vtqa/"
        echo "    ├── mazenav_acc.csv"
        echo "    ├── spatialmap_acc.csv"
        echo "    ├── spatialgrid_acc.csv"
        echo "    └── spatialreal_acc.csv"
        echo "\`\`\`"
        echo ""
        echo "### Logs"
        echo ""
        echo "- Master log: \`${MASTER_LOG}\`"
        echo "- All logs: \`${LOG_DIR}/\`"
        echo ""
        echo "---"
        echo ""
        echo "## Next Steps"
        echo ""
        echo "1. Review accuracy summaries in \`eval_summary/\` directories"
        echo "2. Compare model performance across tasks and modes"
        echo "3. Analyze detailed outputs in \`outputs/\` directories"
        echo "4. Check logs for any errors or warnings"
        echo ""
        echo "---"
        echo ""
        echo "**Report generated by:** \`run_full_pipeline.sh\`"
        echo ""
        
    } > "$report_file"
    
    log "SUCCESS" "Final report saved: ${report_file}"
    echo -e "${GREEN}Final report saved to: ${report_file}${NC}"
}

# =============================================================================
# Main Execution
# =============================================================================

log "INFO" "=========================================="
log "INFO" "Multi-Model Spatial Evaluation Pipeline"
log "INFO" "=========================================="
log "INFO" "Timestamp: ${TIMESTAMP}"
log "INFO" "Master log: ${MASTER_LOG}"
log "INFO" "Project directory: ${PROJECT_DIR}"
log "INFO" "=========================================="

show_banner

# Ask for confirmation
echo -e "${YELLOW}This pipeline will:${NC}"
echo "  1. Run 72 inference operations (may take several hours)"
echo "  2. Evaluate 8 task/mode combinations"
echo "  3. Generate comprehensive reports"
echo ""
echo -e "${YELLOW}Note: API models require OPENAI_API_KEY to be set${NC}"
echo -e "${YELLOW}Note: Local models require CUDA GPU with 16GB+ VRAM${NC}"
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "INFO" "Pipeline cancelled by user"
    echo "Pipeline cancelled."
    exit 0
fi

echo ""

# Check prerequisites
if ! check_prerequisites; then
    echo ""
    echo -e "${RED}Prerequisites check failed. Please fix the issues above and try again.${NC}"
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Phase 1: Run all inference
run_timed "Inference Phase (72 runs)" "bash ${SCRIPT_DIR}/run_all_evaluations.sh" || {
    log "ERROR" "Inference phase failed"
    echo -e "${RED}Inference phase failed. Check logs for details.${NC}"
    exit 1
}

# Phase 2: Run all evaluations
run_timed "Evaluation Phase (8 evaluations)" "bash ${SCRIPT_DIR}/evaluate_all_results.sh" || {
    log "ERROR" "Evaluation phase failed"
    echo -e "${RED}Evaluation phase failed. Check logs for details.${NC}"
    exit 1
}

# Generate final report
echo ""
log "INFO" "Generating final report..."
generate_final_report

# Final summary
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                    ║${NC}"
echo -e "${CYAN}║         ${GREEN}✓ Pipeline Complete!${CYAN}                                        ║${NC}"
echo -e "${CYAN}║                                                                    ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

log "INFO" "=========================================="
log "INFO" "Pipeline Complete!"
log "INFO" "=========================================="
log "INFO" "Outputs: ${PROJECT_DIR}/outputs/"
log "INFO" "Summaries: ${PROJECT_DIR}/eval_summary/"
log "INFO" "Report: ${PROJECT_DIR}/EVALUATION_REPORT_${TIMESTAMP}.md"
log "INFO" "Master log: ${MASTER_LOG}"
log "INFO" "=========================================="

echo -e "${GREEN}✓ All phases completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review the evaluation report: EVALUATION_REPORT_${TIMESTAMP}.md"
echo "  2. Check accuracy summaries: eval_summary/"
echo "  3. Analyze detailed outputs: outputs/"
echo ""

exit 0
