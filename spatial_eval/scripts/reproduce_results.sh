#!/usr/bin/env bash
# =============================================================================
# reproduce_results.sh  —  Full SpatialEval experiment pipeline
# =============================================================================
#
# Runs all inference experiments, evaluations, and generates every plot used
# in the final report. Covers six experiment blocks:
#
#   1. Core experiment    — baseline vs skills, 3 tasks × 2 modes, 100 samples
#   2. MC variants        — 4 skill configs × 3 tasks × MC_RUNS iterations
#   3. Img-only scaling   — static n3/n10/n30 example images, 3 tasks
#   4. Img-only-tool      — preload tool, image-only, n3/n10/n30, 3 tasks
#   5. Preload Q&A        — preload tool + Q&A, offset-n3/n10/n30, 3 tasks
#   6. Contamination val  — img-qa-val-v2 single run (upper-bound check)
#
# After all inference, runs evaluation (accuracy CSVs) and all plot scripts.
#
# Usage (from spatial_eval/ directory):
#   bash scripts/reproduce_results.sh               # full run
#   bash scripts/reproduce_results.sh --plots-only  # skip inference, just plots
#
# Environment variables (override defaults):
#   MODEL=gpt-5.2          # model name
#   FIRST_K=100            # images per question type for core experiment
#   MC_FIRST_K=100         # images per MC iteration (phases 2–5)
#   MC_RUNS=3              # MC iterations per condition
#   WORKERS=8              # parallel API workers
#
# Prerequisites:
#   - OPENAI_API_KEY set in .env or environment
#   - uv sync run from project root
#   - Dataset downloaded: uv run python download_spatial_eval.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Parameters ─────────────────────────────────────────────────────────────
MODEL="${MODEL:-gpt-5.2}"
FIRST_K="${FIRST_K:-100}"
MC_FIRST_K="${MC_FIRST_K:-100}"
MC_RUNS="${MC_RUNS:-3}"
MC_SEED="${MC_SEED:-42}"
WORKERS="${WORKERS:-8}"
OUT="outputs"
EVAL_DIR="eval_summary"
VIS_DIR="eval_summary/result_vis"
PLOTS_ONLY=false

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log()     { echo -e "$(date +'%H:%M:%S') $*"; }
section() { echo -e "\n${BLUE}══════════════════════════════════════════════════════════${NC}"; \
            echo -e "${BLUE}  $*${NC}"; \
            echo -e "${BLUE}══════════════════════════════════════════════════════════${NC}\n"; }
ok()      { echo -e "${GREEN}  ✓ $*${NC}"; }

# ── CLI flags ───────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --plots-only) PLOTS_ONLY=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Preflight checks ────────────────────────────────────────────────────────
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY is not set. Add it to .env or export it.${NC}"
    exit 1
fi

if [[ ! -d "vqa" ]]; then
    echo -e "${RED}ERROR: Dataset not found. Run: uv run python download_spatial_eval.py${NC}"
    exit 1
fi

mkdir -p "$VIS_DIR"

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SpatialEval — Full Experiment Pipeline${NC}"
echo -e "${CYAN}  Model: ${MODEL}  |  first_k: ${FIRST_K}  |  mc_runs: ${MC_RUNS}${NC}"
echo -e "${CYAN}  mc_first_k: ${MC_FIRST_K}  |  workers: ${WORKERS}${NC}"
if $PLOTS_ONLY; then
echo -e "${CYAN}  Mode: PLOTS ONLY (inference skipped)${NC}"
fi
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo ""

TASKS=("mazenav" "spatialgrid" "spatialmap")

# Helper: run inference + eval for one condition
infer_eval() {
    local task="$1"; local mode="$2"; shift 2
    uv run python inference_vlm.py \
        --model_path "$MODEL" --task "$task" --mode "$mode" \
        --workers "$WORKERS" --output_folder "$OUT" "$@"
    uv run python evals/evaluation.py \
        --mode "$mode" --task "$task" \
        --output_folder "$OUT" --eval_summary_dir "$EVAL_DIR"
}

# =============================================================================
# PHASE 1 — Core: baseline vs skills, 3 tasks × 2 modes × FIRST_K samples
# =============================================================================
# Feeds: plot_skills_comparison.py
# Outputs: outputs/MilaWang__SpatialEval/{vqa,vtqa}/{task}/m-*.jsonl
#          eval_summary/{vqa,vtqa}/{task}_acc.csv
# =============================================================================
section "Phase 1 — Core: baseline vs skills (3 tasks × 2 modes × ${FIRST_K} samples)"

if ! $PLOTS_ONLY; then
    for task in "${TASKS[@]}"; do
        for mode in "vqa" "vtqa"; do
            log "${YELLOW}[baseline]  task=${task}  mode=${mode}${NC}"
            infer_eval "$task" "$mode" --first_k "$FIRST_K" --max_new_tokens 1024

            log "${YELLOW}[skills]    task=${task}  mode=${mode}${NC}"
            infer_eval "$task" "$mode" --first_k "$FIRST_K" --max_new_tokens 1024 \
                --use_skills
        done
    done
    ok "Phase 1 inference + eval complete"
fi

# =============================================================================
# PHASE 2 — MC variants: 4 skill configs × 3 tasks × MC_RUNS iterations
# =============================================================================
# Variants: baseline, img-only, img-qa, img-context
# Feeds: plot_mc_results.py, plot_image_skill_variants.py
# Note: example images used indices 497-499 (safe separation from test pool)
# =============================================================================
section "Phase 2 — MC variants (4 configs × 3 tasks × ${MC_RUNS} iterations)"

if ! $PLOTS_ONLY; then
    MC_CONFIGS=("" "img-only" "img-qa" "img-context")
    for task in "${TASKS[@]}"; do
        for config in "${MC_CONFIGS[@]}"; do
            label="${config:-baseline}"
            log "${YELLOW}[MC ${label}]  task=${task}${NC}"
            if [[ -z "$config" ]]; then
                uv run python inference_vlm.py \
                    --model_path "$MODEL" --task "$task" --mode vqa \
                    --first_k "$MC_FIRST_K" --mc_runs "$MC_RUNS" --mc_seed "$MC_SEED" \
                    --workers "$WORKERS" --output_folder "$OUT"
            else
                uv run python inference_vlm.py \
                    --model_path "$MODEL" --task "$task" --mode vqa \
                    --first_k "$MC_FIRST_K" --mc_runs "$MC_RUNS" --mc_seed "$MC_SEED" \
                    --workers "$WORKERS" --output_folder "$OUT" \
                    --use_skills --skills_variant "$config"
            fi
            uv run python evals/evaluation.py \
                --mode vqa --task "$task" \
                --output_folder "$OUT" --eval_summary_dir "$EVAL_DIR"
        done
    done
    ok "Phase 2 inference + eval complete"
fi

# =============================================================================
# PHASE 3 — Img-only scaling: n3 / n10 / n30 static example images
# =============================================================================
# Tests whether more static example images in the skill file improve accuracy.
# Feeds: plot_img_only_range.py
# Note: n3 uses indices 497-499; n10/n30 use highest dataset indices (no overlap
#       with first_k=100 test pool)
# =============================================================================
section "Phase 3 — Img-only scaling (n3 / n10 / n30 × 3 tasks)"

if ! $PLOTS_ONLY; then
    IMG_ONLY_VARIANTS=("img-only-n3" "img-only-n10" "img-only-n30")
    for task in "${TASKS[@]}"; do
        for variant in "${IMG_ONLY_VARIANTS[@]}"; do
            log "${YELLOW}[${variant}]  task=${task}${NC}"
            infer_eval "$task" "vqa" \
                --first_k "$MC_FIRST_K" --mc_runs "$MC_RUNS" --mc_seed "$MC_SEED" \
                --workers "$WORKERS" \
                --use_skills --skills_variant "$variant"
        done
    done
    ok "Phase 3 inference + eval complete"
fi

# =============================================================================
# PHASE 4 — Img-only-tool: preload tool, image-only, n3 / n10 / n30
# =============================================================================
# Agent receives N example images (no Q&A) via read_example(n) tool at runtime.
# offset_k is auto-computed as n_examples × mc_runs (prevents test/example overlap).
# Feeds: plot_img_only_tool.py
# =============================================================================
section "Phase 4 — Img-only-tool preload (n3 / n10 / n30 × 3 tasks)"

if ! $PLOTS_ONLY; then
    TOOL_VARIANTS=("img-only-tool-n3" "img-only-tool-n10" "img-only-tool-n30")
    for task in "${TASKS[@]}"; do
        for variant in "${TOOL_VARIANTS[@]}"; do
            log "${YELLOW}[${variant}]  task=${task}${NC}"
            # offset_k is auto-computed by inference_vlm.py (n_examples × mc_runs)
            uv run python inference_vlm.py \
                --model_path "$MODEL" --task "$task" --mode vqa \
                --first_k "$MC_FIRST_K" --mc_runs "$MC_RUNS" --mc_seed "$MC_SEED" \
                --workers "$WORKERS" --output_folder "$OUT" \
                --use_skills --skills_variant "$variant"
            uv run python evals/evaluation.py \
                --mode vqa --task "$task" \
                --output_folder "$OUT" --eval_summary_dir "$EVAL_DIR"
        done
    done
    ok "Phase 4 inference + eval complete"
fi

# =============================================================================
# PHASE 5 — Preload Q&A scaling: offset-n3 / offset (n10) / offset-n30
# =============================================================================
# Agent receives N labeled examples (image + worked Q&A) via read_example tool.
# Test set is offset by N×3 samples to guarantee no overlap with examples.
# Feeds: plot_preload_scaling.py
# =============================================================================
section "Phase 5 — Preload Q&A scaling (offset-n3 / n10 / n30 × 3 tasks)"

if ! $PLOTS_ONLY; then
    PRELOAD_VARIANTS=("img-qa-val-v2-offset-n3" "img-qa-val-v2-offset" "img-qa-val-v2-offset-n30")
    for task in "${TASKS[@]}"; do
        for variant in "${PRELOAD_VARIANTS[@]}"; do
            log "${YELLOW}[${variant}]  task=${task}${NC}"
            uv run python inference_vlm.py \
                --model_path "$MODEL" --task "$task" --mode vqa \
                --first_k "$MC_FIRST_K" --mc_runs "$MC_RUNS" --mc_seed "$MC_SEED" \
                --workers "$WORKERS" --output_folder "$OUT" \
                --use_skills --skills_variant "$variant"
            uv run python evals/evaluation.py \
                --mode vqa --task "$task" \
                --output_folder "$OUT" --eval_summary_dir "$EVAL_DIR"
        done
    done
    ok "Phase 5 inference + eval complete"
fi

# =============================================================================
# PHASE 6 — Contamination validation: img-qa-val-v2 single run
# =============================================================================
# Agent uses read_example tool on the SAME images as the test set.
# Expected ~100% accuracy — confirms skill improvement is due to reasoning,
# not dataset memorisation. Single run only (no MC).
# Feeds: plot_validation_test.py
# =============================================================================
section "Phase 6 — Contamination validation (img-qa-val-v2, 30 samples)"

if ! $PLOTS_ONLY; then
    for task in "${TASKS[@]}"; do
        log "${YELLOW}[img-qa-val-v2]  task=${task}${NC}"
        infer_eval "$task" "vqa" \
            --first_k 30 \
            --workers "$WORKERS" \
            --use_skills --skills_variant img-qa-val-v2
    done
    ok "Phase 6 inference + eval complete"
fi

# =============================================================================
# PLOTTING — Generate all result visualisations
# =============================================================================
section "Plotting — generating all figures to ${VIS_DIR}/"

log "plot_skills_comparison       (phase 1 — baseline vs skills)"
for task in "${TASKS[@]}"; do
    uv run python eval_summary/plot_skills_comparison.py \
        --eval_summary_dir "$EVAL_DIR" --task "$task" \
        --out_dir "$VIS_DIR" --first_k "$FIRST_K"
done
ok "plot_skills_comparison done"

log "plot_mc_results              (phase 2 — MC mean ± SD, 4 variants)"
uv run python eval_summary/plot_mc_results.py \
    --eval_summary_dir "$EVAL_DIR" --out_dir "$VIS_DIR"
ok "plot_mc_results done"

log "plot_image_skill_variants    (phase 2 — grouped bars per task)"
for task in "${TASKS[@]}"; do
    uv run python eval_summary/plot_image_skill_variants.py \
        --eval_summary_dir "$EVAL_DIR" --task "$task" --out_dir "$VIS_DIR"
done
ok "plot_image_skill_variants done"

log "plot_img_only_range          (phase 3 — static scaling n3/n10/n30)"
uv run python eval_summary/plot_img_only_range.py --out_dir "$VIS_DIR"
ok "plot_img_only_range done"

log "plot_img_only_tool           (phase 4 — tool preload scaling)"
uv run python eval_summary/plot_img_only_tool.py --out_dir "$VIS_DIR"
ok "plot_img_only_tool done"

log "plot_preload_scaling         (phase 5 — Q&A preload scaling)"
uv run python eval_summary/plot_preload_scaling.py --out_dir "$VIS_DIR"
ok "plot_preload_scaling done"

log "plot_validation_test         (phase 6 — contamination validation)"
uv run python eval_summary/plot_validation_test.py --out_dir "$VIS_DIR"
ok "plot_validation_test done"

# =============================================================================
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All done! Figures saved to: spatial_eval/${VIS_DIR}/${NC}"
echo -e "${GREEN}──────────────────────────────────────────────────────────${NC}"
echo -e "${GREEN}  Core comparison:     {task}_skills_comparison.png${NC}"
echo -e "${GREEN}  MC variants:         mc_results_*.png, image_skill_variants_*.png${NC}"
echo -e "${GREEN}  Img-only scaling:    img_only_range_*.png${NC}"
echo -e "${GREEN}  Tool preload:        img_only_tool_*.png${NC}"
echo -e "${GREEN}  Q&A preload:         preload_scaling_*.png${NC}"
echo -e "${GREEN}  Contamination:       validation_test_*.png${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
