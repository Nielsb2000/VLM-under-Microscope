# spatial_eval: Technical Review & Implementation Guide

---

## Overview

**spatial_eval** is a modular Python suite for evaluating spatial reasoning in vision-language models. It supports flexible configuration, batch evaluation, and detailed result analysis for spatial VQA and VTQA tasks. The design emphasizes reproducibility, extensibility, and CLI-driven workflows, with organized outputs for easy comparison and downstream analysis.

---

## Directory Structure

```
spatial_eval/
├── download_spatial_eval.py      # Downloads datasets/resources
├── inference_vlm.py              # Main inference pipeline (CLI)
├── README.md                     # Module usage and details
├── TESTING_GUIDE.md              # Testing instructions
├── configs/
│   └── inference_configs.py      # Model/config presets
├── eval_summary/
│   ├── evaluation_summary_*.txt  # Evaluation summaries
│   ├── plot_results.py           # Plots evaluation results
│   └── result_vis/               # Plots/visualizations
├── evals/
│   └── evaluation.py             # Evaluation logic
├── logs/                         # Run logs
├── models/
│   ├── bunny_model.py
│   ├── gpt4_model.py
│   ├── llava_model.py
│   └── model_utils.py
├── outputs/                      # Model outputs (by run)
├── scripts/
│   ├── evaluate_all_results.sh   # Batch evaluation
│   ├── inf_vlm.sh                # Batch inference
│   ├── run_all_evaluations.sh    # Pipeline runner
│   └── run_full_pipeline.sh      # Full pipeline script
├── utils/
│   ├── format_filename.py
│   └── load_image.py
├── vqa/                          # VQA datasets/tasks
│   ├── maze-nav/
│   ├── spatial-grid/
│   ├── spatial-map/
│   └── spatial-real/
├── vtqa/                         # VTQA datasets/tasks
│   ├── maze-nav/
│   ├── spatial-grid/
│   ├── spatial-map/
│   └── spatial-real/
```

---

## Key Modules & Scripts

### 1. inference_vlm.py
- **Purpose:** Main inference pipeline for running VLMs on spatial tasks.
- **Features:**
  - CLI: Select model, config, dataset, output path.
  - Loads configs, datasets, and runs model inference.
  - Saves outputs to organized folders.
- **Key Functions:**
  - `main()` (CLI entry)
  - `run_inference(model, dataset, config)`
- **Dependencies:** pandas, custom model scripts

### 2. evals/evaluation.py
- **Purpose:** Core evaluation logic for VQA/VTQA tasks.
- **Features:**
  - Computes accuracy, metrics, and aggregates results.
  - Used by pipeline scripts and summary tools.
- **Key Functions:**
  - `evaluate(predictions, ground_truth)`
- **Dependencies:** pandas, numpy

### 3. eval_summary/plot_results.py
- **Purpose:** Plots evaluation results (accuracy, confusion, etc.).
- **Features:**
  - CLI: Select summary file, output dir.
  - Generates and saves plots to `result_vis/`.
- **Key Functions:**
  - `plot_summary(summary_path, output_dir)`
- **Dependencies:** matplotlib, pandas

### 4. configs/inference_configs.py
- **Purpose:** Stores model and inference configuration presets.
- **Features:**
  - Easily add new models/configs.
  - Used by inference pipeline.
- **Key Functions:**
  - `get_config(model_name)`
- **Dependencies:** None (pure Python)

### 5. models/
- **Purpose:** Model wrappers and utilities for supported VLMs.
- **Files:**
  - `bunny_model.py`, `gpt4_model.py`, `llava_model.py`, `model_utils.py`
- **Features:**
  - Unified interface for inference.
  - Easy to extend with new models.

### 6. scripts/
- **Purpose:** Shell scripts for batch and pipeline automation.
- **Files:**
  - `evaluate_all_results.sh`, `inf_vlm.sh`, `run_all_evaluations.sh`, `run_full_pipeline.sh`
- **Features:**
  - Automate multi-model, multi-dataset runs.

---

## Data Flow Diagram (ASCII)

```
[Datasets (vqa/vtqa)]
      |
      v
[configs/inference_configs.py]
      |
      v
[inference_vlm.py] <--- [models/]
      |
      v
[outputs/]
      |
      v
[evals/evaluation.py]
      |
      v
[eval_summary/evaluation_summary_*.txt]
      |
      v
[eval_summary/plot_results.py]
      |
      v
[eval_summary/result_vis/]
```

---

## Implementation Notes

- **Modularity:** Each component is CLI-driven and can be used independently or in pipelines.
- **Extensibility:** Add new models, datasets, or configs with minimal changes.
- **Reproducibility:** Results and outputs are organized by run/model/config for easy comparison.
- **Automation:** Shell scripts enable batch and full-pipeline runs.
- **Dependencies:** Managed via `pyproject.toml` and `.venv` at project root.

---

## Usage Examples

### 1. Run Inference
```bash
python spatial_eval/inference_vlm.py --model gpt4 --config configs/inference_configs.py --dataset vqa/spatial-grid/ --output outputs/gpt4_spatial_grid/
```

### 2. Evaluate Results
```bash
python spatial_eval/evals/evaluation.py --predictions outputs/gpt4_spatial_grid/preds.json --ground-truth vqa/spatial-grid/ground_truth.json
```

### 3. Plot Evaluation Summary
```bash
python spatial_eval/eval_summary/plot_results.py --summary eval_summary/evaluation_summary_20260210_154546.txt --output-dir eval_summary/result_vis/
```

### 4. Run Full Pipeline
```bash
bash spatial_eval/scripts/run_full_pipeline.sh
```

---

## See Also
- [CODEBASE_OVERVIEW.md](../CODEBASE_OVERVIEW.md): Project-wide symbol and structure guide.
- [CODEBASE_TECHNICAL_REVIEW.md](../CODEBASE_TECHNICAL_REVIEW.md): In-depth technical review for spatial_eval.

---

