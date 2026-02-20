# MS_Paint_Reasoning_Evaluation: Technical Review & Implementation Guide

---

## Overview

**MS_Paint_Reasoning_Evaluation** is a Python module for evaluating visual reasoning using MS Paint-style images. It provides tools for image blurring, evaluation, result analysis, and plotting, supporting multiple blur levels and models. The design is modular, CLI-driven, and file-based, enabling reproducible experiments and extensible workflows.

---

## Directory Structure

```
MS_Paint_Reasoning_Evaluation/
├── Blur.py                  # Image blurring utilities
├── Eval_script.py           # Main evaluation pipeline (CLI)
├── export_answers_to_excel.py # Exports answers to Excel (multi-sheet)
├── gpt4_model_MS_Paint.py   # GPT-4 model interface
├── MSP_results.py           # Plots accuracy/heatmaps per blur level
├── plot_token_time_stats.py # Plots token/time stats
├── Results/                 # Organized results (by blur/model)
├── MS_paint_images/         # Source images (blurred/original)
│   ├── heavy_blur_images/
│   ├── med_blur_images/
│   └── ...
├── MS paint answers/        # Model answers (by run)
├── MS paint questions/      # Questions (by set)
└── ...
```

---

## Key Modules & Scripts

### 1. Blur.py
- **Purpose:** Applies Gaussian blur to images at configurable levels.
- **Usage:**
  - CLI or import as a module.
  - Saves blurred images to `MS_paint_images/<blur_level>_images/`.
- **Key Functions:**
  - `blur_image(input_path, output_path, blur_level)`
- **Dependencies:** Pillow (PIL)

### 2. Eval_script.py
- **Purpose:** Main evaluation pipeline for running models on questions/images.
- **Features:**
  - CLI: Select model, blur level, input/output paths.
  - Loads questions, images, and runs model inference.
  - Saves answers and metrics to organized results folders.
- **Key Functions:**
  - `main()` (CLI entry)
  - `evaluate_model(model, questions, images, blur_level)`
- **Dependencies:** pandas, custom model scripts


### 3. MSP_results.py
- **Purpose:** Plots accuracy and heatmaps per blur level/model.
- **Features:**
  - CLI: Select blur level, model, result folder.
  - Dynamically detects and visualizes all models present in the results file (no fixed model list).
  - Generates and saves plots to `Results/res_vis/`.
- **Key Functions:**
  - `plot_accuracy(results_path, blur_level)`
- **Dependencies:** matplotlib, pandas

### 4. plot_token_time_stats.py
- **Purpose:** Plots token and time statistics for model runs.
- **Features:**
  - CLI: Select blur level, model, result folder.
  - Dynamically detects which models are present in the results directory and only plots those (robust to 2+ models).
  - Generates and saves plots to `Results/res_vis/`.
- **Key Functions:**
  - `plot_token_time_stats(results_path, blur_level)`
- **Dependencies:** matplotlib, pandas

### 5. export_answers_to_excel.py
- **Purpose:** Exports model answers to Excel, with a sheet per blur level.
- **Features:**
  - CLI: Select result folder, output Excel path.
  - Organizes answers by blur level/model.
- **Key Functions:**
  - `export_to_excel(results_dir, output_excel)`
- **Dependencies:** pandas, openpyxl

---

## Data Flow Diagram (ASCII)

```
[Images/Questions] 
      |         
      v         
  [Blur.py]  <---
      |         
      v         
[Eval_script.py] <--- [gpt4_model_MS_Paint.py]
      |         
      v         
 [Results/]    
      |         
      v         
[MSP_results.py]   [plot_token_time_stats.py]   [export_answers_to_excel.py]
      |                |                        |
      v                v                        v
 [Plots]           [Plots]                  [Excel]
```

---

## Implementation Notes


- **Modularity:** Each script is CLI-driven and can be used independently or as part of a pipeline.
- **Extensibility:** New models, blur levels, and reasoning modes can be added with minimal changes. All analysis and plotting scripts adapt to the number of models present.
- **Reproducibility:** Results are organized by blur/model/reasoning mode/run for easy comparison.
- **Reasoning Mode Support:** All scripts (evaluation, plotting, export) support reasoning modes (low/medium/high/none) and organize results accordingly.
- **Output Handling:** Output files and Excel exports are robust to both reasoning and non-reasoning models, with clear, readable formatting and per-(blur, reasoning) Excel sheets.
- **Dependencies:** All requirements are managed via `pyproject.toml` and a local `.venv`.

---

## Usage Examples

### 1. Blur Images
```bash
python MS_Paint_Reasoning_Evaluation/Blur.py --input-dir MS_paint_images/original/ --output-dir MS_paint_images/heavy_blur_images/ --blur-level heavy_blur
```

### 2. Run Evaluation
```bash
python MS_Paint_Reasoning_Evaluation/Eval_script.py --model gpt4 --blur-level heavy_blur --questions MS_paint_questions/Questions1.txt --images MS_paint_images/heavy_blur_images/ --output Results/img1/
```

### 3. Plot Results
```bash
python MS_Paint_Reasoning_Evaluation/MSP_results.py --blur-level heavy_blur --results Results/img1/
```

### 4. Export Answers to Excel
```bash
python MS_Paint_Reasoning_Evaluation/export_answers_to_excel.py --results-dir Results/ --output answers.xlsx
```

---

## See Also
- [CODEBASE_OVERVIEW.md](../CODEBASE_OVERVIEW.md): Project-wide symbol and structure guide.
- [CODEBASE_TECHNICAL_REVIEW.md](../CODEBASE_TECHNICAL_REVIEW.md): In-depth technical review for spatial_eval.

---

*This document is auto-generated for agent understanding. Please verify details with the codebase as needed.*
