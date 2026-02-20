## License & Usage

As of February 2026, the SpatialEval dataset and code are released for research and academic use only. No explicit open-source or Creative Commons license is specified in the official GitHub or Hugging Face pages. Users should cite the original paper (see above) and consult the [SpatialEval GitHub](https://github.com/jiayuww/SpatialEval) for any updates or clarifications regarding licensing before using the dataset or code in commercial or derivative works.

**Summary:**
- Intended for research/academic use
- Cite the NeurIPS 2024 paper when using
- Check the original repository for the latest license status
# spatial_eval: Technical Review & Implementation Guide

#
# ---
#
# The following section is imported from README.md for completeness:
#
# ---

# SpatialEval Dataset

This folder contains the **SpatialEval** benchmark datasets for evaluating spatial intelligence in Large Language Models (LLMs) and Vision-Language Models (VLMs).

## 📊 Dataset Overview

**Source**: [MilaWang/SpatialEval on HuggingFace](https://huggingface.co/datasets/MilaWang/SpatialEval)  
**Paper**: [Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models](https://arxiv.org/pdf/2406.14852)  

# spatial_eval: Technical Review & Implementation Guide

## Overview

**spatial_eval** is a modular Python suite and dataset for evaluating spatial reasoning in vision-language models (VLMs) and large language models (LLMs). It provides:
- The **SpatialEval** benchmark dataset (VQA and VTQA tasks)
- Flexible, CLI-driven pipelines for inference, evaluation, and result analysis
- Batch automation scripts and configuration presets
- Organized outputs for reproducibility and comparison

---

## 1. Dataset: SpatialEval

SpatialEval is a benchmark for spatial intelligence in LLMs and VLMs.

**Source:** [MilaWang/SpatialEval on HuggingFace](https://huggingface.co/datasets/MilaWang/SpatialEval)  
**Paper:** [Is A Picture Worth A Thousand Words?](https://arxiv.org/pdf/2406.14852)  
**Code:** [jiayuww/SpatialEval](https://github.com/jiayuww/SpatialEval)

**Size:** ~1.43 GB  
**Total Samples:** ~9,270 (4,635 per modality)  
**Modalities:** 2 (VQA, VTQA)  
**Tasks:** 4 (Spatial-Map, Maze-Nav, Spatial-Grid, Spatial-Real)

### Tasks
- **Spatial-Map:** Map-based spatial relationships
- **Maze-Nav:** Navigation in mazes
- **Spatial-Grid:** Reasoning in grid environments
- **Spatial-Real:** Real-world spatial understanding

### Data Format
Each `data.json` contains samples like:
```json
{
  "id": "spatialmap.0.123",
  "text": "Question text...",
  "oracle_answer": "A",
  "oracle_option": "northeast",
  "oracle_full_answer": "The answer is A. northeast because...",
  "image_path": "images/spatialmap_0_123.png"
}
```
**Fields:**
- `id`: Unique identifier
- `text`: Question prompt
- `oracle_answer`: Concise answer
- `oracle_option`: Detailed answer
- `oracle_full_answer`: Full reasoning
- `image_path`: Relative path to image

### Dataset Statistics
| Modality | Task         | Samples | Images |
|----------|--------------|---------|--------|
| VQA      | Spatial-Map  | ~1,500  | ~1,500 |
| VQA      | Maze-Nav     | ~1,500  | ~1,500 |
| VQA      | Spatial-Grid | ~1,500  | ~1,500 |
| VQA      | Spatial-Real | ~135    | ~135   |
| VTQA     | Spatial-Map  | ~1,500  | ~1,500 |
| VTQA     | Maze-Nav     | ~1,500  | ~1,500 |
| VTQA     | Spatial-Grid | ~1,500  | ~1,500 |
| VTQA     | Spatial-Real | ~135    | ~135   |
**Total:** ~9,270 samples across 8 task-modality combinations

### Loading Data Example
```python
import json
from pathlib import Path
with open('spatial_eval/vqa/spatial-map/data.json', 'r') as f:
    vqa_spatial_map = json.load(f)
```

---

## 2. Implementation & Codebase

### Directory Structure
```
spatial_eval/
├── download_spatial_eval.py      # Download datasets/resources
├── inference_vlm.py              # Main inference pipeline (CLI)
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

### Key Modules & Scripts
- **download_spatial_eval.py:** Download datasets/resources
- **inference_vlm.py:** Main CLI for model inference
- **evals/evaluation.py:** Core evaluation logic
- **eval_summary/plot_results.py:** Plotting and visualization
- **configs/inference_configs.py:** Model/config presets
- **models/**: Model wrappers/utilities
- **scripts/**: Batch and pipeline automation

### Data Flow Diagram
```
[Datasets (vqa/vtqa)]
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
```


### Testing & Usage Guide

#### 1. Configure API Key
Edit your `.env` file with OpenAI/Azure credentials:
```bash
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
```

#### 2. Run Inference
```bash
uv run python inference_vlm.py \
  --model_path gpt-4o \
  --mode vqa \
  --task mazenav \
  --first_k 10 \
  --max_new_tokens 1024 \
  --temperature 0.2
```

#### 3. Evaluate Results
```bash
uv run python evals/evaluation.py \
  --mode vqa \
  --task mazenav \
  --output_folder outputs/ \
  --dataset_id MilaWang/SpatialEval \
  --eval_summary_dir eval_summary
```

#### 4. Configuration Options

**Tasks:**
- `mazenav` - Maze navigation
- `spatialgrid` - Grid-based reasoning
- `spatialmap` - Map understanding
- `spatialreal` - Real-world scenarios

**Modes:**
- `vqa` - Vision + Text
- `tqa` - Text only
- `vtqa` - Vision + Text (text emphasis)

**Model Options:**
- `--model_path gpt-4o` (API-based)
- `--model_path llava-v1.5-7b` (local)
- Others: bunny, qwen, cog, instructblip

**Common Flags:**
- `--first_k 10` - Test first 10 samples
- `--temperature 0.2` - Sampling temperature
- `--max_new_tokens 1024` - Max response length
- `--w_reason` - Step-by-step reasoning
- `--completion` - Add "Answer:" prompt

#### 5. Evaluation Options

**Run evaluation** on any task/mode:
```bash
uv run python evals/evaluation.py --mode vqa --task mazenav
uv run python evals/evaluation.py --mode tqa --task spatialmap
uv run python evals/evaluation.py --mode vtqa --task spatialgrid
```
**Parameters:**
- `--mode {vqa,tqa,vtqa}`
- `--task {mazenav,spatialgrid,spatialmap,spatialreal}`
- `--output_folder outputs/`
- `--dataset_id MilaWang/SpatialEval`
- `--eval_summary_dir eval_summary`

#### 6. Output Files

- **Inference:** `outputs/MilaWang__SpatialEval/{mode}/{task}/m-{model}_{suffix}.jsonl`
  - Contains: id, answer, oracle_answer, oracle_option, prompt, image
- **Evaluation:**
  - Summary: `eval_summary/{mode}/{task}_{model}_eval_summary.jsonl`
  - Accuracy CSV: `eval_summary/{mode}/{task}_acc.csv`

#### 7. Usage Examples

**Test different tasks:**
```bash
uv run python inference_vlm.py --model_path gpt-4o --mode vqa --task mazenav --first_k 10
uv run python inference_vlm.py --model_path gpt-4o --mode vqa --task spatialgrid --first_k 10
uv run python inference_vlm.py --model_path gpt-4o --mode vqa --task all --first_k 5
```
**Evaluate multiple models:**
```bash
uv run python evals/evaluation.py --mode vqa --task mazenav
# Results show all models in mazenav_acc.csv
```

---

## 3. Re-downloading & References

- To re-download datasets: `uv run python download_spatial_eval.py`
- [Project Page](https://spatialeval.github.io/)
- [Paper](https://arxiv.org/pdf/2406.14852)
- [Dataset](https://huggingface.co/datasets/MilaWang/SpatialEval)
- [Code](https://github.com/jiayuww/SpatialEval)
- [NeurIPS Talk](https://neurips.cc/virtual/2024/poster/94371)

### Citation
```bibtex
@inproceedings{wang2024spatial,
  title={Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models},
  author={Wang, Jiayu and Ming, Yifei and Shi, Zhenmei and Vineet, Vibhav and Wang, Xin and Li, Yixuan and Joshi, Neel},
  booktitle={The Thirty-Eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

---
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

