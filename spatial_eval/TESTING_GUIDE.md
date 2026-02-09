# SpatialEval Testing Guide

## Quick Start

### 1. Configure API Key
Edit `.env` file with your OpenAI/Azure credentials:
```bash
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
```

### 2. Run Inference
```bash
uv run python inference_vlm.py \
  --model_path gpt-4o \
  --mode vqa \
  --task mazenav \
  --first_k 10 \
  --max_new_tokens 1024 \
  --temperature 0.2
```

### 3. Evaluate Results
```bash
uv run python evals/evaluation.py \
  --mode vqa \
  --task mazenav \
  --output_folder outputs/ \
  --dataset_id MilaWang/SpatialEval \
  --eval_summary_dir eval_summary
```

## Configuration Options

### Inference Parameters

**Tasks** (choose one):
- `mazenav` - Maze navigation with turn counting
- `spatialgrid` - Grid-based spatial reasoning
- `spatialmap` - Map understanding and navigation
- `spatialreal` - Real-world spatial scenarios

**Modes** (input modality):
- `vqa` - Vision + Text (images + questions)
- `tqa` - Text only
- `vtqa` - Vision + Text with text emphasis

**Model Options**:
- `--model_path gpt-4o` - GPT-4o (API-based)
- `--model_path gpt-4o-mini` - GPT-4o mini (cheaper)
- `--model_path llava-v1.5-7b` - Local LLaVA model
- Other supported: bunny, qwen, cog, instructblip

**Common Flags**:
- `--first_k 10` - Test first 10 samples per question type
- `--temperature 0.2` - Sampling temperature (0.0 = deterministic)
- `--max_new_tokens 1024` - Max response length
- `--w_reason` - Request step-by-step reasoning
- `--completion` - Add "Answer:" prompt format

### Evaluation Options

**Run evaluation** on any task/mode combination:
```bash
# VQA mode
uv run python evals/evaluation.py --mode vqa --task mazenav

# TQA mode  
uv run python evals/evaluation.py --mode tqa --task spatialmap

# VTQA mode
uv run python evals/evaluation.py --mode vtqa --task spatialgrid
```

**Parameters**:
- `--mode {vqa,tqa,vtqa}` - Match inference mode
- `--task {mazenav,spatialgrid,spatialmap,spatialreal}` - Task to evaluate
- `--output_folder outputs/` - Where inference results are saved
- `--dataset_id MilaWang/SpatialEval` - Dataset identifier
- `--eval_summary_dir eval_summary` - Output directory for evaluation

## Output Files

**Inference**: `outputs/MilaWang__SpatialEval/{mode}/{task}/m-{model}_{suffix}.jsonl`
- Contains: id, answer, oracle_answer, oracle_option, prompt, image

**Evaluation**: 
- Summary: `eval_summary/{mode}/{task}_{model}_eval_summary.jsonl`
- Accuracy CSV: `eval_summary/{mode}/{task}_acc.csv`

## Usage Examples

**Test different tasks**:
```bash
# Maze navigation
uv run python inference_vlm.py --model_path gpt-4o --mode vqa --task mazenav --first_k 10

# Spatial grid
uv run python inference_vlm.py --model_path gpt-4o --mode vqa --task spatialgrid --first_k 10

# All tasks
uv run python inference_vlm.py --model_path gpt-4o --mode vqa --task all --first_k 5
```

**Evaluate multiple models**:
```bash
# Run inference for each model, then:
uv run python evals/evaluation.py --mode vqa --task mazenav

# Results show all models in mazenav_acc.csv
```
