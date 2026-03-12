# SpatialEval Testing Guide

## Prerequisites

### 1. Install dependencies
```bash
# From project root
uv sync
```

### 2. Set credentials in `.env` (project root)
```env
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=https://<azure-endpoint>/openai/v1/
MODEL_NAME=gpt-5.2
MODEL_REASONING_EFFORT=medium
```

### 3. Download dataset (first time only)
```bash
cd spatial_eval
uv run python download_spatial_eval.py
# Downloads ~1.43 GB into vqa/ and vtqa/
```

---

## Running the Experiment

All scripts are run from the `spatial_eval/` directory.

### Smoke test — end-to-end sanity check (~10 API calls)
```bash
bash scripts/smoke_test.sh
```
- Task: `mazenav`, 10 samples, VQA + VTQA modes, baseline + skills → 4 inference runs
- Writes to isolated `outputs_smoke_test/` and `eval_summary_smoke_test/` (never pollutes canonical data)
- Produces `eval_summary_smoke_test/mazenav_skills_comparison.png`

### Main experiment — 100 samples, all 3 tasks
```bash
bash scripts/run_experiment.sh
```
- Tasks: `mazenav`, `spatialgrid`, `spatialmap`
- Modes: `vqa`, `vtqa`
- Variants: baseline, with skills → **12 inference + 6 eval runs**
- Outputs: `outputs/MilaWang__SpatialEval/{mode}/{task}/m-*.jsonl`
- Summaries: `eval_summary/{mode}/{task}_acc.csv`
- Plots: `eval_summary/{task}_skills_comparison.png`

### Manual single inference run
```bash
uv run python inference_vlm.py \
  --model_path gpt-5.2 \
  --mode vqa \
  --task mazenav \
  --first_k 10 \
  --max_new_tokens 1024 \
  --output_folder outputs/
# Add --use_skills to enable the spatial DeepAgent
```

### Manual evaluation
```bash
uv run python evals/evaluation.py \
  --mode vqa \
  --task mazenav \
  --output_folder outputs/ \
  --eval_summary_dir eval_summary
```

### Generate comparison plots
```bash
# Single task, single round:
uv run python eval_summary/plot_skills_comparison.py \
  --eval_summary_dir eval_summary \
  --task mazenav \
  --out_dir eval_summary/result_vis

# All tasks, compute accuracy from outputs/ and generate plots:
uv run python eval_summary/compute_and_plot.py

# Multi-round mean ± std (requires rounds 2 & 3 from run_experiment_rounds.sh):
uv run python eval_summary/plot_skills_comparison_multi.py \
  --eval_dirs eval_summary eval_summary_round2 eval_summary_round3 \
  --tasks mazenav spatialgrid spatialmap \
  --out_dir eval_summary_stats
```

---

## CLI Reference

### `inference_vlm.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | required | Model name: `gpt-5.2`, `gpt-4o`, `lmsys/llava-v1.5-7b`, etc. |
| `--mode` | — | `vqa`, `vtqa`, or `tqa` |
| `--task` | `all` | `mazenav`, `spatialgrid`, `spatialmap`, `spatialreal`, or `all` |
| `--first_k` | all samples | Process first N samples per question type |
| `--offset_k` | 0 | Skip first N samples (for non-overlapping multi-round batches) |
| `--max_new_tokens` | 1024 | Max response length |
| `--temperature` | 0.2 | Sampling temperature |
| `--output_folder` | `outputs` | Root output directory |
| `--use_skills` | off | Enable DeepAgent spatial reasoning skills |
| `--w_reason` | off | Request step-by-step reasoning in prompt |
| `--device` | `cuda` | Device for local VLMs: `cuda` or `cpu` |

### `evals/evaluation.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | required | `vqa`, `vtqa`, or `tqa` |
| `--task` | required | `mazenav`, `spatialgrid`, `spatialmap`, or `spatialreal` |
| `--output_folder` | `outputs` | Match the folder used during inference |
| `--eval_summary_dir` | `eval_summary` | Output directory for CSVs and jsonl summaries |
| `--dataset_id` | `MilaWang/SpatialEval` | HuggingFace dataset identifier |

---

## Output File Format

### Inference output (`outputs/MilaWang__SpatialEval/{mode}/{task}/m-{model}_{variant}_{ts}.jsonl`)
Each line is a JSON object:
```json
{
  "id": "mazenav.0.42",
  "answer": "B",
  "oracle_answer": "B",
  "oracle_option": "left",
  "prompt": "...",
  "image": "<base64>"
}
```

### Accuracy CSV (`eval_summary/{mode}/{task}_acc.csv`)
```csv
Model Name,Acc
gpt-5.2_bare_20260306_135320,0.72
gpt-5.2_bare_skills_20260306_141016,0.81
```
- `Acc` is in [0, 1] range (multiply by 100 for %)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY` not set | Set in `.env` at project root |
| Dataset not found | Run `uv run python download_spatial_eval.py` |
| Answer extraction failures | Check regex patterns in `evals/evaluation.py` — patterns differ per model |
| Local VLM out of memory | Use `--device cpu` or reduce `--max_new_tokens` |
| Skills not used | Ensure `--use_skills` flag is passed; check `models/skills/` exists |
| JSONL corruption | Validate with `cat file.jsonl \| python -m json.tool` |
| Wrong Python version | Project requires `>=3.11,<3.12` |
