# SpatialEval — Reference Guide

This file is the single authoritative reference for **conventions**, **naming rules**, **CLI usage**, and **troubleshooting**. It replaces the former `CONVENTIONS.md` and `TESTING_GUIDE.md`.

---

## Prerequisites

### Install dependencies
```bash
# From project root
uv sync
```

### Set credentials in `.env` (project root)
```env
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=https://<azure-endpoint>/openai/v1/
MODEL_NAME=gpt-5.2
MODEL_REASONING_EFFORT=medium
```

### Download dataset (first time only)
```bash
cd spatial_eval
uv run python download_spatial_eval.py
# Downloads ~1.43 GB into vqa/ and vtqa/
```

**All scripts and manual commands are run from the `spatial_eval/` directory**, not the project root (exception: `eval_summary/compute_and_plot.py` must be run from the project root).

---

## Running the Experiment

### Smoke test — end-to-end sanity check (~10 API calls)
```bash
bash scripts/smoke_test.sh
```
- Task: `mazenav`, 10 samples, VQA + VTQA modes, baseline + skills → 4 inference runs
- Writes to isolated `outputs_smoke_test/` and `eval_summary_smoke_test/` (never pollutes canonical data)

### Main experiment — 100 samples, all 3 tasks
```bash
bash scripts/run_experiment.sh
```
- Tasks: `mazenav`, `spatialgrid`, `spatialmap`
- Modes: `vqa`, `vtqa`
- Variants: baseline, with skills → **12 inference + 6 eval runs**
- Outputs: `outputs/MilaWang__SpatialEval/{mode}/{task}/m-*.jsonl`
- Summaries: `eval_summary/{mode}/{task}_acc.csv`
- Plots: `eval_summary/result_vis/{task}_skills_comparison.png`

### Debug a single image (inspect agent tool calls)
```bash
uv run python inference_vlm.py \
  --model_path gpt-5.2 \
  --mode vqa \
  --task mazenav \
  --first_k 1 \
  --use_skills --skills_variant img-only \
  --output_folder outputs_smoke_test \
  --debug
```
The log is written to `logs/debug_<model>_<variant>_<task>_<mode>_<ts>.log`.

### Manual single inference run
```bash
# Baseline (no skills)
uv run python inference_vlm.py \
  --model_path gpt-5.2 \
  --mode vqa \
  --task mazenav \
  --first_k 10 \
  --output_folder outputs/

# With skills — choose a variant: img-only | img-qa | img-context | img-only-annotated | img-annotated-context
uv run python inference_vlm.py \
  --model_path gpt-5.2 \
  --mode vqa \
  --task mazenav \
  --first_k 10 \
  --use_skills --skills_variant img-only \
  --output_folder outputs/
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
  --out_dir eval_summary/result_vis

# MC skill variants (4-bar chart per task):
uv run python eval_summary/plot_mc_results.py --out_dir eval_summary/result_vis

# Image skill variants breakdown:
uv run python eval_summary/plot_image_skill_variants.py --out_dir eval_summary/result_vis

# Img-only scaling (static n3/n10/n30):
uv run python eval_summary/plot_img_only_static.py --out_dir eval_summary/result_vis

# Img-only range learning curve:
uv run python eval_summary/plot_img_only_range.py --out_dir eval_summary/result_vis

# Preload tool scaling (img-only-tool n3/n10/n30):
uv run python eval_summary/plot_img_only_tool.py --out_dir eval_summary/result_vis

# Preload Img+Q&A tool scaling:
uv run python eval_summary/plot_preload_scaling.py --out_dir eval_summary/result_vis
```

### Contamination validation test
```bash
# Single task (30 samples, VQA mode)
uv run python inference_vlm.py \
  --model_path gpt-5.2 \
  --mode vqa \
  --task mazenav \
  --first_k 30 \
  --use_skills --skills_variant img-qa-val-v2 \
  --output_folder outputs/

# Generate the 4-bar validation chart
uv run python eval_summary/plot_validation_test.py --out_dir eval_summary/result_vis
```
**Do not run MC iterations** for `img-qa-val-v2` — it is a single-run contamination check only.

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
| `--runs` | 1 | Number of deterministic repeat runs (same data each time) |
| `--mc_runs` | 0 | Monte Carlo runs: random sample without replacement per run |
| `--mc_seed` | 42 | Base random seed for MC runs |
| `--workers` | 1 | Parallel HTTP workers (GPT models only) |
| `--max_new_tokens` | 1024 | Max response length |
| `--temperature` | 0.2 | Sampling temperature |
| `--output_folder` | `outputs` | Root output directory |
| `--use_skills` | off | Enable DeepAgent spatial reasoning skills |
| `--skills_variant` | none | Skill variant (see Skill Variants table below) |
| `--w_reason` | off | Request step-by-step reasoning in prompt |
| `--debug` | off | Log all agent intermediate messages to `logs/debug_*.log` |
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
  "oracle_full_answer": "B. left",
  "prompt": "...",
  "image": "<base64 or path>"
}
```

### Accuracy CSV (`eval_summary/{mode}/{task}_acc.csv`)
```csv
Model Name,Acc
gpt-5.2_bare_20260306_135320,0.72
gpt-5.2_bare_skills_20260306_141016,0.81
```
`Acc` is in [0, 1] range (multiply by 100 for %).

---

## Naming Conventions

### Output filename format
```
m-{model_name}_{variant}_{timestamp}.jsonl
```
Examples:
- `m-gpt-5.2_bare_20260306_135320.jsonl` — baseline, no skills
- `m-gpt-5.2_bare_skills_20260306_141016.jsonl` — with spatial skills
- `m-gpt-4o_bare.jsonl` — old format (week 3, no timestamp)

### Variant suffixes in filenames
- `_bare_` — no special flags (baseline)
- `_bare_skills_` — DeepAgent spatial skills enabled (`--use_skills`)
- `_bare_skills_img-only_` / `_bare_skills_img-qa_` / `_bare_skills_img-context_` — MC variant runs
- `_bare_skills_img-qa-val-v2_` — preload-architecture validation run (no MC tag; single run)
- `_w_reason_` — step-by-step reasoning prompt (`--w_reason`)

### Week labeling
- **Week 3**: Multi-model comparison — bunny, LLaVA, GPT-4o, GPT-5.1. Files have **no timestamp**.
- **Week 6**: GPT-5.2 skills vs baseline experiment. Files have **`_bare_` + timestamp**.

### Folder naming
| Folder | Purpose | Rule |
|--------|---------|------|
| `outputs/` | Canonical inference results | The single source of truth for GPT-5.2 runs |
| `outputs_smoke_test/` | Smoke test isolation | **Never** write smoke test output to `outputs/` |
| `eval_summary/` | Evaluation summaries + plot scripts | Tracked in git (scripts); data subdirs are gitignored |
| `eval_summary_smoke_test/` | Smoke test evaluation | Isolated, gitignored |
| `legacy/outputs_week3/` | Historical multi-model outputs | Do not delete; reference only |
| `legacy/eval_summary_week6_presentation/` | Round 1 results used in week 6 presentation | Do not delete; reference only |

---

## Skill Variants

| `--skills_variant` | Description | Skill folder |
|--------------------|-------------|--------------|
| *(omitted)* | Baseline DeepAgent skills, no examples | `models/skills/` |
| `img-only` | Image paths embedded in SKILL.md | `models/skills_img_only/` |
| `img-only-annotated` | Annotated images embedded in SKILL.md | `models/skills_img_only_annotated/` |
| `img-only-n3` | 3 static images embedded | `models/skills_img_only_n3/` |
| `img-only-n10` | 10 static images embedded | `models/skills_img_only_n10/` |
| `img-only-n30` | 30 static images embedded | `models/skills_img_only_n30/` |
| `img-qa` | Images + worked Q&A embedded in SKILL.md | `models/skills_img_qa/` |
| `img-qa-val` | Images + full Q&A (validation holdout) | `models/skills_img_qa_val/` |
| `img-qa-val-v2` | Preload architecture: agent calls `read_example(n)` tool | `models/deepagent_preload_model.py` + `models/skills_img_qa_val_v2/examples/` |
| `img-context` | Images + domain context (no Q&A answers) | `models/skills_img_context/` |
| `img-annotated-context` | Annotated images + domain context | `models/skills_img_annotated_context/` |

### Preload architecture (`img-qa-val-v2`)
- Examples stored as `example_N.txt` + `example_N.png` under `models/skills_img_qa_val_v2/examples/{task}/`.
- `example_N.txt` begins with `# Image identifier: S=<pos>, E=<pos>` for unique identification via pixel analysis.
- The model is **never shown answers in a static skill file** — it calls `read_example(n)` and matches by visual features.
- Run as a **single contamination-validation run only**; do not use MC tags.

### Generator scripts
The assets in the skill folders were generated by one-off scripts, which have been run already. The generated outputs are committed to the repository. The scripts are preserved in `utils/generators/` for reference:

| Script | What it generated |
|--------|------------------|
| `generate_preload_examples.py` | `models/skills_img_qa_val_v2/examples/` |
| `regenerate_examples_10_29.py` | Fixed malformed examples 10–29 in same folder |
| `generate_img_qa_val_skill.py` | `models/skills_img_qa_val/` SKILL.md lookup tables |
| `generate_img_only_range_skills.py` | `models/skills_img_only_n10/` and `skills_img_only_n30/` SKILL.md |

---

## Scripts Reference

| Script | Purpose | Output dirs |
|--------|---------|-------------|
| `scripts/smoke_test.sh` | Sanity check — 10 samples, mazenav | `outputs_smoke_test/`, `eval_summary_smoke_test/` |
| `scripts/run_experiment.sh` | Main experiment — 100 samples, 3 tasks | `outputs/`, `eval_summary/` |
| `scripts/run_experiment_rounds.sh` | Multi-round for mean ± std | `outputs/`, additional round folders |
| `scripts/run_mc_experiment.sh` | Monte Carlo: 3 tasks × 4 configs × N iterations | `outputs/` |
| `scripts/run_skills_image_variants.sh` | 4 conditions × mazenav × VQA | `outputs/` |
| `scripts/run_skills_image_variants_spatialgrid.sh` | Same for spatialgrid | `outputs/` |
| `scripts/run_skills_image_variants_spatialmap.sh` | Same for spatialmap | `outputs/` |
| `scripts/run_mazenav_5x100_variants.sh` | 5 runs × 100 samples × 4 variants, mazenav | `outputs/` |
| `scripts/deprecated/` | Old multi-model scripts | Kept for history; do not use |

---

## Dos and Don'ts

### DO
- Run the smoke test before and after making changes to inference or evaluation code.
- Use isolated output folders (`outputs_smoke_test/`) for all test/dev runs.
- Keep `legacy/` data intact — it's historical reference, not waste.
- Ask before removing any script, output file, or data directory.
- Update this file whenever a new convention is established.
- Commit Python scripts in `eval_summary/` but not the data CSVs/jsonl (gitignored).
- Validate JSONL files after inference: `cat file.jsonl | python -m json.tool`.
- Always pass `--first_k 10` during development to avoid burning large API budgets.

### DON'T
- Don't write smoke test output to `outputs/` (the canonical results folder).
- Don't delete files in `legacy/` without explicit confirmation.
- Don't rename `outputs/` without updating `eval_summary/compute_and_plot.py` and scripts.
- Don't run `run_experiment.sh` without ensuring `OPENAI_API_KEY` is set.
- Don't change regex patterns in `evals/evaluation.py` without verifying on existing data first.
- Don't use `python3.12` — the project requires `>=3.11,<3.12`.
- Don't push large output files to git — `outputs/` and `eval_summary/vqa|vtqa/` are gitignored.

---

## Adding New Models

1. Add an `elif` branch for the model in `inference_vlm.py`.
2. Add model-specific answer extraction patterns in `evals/evaluation.py`.
3. Test accuracy extraction with `--first_k 5` before a full run.
4. If adding a DeepAgent-compatible model, also add a skill wrapper in `models/`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY` not set | Set in `.env` at project root |
| Dataset not found | Run `uv run python download_spatial_eval.py` |
| Answer extraction failures | Check regex patterns in `evals/evaluation.py` — patterns differ per model |
| Local VLM out of memory | Use `--device cpu` or reduce `--max_new_tokens` |
| Skills not used | Ensure `--use_skills` flag is passed; check that the relevant skill folder exists |
| JSONL corruption | Validate with `cat file.jsonl \| python -m json.tool` |
| Wrong Python version | Project requires `>=3.11,<3.12` |

---

## Security Notes

- API keys are loaded exclusively from `.env` via `config.py` — never hardcode them.
- `.env` is gitignored; never commit it.
- JSONL output files may contain base64-encoded images — treat them as sensitive data.
- When evaluating model outputs, use the regex extraction pipeline; do not `eval()` model responses as code.
