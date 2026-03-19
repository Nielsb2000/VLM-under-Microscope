# SpatialEval — Spatial Reasoning Benchmarks for VLMs

Evaluation suite for benchmarking spatial reasoning in vision-language models (VLMs) and LLMs. Tests GPT-5.2 (skills vs baseline) across four spatial reasoning tasks using the [SpatialEval dataset](https://huggingface.co/datasets/MilaWang/SpatialEval).

**Paper**: [Is A Picture Worth A Thousand Words? (NeurIPS 2024)](https://arxiv.org/pdf/2406.14852)  
**Dataset**: [MilaWang/SpatialEval on HuggingFace](https://huggingface.co/datasets/MilaWang/SpatialEval) — ~1.43 GB, ~9,270 samples

---

## Folder Structure

```
spatial_eval/
├── configs/
│   └── inference_configs.py      # CLI argument parser (shared by all inference runs)
├── evals/
│   └── evaluation.py             # Answer extraction + accuracy computation
├── eval_summary/
│   ├── vqa/                      # Week 3: multi-model accuracy CSVs + jsonl (bunny, llava, gpt-4o, gpt-5.1)
│   │   └── week6/                # Week 6: gpt-5.2 & timestamped gpt-4o/5.1 bare runs
│   ├── vtqa/                     # same structure as vqa/
│   │   └── week6/
│   ├── result_vis/               # Saved comparison plots (PNG)
│   ├── compute_and_plot.py       # Batch accuracy computation + plot from outputs/
│   ├── plot_results.py           # Bar plots per task/mode (all models)
│   ├── plot_skills_comparison.py # Skills vs baseline bar chart (single round)
│   └── plot_skills_comparison_multi.py  # Multi-round mean ± std bar chart
├── legacy/
│   ├── outputs_week3/            # Week 3 outputs: multi-model (gpt-4o, gpt-5.1, bunny, llava)
│   └── eval_summary_week6_presentation/  # Round 1 of final experiment (week 6 presentation)
├── logs/                         # Run logs (gitignored)
├── models/
│   ├── gpt4_model.py             # GPT-4/5 Vision wrapper
│   ├── deepagent_model.py        # GPT-5.2 + spatial skills via DeepAgent
│   ├── bunny_model.py
│   ├── llava_model.py
│   ├── model_utils.py
│   ├── skills/                   # Baseline skill files for DeepAgent (no variant)
│   │   ├── master-skill/SKILL.md
│   │   ├── mazenav/SKILL.md
│   │   ├── spatialgrid/SKILL.md
│   │   └── spatialmap/SKILL.md
│   ├── skills_img_only/skills/   # --skills_variant img-only: image-path examples
│   │   ├── master-skill/SKILL.md
│   │   ├── mazenav/SKILL.md  (+assets/)
│   │   ├── spatialgrid/SKILL.md  (+assets/)
│   │   └── spatialmap/SKILL.md  (+assets/)
│   ├── skills_img_qa/skills/     # --skills_variant img-qa: image + worked Q&A
│   │   └── (same structure as skills_img_only/skills/)
│   └── skills_img_context/skills/ # --skills_variant img-context: image + domain context
│       └── (same structure as skills_img_only/skills/)
├── outputs/                      # Canonical outputs: gpt-5.2 skills vs baseline, 100 samples
│   └── MilaWang__SpatialEval/
│       ├── vqa/{task}/m-*.jsonl
│       └── vtqa/{task}/m-*.jsonl
├── scripts/
│   ├── smoke_test.sh             # Quick sanity check (10 samples, mazenav only)
│   ├── run_experiment.sh         # Main experiment (100 samples, all 3 tasks)
│   ├── run_experiment_rounds.sh  # Multi-round experiment for mean ± std
│   └── deprecated/               # Old multi-model scripts (kept for reference)
├── utils/
│   ├── format_filename.py        # Output path + filename formatting
│   └── load_image.py
├── vqa/                          # Dataset: VQA split (gitignored)
├── vtqa/                         # Dataset: VTQA split (gitignored)
├── inference_vlm.py              # Main inference CLI entry point
├── download_spatial_eval.py      # Download dataset from HuggingFace
├── TESTING_GUIDE.md              # Step-by-step usage guide
└── CONVENTIONS.md                # Codebase conventions and work requirements
```

---

## Tasks & Modalities

| Task | Description |
|------|-------------|
| `mazenav` | Navigate mazes by counting turns |
| `spatialgrid` | Spatial reasoning in grid environments |
| `spatialmap` | Map-based spatial relationships |
| `spatialreal` | Real-world spatial understanding |

| Mode | Input |
|------|-------|
| `vqa` | Image + question (vision + text) |
| `vtqa` | Image + text representation of image + question |
| `tqa` | Text only (no image) |

---

## Quick Start

### 1. Install dependencies
```bash
# From project root
uv sync
```

### 2. Configure credentials
```bash
# .env at project root
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=https://<azure-endpoint>/openai/v1/
MODEL_NAME=gpt-5.2
MODEL_REASONING_EFFORT=medium
```

### 3. Download dataset
```bash
cd spatial_eval
uv run python download_spatial_eval.py
```

### 4. Smoke test (verify end-to-end, ~10 API calls)
```bash
cd spatial_eval
bash scripts/smoke_test.sh
# Writes to: outputs_smoke_test/  and  eval_summary_smoke_test/
```

### 5. Run main experiment (100 samples × 3 tasks × 2 modes = 12 inference runs)
```bash
cd spatial_eval
bash scripts/run_experiment.sh
# Writes to: outputs/  and  eval_summary/
```

### 6. Evaluate a single run manually
```bash
cd spatial_eval
uv run python evals/evaluation.py \
  --mode vqa --task mazenav \
  --output_folder outputs/ \
  --eval_summary_dir eval_summary
```

### 7. Generate comparison plots
```bash
cd spatial_eval
# Single-round skills vs baseline:
uv run python eval_summary/plot_skills_comparison.py \
  --eval_summary_dir eval_summary --task mazenav --out_dir eval_summary/result_vis

# All tasks at once + accuracy computation from outputs/:
uv run python eval_summary/compute_and_plot.py
```

---

## Data Organisation

### Canonical results (`outputs/`)
GPT-5.2 skills vs baseline, 100 samples each, all 3 tasks × 2 modes.

Output filename format: `m-{model}_{variant}_{timestamp}.jsonl`
- `m-gpt-5.2_bare_{ts}.jsonl` — baseline (no skills)
- `m-gpt-5.2_bare_skills_{ts}.jsonl` — with spatial skills

### Evaluation summaries (`eval_summary/`)
- `{vqa,vtqa}/{task}_acc.csv` — accuracy per model (all experiments, live reference)
- `{vqa,vtqa}/` (non-`_bare_` jsonl) — Week 3 multi-model results (bunny / llava / gpt-4o / gpt-5.1)
- `{vqa,vtqa}/week6/` — Week 6 `_bare_` timestamped results (gpt-5.2 skills vs baseline)
- `result_vis/` — saved PNG plots

### Legacy data (`legacy/`)
- `outputs_week3/` — original multi-model inference outputs (week 3)
- `eval_summary_week6_presentation/` — round 1 results used for week 6 presentation

---

## Adding a New Model

1. Add an `elif` branch for the model in [inference_vlm.py](inference_vlm.py)
2. Add model-specific answer extraction patterns in [evals/evaluation.py](evals/evaluation.py)
3. Test with `--first_k 5` before a full run

---

## Citation

```bibtex
@inproceedings{wang2024spatial,
  title={Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models},
  author={Wang, Jiayu and Ming, Yifei and Shi, Zhenmei and Vineet, Vibhav and Wang, Xin and Li, Yixuan and Joshi, Neel},
  booktitle={The Thirty-Eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
