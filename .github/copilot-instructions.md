# Copilot Instructions

## Project Overview

This is a **research-grade VLM/LLM evaluation and benchmarking suite** with **three distinct parts**. Each part is self-contained but all share the same `.env` config and `uv` virtual environment.

| Part | Folder | Entry point |
|------|--------|-------------|
| 1 — SpatialEval | `spatial_eval/` | `spatial_eval/inference_vlm.py` |
| 2 — MS Paint Reasoning Evaluation | `MS_Paint_Reasoning_Evaluation/` | `MS_Paint_Reasoning_Evaluation/Eval_script.py` |
| 3 — DeepAgent Framework | project root | `main.py` |

---

## Shared Setup

**Python version**: `>=3.11,<3.12`. Package manager: `uv`. The `.venv` lives at the project root.

```bash
# Install all dependencies
uv sync
```

API credentials go in a `.env` file at the project root (see [config.py](config.py)):
```env
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=https://<azure-endpoint>/openai/v1/
MODEL_NAME=gpt-5.2
MODEL_REASONING_EFFORT=medium   # low | medium | high (GPT-5.x only)
OPIK_ENABLED=false
HUGGINGFACEHUB_API_TOKEN=<token>  # For local HuggingFace model downloads
```

---

## Part 1 — SpatialEval

### Overview

Benchmarks spatial reasoning across 4 tasks (Spatial-Map, Maze-Nav, Spatial-Grid, Spatial-Real) using cloud VLMs (GPT-4o, GPT-5.x) and local models (LLaVA, Bunny). Supports three question modalities: VQA (image + text), VTQA (image + text with emphasis), TQA (text-only). Evaluation and plotting scripts live in `spatial_eval/evals/` and `spatial_eval/eval_summary/`.

**Tasks**: `mazenav`, `spatialgrid`, `spatialmap`, `spatialreal`, `all`  
**Modes**: `vqa` (vision+text), `tqa` (text-only), `vtqa` (vision+text emphasis)

### Setup

```bash
# Download the HuggingFace dataset (~1.4 GB)
uv run python spatial_eval/download_spatial_eval.py
```

### Commands

All scripts and single-run commands are executed from the `spatial_eval/` directory.

```bash
# Quick smoke test — 10 samples, mazenav, isolated output (never pollutes outputs/)
cd spatial_eval && bash scripts/smoke_test.sh

# Main experiment — 100 samples, all 3 tasks, both modes, baseline + skills (12 runs)
cd spatial_eval && bash scripts/run_experiment.sh

# Single inference run
uv run python inference_vlm.py \
  --model_path gpt-4o --mode vqa --task mazenav --first_k 10

# With spatial skills (DeepAgent)
uv run python inference_vlm.py \
  --model_path gpt-5.2 --mode vtqa --task spatialmap \
  --first_k 100 --offset_k 0 --use_skills

# Local VLM
uv run python inference_vlm.py \
  --model_path lmsys/llava-v1.5-7b --device cuda --mode vqa --task spatialgrid

# Single task/mode evaluation
uv run python evals/evaluation.py \
  --mode vqa --task mazenav \
  --output_folder outputs/ \
  --eval_summary_dir eval_summary

# Generate comparison plots
uv run python eval_summary/plot_skills_comparison.py \
  --eval_summary_dir eval_summary --task mazenav --out_dir eval_summary/result_vis

# Batch compute accuracy + plots for all tasks (run from project root)
uv run python spatial_eval/eval_summary/compute_and_plot.py
```

Results: `eval_summary/{mode}/{task}_acc.csv` and `{task}_{model}_eval_summary.jsonl`

### Folder Structure

```
spatial_eval/
├── configs/                      # CLI argument parser
├── evals/                        # Answer extraction + accuracy computation
├── eval_summary/
│   ├── vqa/                      # Week 3 multi-model results (acc.csv + jsonl)
│   │   └── week6/                # Week 6 timestamped _bare_ runs
│   ├── vtqa/                     # same as vqa/
│   │   └── week6/
│   ├── result_vis/               # Saved PNG comparison plots
│   ├── compute_and_plot.py       # Batch accuracy + plots from outputs/
│   ├── plot_results.py           # Per-task bar charts
│   ├── plot_skills_comparison.py # Skills vs baseline chart (single round)
│   └── plot_skills_comparison_multi.py  # Multi-round mean ± std
├── legacy/
│   ├── outputs_week3/            # Week 3 multi-model outputs (historical)
│   └── eval_summary_week6_presentation/ # Round 1 results (week 6 presentation)
├── models/                       # Model wrappers (gpt4, deepagent, bunny, llava)
│   ├── skills/                   # Skill files for DeepAgent (master, mazenav, …)
│   ├── skills_img_only/          # --skills_variant img-only
│   ├── skills_img_qa/            # --skills_variant img-qa
│   ├── skills_img_context/       # --skills_variant img-context
│   ├── deepagent_preload_model.py # --skills_variant img-qa-val-v2 (preload architecture)
│   └── skills_img_qa_val_v2/     # 10 example txt+png per task for preload variant
├── outputs/                      # Canonical: gpt-5.2 skills vs baseline, 100 samples
│   └── MilaWang__SpatialEval/{vqa,vtqa}/{task}/m-*.jsonl
├── scripts/
│   ├── smoke_test.sh             # Quick sanity check (10 samples)
│   ├── run_experiment.sh         # Main experiment (100 samples × 3 tasks)
│   ├── run_experiment_rounds.sh  # Multi-round replication
│   └── deprecated/               # Old multi-model scripts
├── utils/                        # format_filename.py, load_image.py
├── vqa/                          # Dataset VQA split (gitignored)
├── vtqa/                         # Dataset VTQA split (gitignored)
├── inference_vlm.py              # Main inference entry point
├── download_spatial_eval.py      # HuggingFace dataset downloader
├── TESTING_GUIDE.md              # CLI reference + step-by-step usage
└── CONVENTIONS.md                # Codebase conventions and work requirements
```

Data flow: **Inference** → load dataset → format prompt → call LLM API → save JSONL  
→ **Evaluation** → load JSONL → regex-extract answers → compare to oracle → CSV/summary

### Key Files

| File | Purpose |
|------|---------|
| [spatial_eval/inference_vlm.py](spatial_eval/inference_vlm.py) | Main VLM inference CLI |
| [spatial_eval/evals/evaluation.py](spatial_eval/evals/evaluation.py) | Answer extraction + accuracy metrics |
| [spatial_eval/configs/inference_configs.py](spatial_eval/configs/inference_configs.py) | CLI argument parser for inference |
| [spatial_eval/models/gpt4_model.py](spatial_eval/models/gpt4_model.py) | GPT-4/5 model wrapper |
| [spatial_eval/models/deepagent_model.py](spatial_eval/models/deepagent_model.py) | GPT-5.2 + DeepAgent spatial skills wrapper |
| [spatial_eval/models/deepagent_preload_model.py](spatial_eval/models/deepagent_preload_model.py) | Preload architecture: agent calls `read_example(n)` tool at inference time |
| [spatial_eval/eval_summary/compute_and_plot.py](spatial_eval/eval_summary/compute_and_plot.py) | Batch accuracy computation + plot generation |
| [spatial_eval/eval_summary/plot_mc_results.py](spatial_eval/eval_summary/plot_mc_results.py) | MC run mean ± SD bar chart (4 skill variants) |
| [spatial_eval/eval_summary/plot_validation_test.py](spatial_eval/eval_summary/plot_validation_test.py) | Contamination validation: 4-bar chart (baseline → img-qa → img-qa-val → img-qa-val-v2) |
| [spatial_eval/models/skills/](spatial_eval/models/skills/) | Spatial reasoning skill files for DeepAgent |
| [spatial_eval/scripts/](spatial_eval/scripts/) | Active scripts: `smoke_test.sh`, `run_experiment.sh`, `run_experiment_rounds.sh` |
| [spatial_eval/TESTING_GUIDE.md](spatial_eval/TESTING_GUIDE.md) | Step-by-step CLI reference |
| [spatial_eval/CONVENTIONS.md](spatial_eval/CONVENTIONS.md) | Naming conventions, dos/don'ts |

### Conventions

- **Result filenames**: `m-{model}_{variant}_{timestamp}.jsonl` (e.g., `m-gpt-5.2_bare_20260306_135320.jsonl`)
- **Variant labels**: `_bare_` = no skills (baseline); `_bare_skills_` = with DeepAgent spatial skills; `_bare_skills_img-qa-val-v2_` = preload validation (single run only, no MC tag)
- **Week labeling**: Week 3 = multi-model comparison (bunny/llava/gpt-4o/gpt-5.1, no timestamps); Week 6 = gpt-5.2 skills vs baseline (timestamped, `_bare_` prefix)
- **Skill files**: Markdown with YAML frontmatter (`name`, `description`), stored in `spatial_eval/models/skills/<skill-name>/SKILL.md`
- **Preload architecture** (`img-qa-val-v2`): examples stored as `example_N.txt` + `example_N.png` under `models/skills_img_qa_val_v2/examples/{task}/`. The agent calls `read_example(n)` at inference time — no static answer embedding in SKILL.md.
- **Answer extraction**: Regex-based in `evals/evaluation.py`; patterns are model-specific
- **Data I/O**: JSONL (one JSON object per line) for inference results; JSON/CSV for evaluation output
- **Reasoning modes**: `none`, `low`, `medium`, `high` — only GPT-5.x supports non-`none` modes
- **Batching with offset**: Use `--first_k <n> --offset_k <start>` for non-overlapping dataset slices
- **Legacy data**: kept in `spatial_eval/legacy/` — do not delete; historical reference only
- **Smoke test isolation**: always write to `outputs_smoke_test/` — never to canonical `outputs/`
- Add new model support in [spatial_eval/inference_vlm.py](spatial_eval/inference_vlm.py) via `if/elif` branches

### Common Pitfalls

- **API rate limits**: use `--first_k 10` during development; split runs for large batches.
- **Answer extraction failures**: regex patterns are task- and model-specific — extend them when adding new models.
- **Local VLM OOM**: pass `--device cpu` or reduce `--max_new_tokens`; GPU recommended for ≥7B models.
- **JSONL corruption**: each line must be valid JSON — validate with `cat file.jsonl | python -m json.tool`.
- **Wrong working directory**: `inference_vlm.py` uses relative imports and must be run from `spatial_eval/`.
- **`outputs_100/` not found**: renamed to `outputs/` after reorganisation — use `outputs/` everywhere.

---

## Part 2 — MS Paint Reasoning Evaluation

### Overview

Evaluates visual reasoning on MS Paint-style images with varying transformations (heavy/medium blur, greyscale, inversion) using GPT-4o, GPT-5.1, and GPT-5.2. Measures how image degradation and DeepAgent **spatial skills** affect model accuracy.

Every run is parameterised by four independent dimensions:

| Dimension | Values |
|-----------|--------|
| **Image type** | `color`, `greyscale`, `inverted_greyscale` |
| **Blur level** | `no_blur`, `med_blur`, `heavy_blur` |
| **Reasoning effort** | `low`, `medium`, `high` |
| **Skills** | `yes` (DeepAgent), `no` (plain chat) |

> `gpt-4o` ignores `--reasoning-effort` and is skipped for effort ≠ `low`.

### Setup

No extra downloads needed — images are committed in `MS_Paint_Reasoning_Evaluation/MS_paint_images/`.  
To regenerate processed image variants from originals:
```bash
uv run python MS_Paint_Reasoning_Evaluation/process_images.py
```

### Commands

All commands run from the **project root**.

```bash
# Smoke test — img1 only, color, no blur, low effort, skills on
uv run python MS_Paint_Reasoning_Evaluation/Eval_script.py \
  --image-types color --blur-levels no_blur \
  --reasoning-effort low --skills yes --models gpt-5.2 --smoke

# Single image/question inspection
uv run python MS_Paint_Reasoning_Evaluation/tiny_test_eval.py \
  color no_blur 1 1 gpt-5.2 medium --skills yes

# Full batch run (each flag accepts multiple values — runs full cross-product)
uv run python MS_Paint_Reasoning_Evaluation/Eval_script.py \
  --image-types color greyscale inverted_greyscale \
  --blur-levels no_blur med_blur heavy_blur \
  --reasoning-effort medium \
  --skills yes no \
  --models gpt-4o gpt-5.1 gpt-5.2

# Per-run heatmap + accuracy plot
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heatmap.py \
  --image-type color --blur-level heavy_blur --reasoning-mode medium --skills-mode no_skills

# Accuracy comparison across blur levels
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_by_blur.py \
  --image-type color --blur-levels no_blur med_blur heavy_blur --reasoning-mode medium

# Export results to DataFrame
uv run python MS_Paint_Reasoning_Evaluation/json_results_to_df.py

# Launch interactive dashboard
cd MS_Paint_Reasoning_Evaluation && docker-compose up -d
# → http://localhost:8050
```

### Folder Structure

```
MS_Paint_Reasoning_Evaluation/
├── Eval_script.py                         # Batch evaluation entry point
├── tiny_test_eval.py                      # Single image/question test
├── llm_check_answer.py                    # Automated answer checking
├── process_images.py                      # Generate blurred/greyscale/inverted images
├── json_results_to_df.py                  # Parse dashboard_data/ → DataFrame
├── dashboard.py                           # Plotly Dash dashboard (Docker)
├── docker-compose.yml / Dockerfile        # Dashboard container
├── viz/                                   # All visualization scripts
│   ├── plot_accuracy_heatmap.py           # Per-run heatmap + accuracy bar chart
│   ├── plot_accuracy_by_blur.py           # Accuracy across blur levels
│   ├── plot_accuracy_all_conditions.py    # All models × blur levels
│   ├── plot_accuracy_heavy_blur_high.py   # Heavy blur: two reasoning modes
│   └── plot_token_time_stats.py           # Token usage + elapsed time
├── skills/master-skill/                   # DeepAgent skill files
│   ├── SKILL.md                           # Master routing skill
│   ├── colored-images/SKILL.md
│   ├── grayscale-images/SKILL.md
│   ├── inverted-grayscale-images/SKILL.md
│   └── recognizing-shapes/SKILL.md
├── MS_paint_images/                       # Source images (committed)
│   ├── original_images/                   # color, no blur
│   ├── original_{med,heavy}_blur_images/
│   ├── {greyscale,inverted_greyscale}_images/
│   ├── {med,heavy}_blur_{greyscale,inverted_greyscale}_images/
│   ├── MS paint questions/                # QuestionsN.txt per image
│   └── MS paint answers/                  # Ground truth
├── Results/                               # All output (gitignored except dashboard_data/)
│   ├── {image_type}_{blur_level}_{reasoning_effort}/imgN/qN/
│   │   ├── answer_{model}_skills.txt
│   │   └── answer_{model}_noskills.txt
│   ├── dashboard_data/                    # Aggregated JSONs (committed)
│   ├── res_vis/                           # Generated plots (gitignored)
│   └── legacy/week3_no_skills/            # Historical — old naming, no skills
├── logs/                                  # Debug logs from tiny_test_eval.py (gitignored)
├── tests/                                 # Dev/debug scripts
├── TESTING_GUIDE.md                       # Step-by-step CLI reference
└── CONVENTIONS.md                         # Naming conventions, dos/don'ts
```

### Key Files

| File | Purpose |
|------|---------|
| [MS_Paint_Reasoning_Evaluation/Eval_script.py](MS_Paint_Reasoning_Evaluation/Eval_script.py) | Batch evaluation entry point |
| [MS_Paint_Reasoning_Evaluation/tiny_test_eval.py](MS_Paint_Reasoning_Evaluation/tiny_test_eval.py) | Single image/question test; called by `Eval_script.py` |
| [MS_Paint_Reasoning_Evaluation/llm_check_answer.py](MS_Paint_Reasoning_Evaluation/llm_check_answer.py) | Automated answer checking |
| [MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heatmap.py](MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heatmap.py) | Per-run heatmap + accuracy plot |
| [MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_by_blur.py](MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_by_blur.py) | Accuracy comparison across blur levels |
| [MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_all_conditions.py](MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_all_conditions.py) | All models × all blur levels bar chart |
| [MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heavy_blur_high.py](MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heavy_blur_high.py) | Heavy blur: two reasoning modes comparison |
| [MS_Paint_Reasoning_Evaluation/viz/plot_token_time_stats.py](MS_Paint_Reasoning_Evaluation/viz/plot_token_time_stats.py) | Token usage + elapsed time stats |
| [MS_Paint_Reasoning_Evaluation/json_results_to_df.py](MS_Paint_Reasoning_Evaluation/json_results_to_df.py) | Load `dashboard_data/` JSONs into DataFrame |
| [MS_Paint_Reasoning_Evaluation/dashboard.py](MS_Paint_Reasoning_Evaluation/dashboard.py) | Plotly Dash app (served via Docker) |
| [MS_Paint_Reasoning_Evaluation/TESTING_GUIDE.md](MS_Paint_Reasoning_Evaluation/TESTING_GUIDE.md) | Step-by-step CLI reference |
| [MS_Paint_Reasoning_Evaluation/CONVENTIONS.md](MS_Paint_Reasoning_Evaluation/CONVENTIONS.md) | Naming conventions, dos/don'ts |

### Common Pitfalls

- **Wrong working directory**: all scripts use paths relative to the project root — always run from there.
- **`gpt-4o` skipped silently**: expected; `gpt-4o` has no reasoning effort parameter.
- **Inconsistent answer filename suffixes**: canonical is `_skills` / `_noskills` — avoid `_no_skills`.
- **Dashboard shows no data**: `Results/dashboard_data/` must be populated by running a batch experiment first.
- **`res_vis/` messy**: it is fully regeneratable — safe to clear and replot from canonical data.

---

## Part 3 — DeepAgent Framework

### Overview

An interactive agent loop that executes code and reasoning tasks inside an AIO Sandbox container (Docker, port 8080). The agent is driven by modular skill files (`skills/master-skill/`) and uses MCP JSON-RPC to communicate with the sandbox. Also powers the `--use_skills` flag in SpatialEval inference.

### Setup

```bash
# Start the AIO Sandbox container
docker-compose up -d

# Verify sandbox connectivity
python test_sandbox.py
# or: curl http://localhost:8080/v1/sandbox
```

### Commands

```bash
python main.py
```

### Architecture

```
main.py                            ← Interactive agent loop
    │
    ├── llm_client.py              DeepAgent setup + system prompts
    ├── config.py                  .env loader (API keys, model name)
    ├── agent_tools.py             Tool defs (browser, skills, MCP)
    ├── skills_utils.py            Skill discovery + metadata parsing
    ├── sandbox_backend.py         File ops in AIO Sandbox
    ├── sandbox_core_functions.py  Shell/Python exec in sandbox
    └── MCP_functions.py           MCP JSON-RPC calls to sandbox hub
```

Skills are mounted into the sandbox at `/workspace/skills/`. Screenshots go to `/workspace/screenshots/`. Files on the **host** at `./skills/` map to `/workspace/skills/` inside the container.

### Docker / Sandbox

The AIO Sandbox exposes:
- **API**: `http://localhost:8080/v1/` (MCP tools, code execution)
- **VNC**: `http://localhost:8080/vnc/index.html?autoconnect=true`
- **Docs**: `http://localhost:8080/v1/docs`

### Key Files

| File | Purpose |
|------|---------|
| [main.py](main.py) | Interactive agent loop |
| [llm_client.py](llm_client.py) | Agent setup, system prompts, skill integration |
| [agent_tools.py](agent_tools.py) | Tool definitions (browser, skills, MCP) |
| [skills_utils.py](skills_utils.py) | Skill discovery + metadata parsing |
| [sandbox_backend.py](sandbox_backend.py) | File ops in AIO Sandbox |
| [sandbox_core_functions.py](sandbox_core_functions.py) | Shell/Python exec in sandbox |
| [MCP_functions.py](MCP_functions.py) | MCP JSON-RPC calls to sandbox hub |
| [config.py](config.py) | Loads `.env`, exposes `OPENAI_API_KEY`, `MODEL_NAME`, etc. |

### Common Pitfalls

- **Sandbox not running**: most agent features fail silently — always `docker-compose up -d` first.
- **Skill not discovered**: check that `skills/master-skill/<name>/SKILL.md` exists with valid YAML frontmatter.
- **Python version conflict**: project requires `>=3.11,<3.12`; do not use 3.12+.
