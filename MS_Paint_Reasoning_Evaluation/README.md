# MS Paint Reasoning Evaluation

Evaluates visual reasoning on MS Paint-style images with varying image transformations (blur, greyscale, inversion) using GPT-4o, GPT-5.1, and GPT-5.2. Measures how image degradation and DeepAgent **spatial skills** affect model accuracy.

For naming conventions, CLI reference, and workflow rules see [CONVENTIONS.md](CONVENTIONS.md).

---

## Experiment Dimensions

Every run is parameterised by four independent dimensions:

| Dimension | Values |
|-----------|--------|
| **Image type** | `color`, `greyscale`, `inverted_greyscale` |
| **Blur level** | `no_blur`, `med_blur`, `heavy_blur` |
| **Reasoning effort** | `low`, `medium`, `high` |
| **Skills** | `yes` (DeepAgent), `no` (plain chat completion) |

Models: `gpt-4o`, `gpt-5.1`, `gpt-5.2`  
> `gpt-4o` ignores `--reasoning-effort`; it is automatically skipped for effort ≠ `low`.

---

## Folder Structure

```
MS_Paint_Reasoning_Evaluation/
├── reproduce_results.py                  # Single script: runs full experiment + all visualizations
├── README.md                             # This file
├── CONVENTIONS.md                        # Naming conventions, CLI reference, dos/don'ts
├── evaluation/                           # Evaluation pipeline scripts
│   ├── Eval_script.py                    # Batch evaluation entry point
│   ├── tiny_test_eval.py                 # Single image/question test; called by Eval_script.py
│   ├── llm_check_answer.py               # Automated answer checking; called by Eval_script.py
│   └── json_results_to_df.py             # Parse Results/dashboard_data/ → pandas DataFrame
├── utils/
│   └── process_images.py                 # One-time setup: regenerate all image variants
├── viz/                                  # Standalone visualization scripts
│   ├── Week6_presentation_visualizations.ipynb  # Interactive Plotly charts for presentation
│   ├── plot_accuracy_heatmap.py          # Per-run heatmap + accuracy bar chart
│   ├── plot_accuracy_by_blur.py          # Accuracy comparison across blur levels
│   ├── plot_accuracy_all_conditions.py   # All models × all blur levels bar chart
│   ├── plot_accuracy_heavy_blur_high.py  # Heavy blur: compare two reasoning modes
│   └── plot_token_time_stats.py          # Token usage + elapsed time stats
├── skills/                               # DeepAgent skill files used by tiny_test_eval.py
│   └── master-skill/
│       ├── SKILL.md                      # Master routing skill
│       ├── colored-images/SKILL.md
│       ├── grayscale-images/SKILL.md
│       ├── inverted-grayscale-images/SKILL.md
│       └── recognizing-shapes/SKILL.md
├── MS_paint_images/                      # Source images (committed)
│   ├── original_images/                  # color, no blur
│   ├── original_med_blur_images/         # color, med blur
│   ├── original_heavy_blur_images/       # color, heavy blur
│   ├── greyscale_images/                 # greyscale, no blur
│   ├── med_blur_greyscale_images/
│   ├── heavy_blur_greyscale_images/
│   ├── inverted_greyscale_images/
│   ├── med_blur_inverted_greyscale_images/
│   ├── heavy_blur_inverted_greyscale_images/
│   ├── MS paint questions/               # QuestionsN.txt per image
│   └── MS paint answers/                 # Ground truth answers
├── Results/                              # All experiment output
│   ├── {image_type}_{blur_level}_{reasoning_effort}/   # Per-run per-question answer files
│   │   └── imgN/qN/
│   │       ├── answer_{model}_skills.txt
│   │       └── answer_{model}_noskills.txt
│   ├── dashboard_data/                   # Aggregated JSONs — canonical results (committed)
│   ├── res_vis/                          # Generated plots (gitignored, regeneratable)
│   └── legacy/
│       └── week3_no_skills/              # Historical week 3 data (old naming scheme)
├── logs/                                 # Debug logs from tiny_test_eval.py (gitignored)
├── legacy_dashboard/                     # Deprecated Plotly Dash dashboard (no longer maintained)
│   ├── dashboard.py                      # Dash web UI (port 8050)
│   ├── json_results_to_df.py             # Copy of parser, path adjusted for this location
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── requirements-dashboard.txt
└── tests/
    └── test_deepagents_tools.py          # Dev debug script for DeepAgent tool testing
```

---

## Data Flow

```
[MS_paint_images/]
       │
       ▼
 utils/process_images.py  (one-time: generates blur/greyscale/inverted variants)
       │
       ▼
 evaluation/Eval_script.py  (batch: iterates image_type × blur × effort × skills)
       │  subprocess
       ├──► evaluation/tiny_test_eval.py   (per image/question: calls LLM or DeepAgent)
       │         │ writes
       │         ▼
       │   Results/{conditions}/imgN/qN/answer_{model}_{skills}.txt
       │  subprocess
       └──► evaluation/llm_check_answer.py  (auto-verifies answer → 0 or 1)
                 │ writes
                 ▼
           Results/dashboard_data/llm_results_*.json  (aggregated, committed)
                 │
       ┌─────────┴─────────────────────────────────┐
       ▼                                           ▼
  evaluation/json_results_to_df.py           (same module)
  (load_results_df → DataFrame)             imported by:
       │                                           │
       ▼                                    viz/plot_*.py
  viz/Week6_presentation_visualizations.ipynb    │
  (interactive Plotly charts for presentation)   ▼
                                          Results/res_vis/*.png
```

---

## Script Reference

### Core Pipeline

| Script | Role | How to call |
|--------|------|-------------|
| `reproduce_results.py` | **Full reproducibility** — eval + all visualizations in one command | CLI (project root) |
| `evaluation/Eval_script.py` | Batch orchestrator — cross-product of all dimensions | CLI (project root) |
| `evaluation/tiny_test_eval.py` | Single image/question evaluator | Called by `Eval_script.py` via subprocess; also directly callable |
| `evaluation/llm_check_answer.py` | LLM-based answer verifier (outputs `0` or `1`) | Called by `Eval_script.py` via subprocess |
| `utils/process_images.py` | Generates all image variants from originals | One-time setup CLI |
| `evaluation/json_results_to_df.py` | Parses `Results/dashboard_data/*.json` → DataFrame | Imported by viz scripts; can be run as CLI |

### Visualization

All `viz/` scripts are standalone CLI tools. They import `json_results_to_df.load_results_df()` and write PNGs to `Results/res_vis/`.

| Script | Output | Key flags |
|--------|--------|-----------|
| `viz/plot_accuracy_heatmap.py` | Heatmap + bar chart per model | `--image-type --blur-level --reasoning-mode --skills-mode` |
| `viz/plot_accuracy_by_blur.py` | Accuracy across blur progression | `--image-type --blur-levels (multiple) --reasoning-mode` |
| `viz/plot_accuracy_all_conditions.py` | All models × all blur levels | `--image-type --reasoning-mode --skills-mode` |
| `viz/plot_accuracy_heavy_blur_high.py` | Two reasoning modes vs heavy blur | `--mode-a --mode-b --models` |
| `viz/plot_token_time_stats.py` | Token usage + elapsed time + cost | `--image-type --blur-level --reasoning-mode` |

### Presentation Results (Week 6)

`viz/Week6_presentation_visualizations.ipynb` contains 13 cells with interactive Plotly charts covering:
- Accuracy across blur levels and image types
- Model comparison (grouped bars)
- Heatmaps (image_type × blur_level per model)
- Token/effort comparison
- Skills on/off comparison
- Token usage vs accuracy scatter

Run the notebook with `uv run jupyter notebook` or VS Code notebook runner. It loads from `Results/dashboard_data/` via `evaluation/json_results_to_df.load_results_df()`.

---

## Quick Start

### Reproduce all results (for supervisors / reviewers)
```bash
# From project root — runs full evaluation + all visualizations
uv run python MS_Paint_Reasoning_Evaluation/reproduce_results.py

# Skip evaluation, regenerate plots only (if Results/ already populated)
uv run python MS_Paint_Reasoning_Evaluation/reproduce_results.py --skip-eval

# Quick sanity check on img1 only (~few API calls)
uv run python MS_Paint_Reasoning_Evaluation/reproduce_results.py --smoke
```

### Manual step-by-step

```bash
# 1. Install dependencies
uv sync

# 2. Smoke test — img1 only, color, no blur, low effort, skills on
uv run python MS_Paint_Reasoning_Evaluation/evaluation/Eval_script.py \
  --image-types color --blur-levels no_blur \
  --reasoning-effort low --skills yes --models gpt-5.2 --smoke

# 3. Single image/question inspection
uv run python MS_Paint_Reasoning_Evaluation/evaluation/tiny_test_eval.py \
  color no_blur 1 1 gpt-5.2 medium --skills yes

# 4. Full batch run (cross-product of selections)
uv run python MS_Paint_Reasoning_Evaluation/evaluation/Eval_script.py \
  --image-types color greyscale inverted_greyscale \
  --blur-levels no_blur med_blur heavy_blur \
  --reasoning-effort medium \
  --skills yes no \
  --models gpt-4o gpt-5.1 gpt-5.2

# 5. Visualise a single condition
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heatmap.py \
  --image-type color --blur-level heavy_blur --reasoning-mode medium --skills-mode no_skills

# 6. Visualise across blur levels
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_by_blur.py \
  --image-type color --blur-levels no_blur med_blur heavy_blur --reasoning-mode medium
```

---

## Skills System

Skills are Markdown files with YAML frontmatter in `skills/master-skill/`. The master skill routes to the appropriate sub-skill based on image type:

| Skill | Purpose |
|-------|---------|
| `SKILL.md` | Master routing authority — always read first |
| `colored-images/SKILL.md` | Colour image analysis guidance |
| `grayscale-images/SKILL.md` | Greyscale image analysis guidance |
| `inverted-grayscale-images/SKILL.md` | Inverted greyscale guidance |
| `recognizing-shapes/SKILL.md` | Shape identification guidance |

---

## Results Data

- **`Results/dashboard_data/`** — 54 aggregated JSON files covering all combinations of image type × blur × reasoning effort × skills mode. This is the canonical results archive and is committed to git.
- **`Results/legacy/week3_no_skills/`** — Historical week 3 results (old naming scheme, no skills). Do not delete.
- **`Results/res_vis/`** — Generated plot PNGs. Gitignored and fully regeneratable.

---

## Deprecated

**`legacy_dashboard/`** contains the Plotly Dash web dashboard and Docker configuration. It is no longer actively maintained. The `Results/dashboard_data/` JSONs and the `viz/` scripts + `Week6_presentation_visualizations.ipynb` are the primary way to explore results.

To run the legacy dashboard anyway:
```bash
cd MS_Paint_Reasoning_Evaluation/legacy_dashboard && docker-compose up -d
# → http://localhost:8050
```
