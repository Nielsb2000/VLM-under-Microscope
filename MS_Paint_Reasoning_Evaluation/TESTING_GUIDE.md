# MS Paint Reasoning Evaluation — Testing Guide

All commands are run from the **project root** (`my-vscode-project/`), not from inside `MS_Paint_Reasoning_Evaluation/`.

---

## Prerequisites

```bash
uv sync                         # install dependencies
cp .env.example .env            # fill in OPENAI_API_KEY and OPENAI_BASE_URL
```

The dataset images live in `MS_Paint_Reasoning_Evaluation/MS_paint_images/` (committed).  
Results write to `MS_Paint_Reasoning_Evaluation/Results/` (gitignored except `dashboard_data/`).

---

## CLI Dimensions

Every experiment run is defined by four independent dimensions:

| Dimension | Flag | Values |
|-----------|------|--------|
| Image type | `--image-types` | `color`, `greyscale`, `inverted_greyscale` |
| Blur level | `--blur-levels` | `no_blur`, `med_blur`, `heavy_blur` |
| Reasoning effort | `--reasoning-effort` | `low`, `medium`, `high` |
| Skills | `--skills` | `yes` (DeepAgent), `no` (plain chat) |
| Model | `--models` | `gpt-4o`, `gpt-5.1`, `gpt-5.2` |

> **Note**: `gpt-4o` ignores `--reasoning-effort`; it is always skipped for effort ≠ `low`.

---

## 1. Smoke Test (single image, no API cost)

The `--smoke` flag limits the run to `img1` only across all selected questions.

```bash
# Quickest possible check — color, no blur, low effort, skills on, img1 only
uv run python MS_Paint_Reasoning_Evaluation/Eval_script.py \
  --image-types color \
  --blur-levels no_blur \
  --reasoning-effort low \
  --skills yes \
  --models gpt-5.2 \
  --smoke
```

Results land in:
```
MS_Paint_Reasoning_Evaluation/Results/color_no_blur_low/img1/q1/answer_gpt-5.2_skills.txt
```

---

## 2. Single Image / Single Question (tiny_test_eval.py)

For inspecting one specific case interactively:

```bash
# Usage
uv run python MS_Paint_Reasoning_Evaluation/tiny_test_eval.py \
  <image_type> <blur_type> <img_index> <q_index> <model> [reasoning_effort] [--skills yes|no]

# Examples
uv run python MS_Paint_Reasoning_Evaluation/tiny_test_eval.py \
  color no_blur 1 1 gpt-5.2 medium --skills yes

uv run python MS_Paint_Reasoning_Evaluation/tiny_test_eval.py \
  greyscale heavy_blur 3 2 gpt-4o none --skills no
```

Debug output is written to `MS_Paint_Reasoning_Evaluation/logs/debug_{image_type}_{blur}_img{N}_q{N}_{model}_{skills_mode}_{timestamp}.txt` (gitignored).

---

## 3. Batch Experiment (Eval_script.py)

`Eval_script.py` iterates all images × questions for the selected dimensions,  
calling `tiny_test_eval.py` per case and `llm_check_answer.py` for answer verification.

All flags accept **multiple values** so a single command can run a cross-product:

```bash
# Full experiment — all image types × all blur levels × medium effort × skills on/off × all models
uv run python MS_Paint_Reasoning_Evaluation/Eval_script.py \
  --image-types color greyscale inverted_greyscale \
  --blur-levels no_blur med_blur heavy_blur \
  --reasoning-effort medium \
  --skills yes no \
  --models gpt-4o gpt-5.1 gpt-5.2

# Focused run — heavy blur only, high effort, skills on, gpt-5.2 only
uv run python MS_Paint_Reasoning_Evaluation/Eval_script.py \
  --image-types color \
  --blur-levels heavy_blur \
  --reasoning-effort high \
  --skills yes \
  --models gpt-5.2
```

**Result directory pattern**: `Results/{image_type}_{blur_level}_{reasoning_effort}/imgN/qN/`  
**Answer file pattern**: `answer_{model}_skills.txt` or `answer_{model}_noskills.txt`

---

## 4. Visualising Results

All plotting scripts live in `viz/` and are run from the project root.  
All outputs save to `Results/res_vis/{tag}/` and are gitignored (fully regeneratable).  
All scripts read from `Results/dashboard_data/` — no local experiment run required.

### Per-run heatmap + accuracy bar chart
```bash
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heatmap.py \
  --image-type color --blur-level heavy_blur --reasoning-mode medium --skills-mode no_skills
# → Results/res_vis/color_heavy_blur_medium_no_skills/model_accuracy.png
# → Results/res_vis/color_heavy_blur_medium_no_skills/{model}_heatmap.png
```

### Accuracy comparison across blur levels
```bash
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_by_blur.py \
  --image-type color --blur-levels no_blur med_blur heavy_blur \
  --reasoning-mode medium --skills-mode no_skills
# → Results/res_vis/color_medium_no_skills/accuracy_by_blur.png
```

### All models × all blur levels
```bash
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_all_conditions.py \
  --image-type color --reasoning-mode medium --skills-mode no_skills
# → Results/res_vis/color_medium_no_skills/accuracy_all_blur.png
```

### Heavy blur: compare two reasoning modes
```bash
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heavy_blur_high.py \
  --image-type color --mode-a medium --mode-b high --skills-mode no_skills
# → Results/res_vis/color_heavy_blur_medium_vs_high_no_skills/heavy_blur_reasoning_comparison.png
```

### Token + time statistics
```bash
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_token_time_stats.py \
  --image-type color --blur-level heavy_blur --reasoning-mode medium --skills-mode no_skills
# → Results/res_vis/color_heavy_blur_medium_no_skills/token_time_stats.png
```

> **Note**: `gpt-4o` is always included in filtering regardless of `--reasoning-mode`
> (it has no reasoning effort parameter). To show only gpt-5.x models, pass `--models gpt-5.1 gpt-5.2`.

---

## 5. Parse Results to DataFrame

```bash
# Parse dashboard_data/ JSONs into DataFrame (printed to stdout)
uv run python MS_Paint_Reasoning_Evaluation/json_results_to_df.py
```

Debug logs from `tiny_test_eval.py` are written to `MS_Paint_Reasoning_Evaluation/logs/` — e.g. `debug_color_no_blur_img1_q1_gpt-5.2_skills_20260101_120000.txt`. These are gitignored.

---

## 6. Interactive Dashboard (Docker)

The Plotly Dash dashboard reads from `Results/dashboard_data/` JSONs.

```bash
cd MS_Paint_Reasoning_Evaluation
docker-compose up -d
# Open: http://localhost:8050
```

To rebuild after code changes:
```bash
docker-compose up -d --build
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `answer_*.txt` empty | API error or timeout | Check `.env`; rerun with `--smoke` first |
| `gpt-4o` skipped | Reasoning effort ≠ `low` | Expected — `gpt-4o` has no reasoning effort parameter |
| Skills not activating | `--skills yes` but skill files missing | Check `skills/master-skill/SKILL.md` exists |
| Dashboard shows no data | `dashboard_data/` empty | Run `json_results_to_df.py` or a full batch experiment |
| `ModuleNotFoundError: deepagents` | Missing package | `uv sync` from project root |
| Wrong paths in scripts | Running from inside `MS_Paint_Reasoning_Evaluation/` | Run all scripts from project root |
