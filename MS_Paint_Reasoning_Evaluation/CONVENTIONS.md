# MS Paint Reasoning Evaluation — Conventions, Workflow & CLI Reference

This is the single reference for naming conventions, how to run experiments, how to add new conditions, and the rules that keep results reproducible.

For a project overview and folder structure see [README.md](README.md).

---

## Naming Conventions

### Result Directories

```
Results/{image_type}_{blur_level}_{reasoning_effort}/imgN/qN/
```

| Component | Values |
|-----------|--------|
| `image_type` | `color`, `greyscale`, `inverted_greyscale` |
| `blur_level` | `no_blur`, `med_blur`, `heavy_blur` |
| `reasoning_effort` | `low`, `medium`, `high` |

Examples:
- `Results/color_heavy_blur_medium/img1/q1/`
- `Results/inverted_greyscale_no_blur_high/img3/q2/`

### Answer Files

| Suffix | Meaning |
|--------|---------|
| `answer_{model}_skills.txt` | Run with DeepAgent spatial skills |
| `answer_{model}_noskills.txt` | Run with plain chat completion |

**Use `_skills` / `_noskills`** — do not use `_no_skills` (old inconsistent variant).

### Aggregated JSON (dashboard_data/)

```
llm_results_{image_type}_{blur_level}_{models}_{reasoning_effort}_{skills_mode}.json
```

Example: `llm_results_color_heavy_blur_gpt-4o-gpt-5.1-gpt-5.2_medium_skills.json`

### Debug Logs (logs/)

```
logs/debug_{image_type}_{blur_type}_img{N}_q{N}_{model}_{skills_mode}_{timestamp}.txt
```

`logs/` is gitignored. Do not commit debug logs.

---

## Running the Experiment

**All commands run from the project root** (`my-vscode-project/`).

### Smoke Test (fast sanity check)
```bash
uv run python MS_Paint_Reasoning_Evaluation/evaluation/Eval_script.py \
  --image-types color --blur-levels no_blur \
  --reasoning-effort low --skills yes --models gpt-5.2 --smoke
```
- Evaluates img1 only; safe to run during development

### Single Image/Question (debug)
```bash
uv run python MS_Paint_Reasoning_Evaluation/evaluation/tiny_test_eval.py \
  color no_blur 1 1 gpt-5.2 medium --skills yes
```
Arguments: `<image_type> <blur_type> <img_index> <q_index> <model> [reasoning_effort] [--skills yes|no]`

Writes debug log to `logs/debug_*.txt` and answer to `Results/{conditions}/imgN/qN/answer_*.txt`.

### Full Batch Run
```bash
uv run python MS_Paint_Reasoning_Evaluation/evaluation/Eval_script.py \
  --image-types color greyscale inverted_greyscale \
  --blur-levels no_blur med_blur heavy_blur \
  --reasoning-effort medium \
  --skills yes no \
  --models gpt-4o gpt-5.1 gpt-5.2
```
Runs the full cross-product. `gpt-4o` is silently skipped for effort ≠ `low` — this is expected.

### Eval_script.py CLI Flags

| Flag | Values | Description |
|------|--------|-------------|
| `--image-types` | `color greyscale inverted_greyscale` | One or more image types |
| `--blur-levels` | `no_blur med_blur heavy_blur` | One or more blur levels |
| `--reasoning-effort` | `low medium high` | Single reasoning effort level |
| `--skills` | `yes no` | One or both |
| `--models` | `gpt-4o gpt-5.1 gpt-5.2` | One or more models |
| `--smoke` | flag | Restrict to img1 only |

---

## Generating Visualizations

All `viz/` scripts read from `Results/dashboard_data/` via `json_results_to_df.load_results_df()`.

### Heatmap + accuracy bar chart (single condition)
```bash
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heatmap.py \
  --image-type color --blur-level heavy_blur --reasoning-mode medium --skills-mode no_skills
```

### Accuracy across blur levels
```bash
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_by_blur.py \
  --image-type color --blur-levels no_blur med_blur heavy_blur --reasoning-mode medium
```

### All models × all blur levels
```bash
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_all_conditions.py \
  --image-type color --reasoning-mode medium --skills-mode no_skills
```

### Heavy blur: two reasoning modes compared
```bash
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_accuracy_heavy_blur_high.py \
  --image-type color --mode-a medium --mode-b high
```

### Token usage + elapsed time + cost
```bash
uv run python MS_Paint_Reasoning_Evaluation/viz/plot_token_time_stats.py \
  --image-type color --blur-level heavy_blur --reasoning-mode medium
```

All PNGs are saved to `Results/res_vis/`. Safe to clear and regenerate at any time.

### Presentation visualizations (Week 6)
```bash
# Run from the notebook runner in VS Code or:
uv run jupyter notebook MS_Paint_Reasoning_Evaluation/viz/Week6_presentation_visualizations.ipynb
```

---

## Regenerating Images

Only needed if original images change. Generates all blur/greyscale/inverted variants into the appropriate folders:
```bash
uv run python MS_Paint_Reasoning_Evaluation/utils/process_images.py
```

---

## Skills System

Skills are in `skills/master-skill/`. Each skill file is Markdown with YAML frontmatter (`name`, `description`). The master skill routes to the correct sub-skill at inference time.

To add a new sub-skill:
1. Create `skills/master-skill/<skill-name>/SKILL.md` with valid YAML frontmatter
2. Reference it from `skills/master-skill/SKILL.md` in the routing logic

---

## Dos and Don'ts

### DO
- Run the smoke test (`--smoke`) before and after changing inference or evaluation code.
- Keep `Results/dashboard_data/` committed — it is the canonical results archive.
- Keep `Results/legacy/week3_no_skills/` intact — historical reference, do not delete.
- Use `_skills` / `_noskills` answer file suffixes.
- Run all scripts from the **project root**.
- Update `README.md` and `CONVENTIONS.md` when changing naming or workflow.

### DON'T
- Don't run scripts from inside `MS_Paint_Reasoning_Evaluation/` — paths will break.
- Don't use `_no_skills` suffix — canonical is `_noskills`.
- Don't add `gpt-4o` to runs with `--reasoning-effort` > `low` — it will be silently skipped.
- Don't commit `Results/{image_type}_*` directories — they are gitignored and regeneratable.
- Don't clear `Results/dashboard_data/` without backing up first.
- Don't add new one-off plotting scripts for every sub-experiment — extend the existing `viz/` scripts with CLI flags instead.

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| Answer files empty | API error or wrong env vars | Check `.env`; run smoke test |
| `gpt-4o` missing from results | Reasoning effort ≠ `low` | Expected — always skipped silently |
| Skills not activating | `skills/master-skill/SKILL.md` missing or malformed YAML | Check YAML frontmatter |
| `res_vis/` plots stale after rerun | Old plots not cleared | `rm -rf Results/res_vis/*` then replot |
| `ModuleNotFoundError: deepagents` | Package missing | `uv sync` from project root |
| `_no_skills` files not found | Old suffix variant | Rerun with current `Eval_script.py` |
| Dashboard link broken | `legacy_dashboard/` is deprecated | Use `viz/` scripts or the notebook instead |

---

## Script Responsibility Matrix

| Script | Role | Called by |
|--------|------|-----------|
| `reproduce_results.py` | **Full reproducibility run** | CLI (project root) |
| `evaluation/Eval_script.py` | Batch orchestrator | CLI (project root) |
| `evaluation/tiny_test_eval.py` | Single image/question evaluator | `Eval_script.py` subprocess, or direct CLI |
| `evaluation/llm_check_answer.py` | Answer verifier (outputs `0` or `1`) | `Eval_script.py` subprocess |
| `evaluation/json_results_to_df.py` | Parses dashboard_data → DataFrame | Imported by `viz/*.py` and the notebook |
| `utils/process_images.py` | Regenerate all image variants | One-time setup CLI |
| `viz/plot_accuracy_heatmap.py` | Per-run heatmap + accuracy bar chart | CLI (project root) |
| `viz/plot_accuracy_by_blur.py` | Accuracy across blur levels | CLI (project root) |
| `viz/plot_accuracy_all_conditions.py` | All models × all blur levels | CLI (project root) |
| `viz/plot_accuracy_heavy_blur_high.py` | Heavy blur: two reasoning modes | CLI (project root) |
| `viz/plot_token_time_stats.py` | Token/time/cost stats | CLI (project root) |
| `viz/Week6_presentation_visualizations.ipynb` | Interactive Plotly charts for presentation | Jupyter / VS Code notebook runner |
| `legacy_dashboard/dashboard.py` | Deprecated Plotly Dash UI | Docker only (legacy_dashboard/) |
| `tests/test_deepagents_tools.py` | Dev debug — DeepAgent tool testing | Dev/debug only |
