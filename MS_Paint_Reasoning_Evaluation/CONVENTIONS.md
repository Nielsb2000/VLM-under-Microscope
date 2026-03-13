# MS Paint Reasoning Evaluation — Conventions & Work Requirements

This file captures naming conventions, data organisation rules, and the dos/don'ts for the MS Paint Reasoning Evaluation experiment.

---

## Result Directory Naming

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

## Answer File Naming

| Suffix | Meaning |
|--------|---------|
| `answer_{model}_skills.txt` | Run with DeepAgent spatial skills |
| `answer_{model}_noskills.txt` | Run with plain chat completion |

**Canonical suffix is `_skills` / `_noskills`** — do not use `_no_skills` (old inconsistent variant).

## Aggregated JSON Naming (dashboard_data/)

```
llm_results_{image_type}_{blur_level}_{models}_{reasoning_effort}_{skills_mode}.json
```

Example: `llm_results_color_heavy_blur_gpt-4o-gpt-5.1-gpt-5.2_medium_skills.json`

## Legacy Data

`Results/legacy/week3_no_skills/` — historical results using old naming scheme (no `{image_type}_` prefix, no skills). Contains subdirs: `original_heavy_blur`, `original_med_blur`, `original_no_blur`, `original_heavy_blur_high`. Do not delete — kept for reference.

---

## Debug Log Naming (logs/)

`tiny_test_eval.py` writes per-run debug logs to `logs/`:

```
logs/debug_{image_type}_{blur_type}_img{N}_q{N}_{model}_{skills_mode}_{timestamp}.txt
```

Example: `logs/debug_color_heavy_blur_img1_q2_gpt-5.2_skills_20260101_120000.txt`

The `logs/` directory is gitignored. Do not commit debug logs.

---

## Scripts

| Script | Role | Called from |
|--------|------|-------------|
| `Eval_script.py` | Batch entry point | Project root CLI |
| `tiny_test_eval.py` | Single image/question | `Eval_script.py` (subprocess) or direct CLI |
| `llm_check_answer.py` | Answer verification | Called by `Eval_script.py` |
| `viz/plot_accuracy_heatmap.py` | Per-run heatmap + accuracy bar chart | Project root CLI |
| `viz/plot_accuracy_by_blur.py` | Accuracy across blur levels | Project root CLI |
| `viz/plot_accuracy_all_conditions.py` | All models × all blur levels | Project root CLI |
| `viz/plot_accuracy_heavy_blur_high.py` | Heavy blur: two reasoning modes | Project root CLI |
| `viz/plot_token_time_stats.py` | Token/time stats | Project root CLI |
| `json_results_to_df.py` | Load dashboard data | Imported by `dashboard.py` or CLI |
| `dashboard.py` | Dash UI | Via Docker only |
| `process_images.py` | Regenerate images | One-time setup CLI |

**All scripts run from project root** (not from inside `MS_Paint_Reasoning_Evaluation/`).

---

## Dos and Don'ts

### DO
- Run the smoke test (`--smoke`) before and after changing inference or evaluation code.
- Use `--smoke` combined with a single model and effort level for fast iteration.
- Keep `Results/dashboard_data/` committed — it is the canonical aggregated data source.
- Keep `Results/legacy/week3_no_skills/` intact — historical reference.
- Use `_skills` / `_noskills` suffixes (not `_no_skills`).
- Run all scripts from the project root (paths are relative to it).
- Add new image types or blur levels to `process_images.py` before adding them to `Eval_script.py`.
- Update this file when a new naming convention or workflow rule is established.

### DON'T
- Don't run scripts from inside `MS_Paint_Reasoning_Evaluation/` — paths will break.
- Don't use `_no_skills` suffix — the canonical suffix is `_noskills`.
- Don't add `gpt-4o` to runs with `--reasoning-effort` > `low` — it will be silently skipped.
- Don't commit `Results/{image_type}_*` directories — they are gitignored and regeneratable.
- Don't clear `Results/dashboard_data/` without backing it up first.
- Don't delete `Results/legacy/` without explicit confirmation.
- Don't add new specialised one-off plotting scripts for every sub-experiment — use `viz/plot_accuracy_heatmap.py` and `viz/plot_accuracy_by_blur.py` with their CLI flags instead.

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| Answer files empty | API error or wrong env vars | Check `.env`; run `--smoke` with `--models gpt-5.2` |
| `gpt-4o` missing from results | Reasoning effort ≠ `low` | Expected — always skipped silently |
| Skills not activating | `skills/master-skill/SKILL.md` missing or malformed | Check YAML frontmatter in SKILL.md |
| Dashboard shows no data | `dashboard_data/` empty | Run batch experiment and `json_results_to_df.py` |
| `res_vis/` plots stale after rerun | Old plots not cleared | `rm -rf Results/res_vis/*` then rerun plot scripts |
| `ModuleNotFoundError: deepagents` | Package missing | `uv sync` from project root |
| `_no_skills` files not found by `llm_check_answer.py` | Old suffix `_no_skills` | Rerun with new `Eval_script.py` to generate `_noskills` files |
