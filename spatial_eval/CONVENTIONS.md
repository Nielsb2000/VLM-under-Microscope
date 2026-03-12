# SpatialEval — Conventions & Work Requirements

This file captures the project's naming conventions, data organisation rules,
and the dos/don'ts established during development. It is the authoritative
reference for any AI agent or contributor working on this codebase.

---

## Folder Naming Conventions

| Folder | Purpose | Rule |
|--------|---------|------|
| `outputs/` | Canonical inference results | The single source of truth for GPT-5.2 runs |
| `outputs_smoke_test/` | Smoke test isolation | **Never** write smoke test output to `outputs/` |
| `outputs_round2/`, `outputs_round3/` | Multi-round experiment | Use for additional statistical rounds only |
| `eval_summary/` | Evaluation summaries + plot scripts | Tracked in git (scripts); data subdirs are gitignored |
| `eval_summary/vqa/` + `eval_summary/vtqa/` | Week 3 model results | Bunny, LLaVA, GPT-4o, GPT-5.1 (no timestamps) |
| `eval_summary/vqa/week6/` + `eval_summary/vtqa/week6/` | Week 6 results | Any file with `_bare_` in name |
| `eval_summary_smoke_test/` | Smoke test evaluation | Isolated, gitignored |
| `legacy/outputs_week3/` | Historical multi-model outputs | Do not delete; reference only |
| `legacy/eval_summary_week6_presentation/` | Round 1 results used in week 6 presentation | Do not delete; reference only |

---

## Output Filename Format

```
m-{model_name}_{variant}_{timestamp}.jsonl
```

Examples:
- `m-gpt-5.2_bare_20260306_135320.jsonl` — baseline, no skills
- `m-gpt-5.2_bare_skills_20260306_141016.jsonl` — with spatial skills
- `m-gpt-4o_bare.jsonl` — old format (week 3, no timestamp)

### Variant suffixes
- `_bare_` — no special flags (baseline)
- `_bare_skills_` — DeepAgent spatial skills enabled (`--use_skills`)
- `_w_reason_` — step-by-step reasoning prompt (`--w_reason`)

### Week labeling
- **Week 3**: Multi-model comparison — bunny, LLaVA, GPT-4o, GPT-5.1. Files have **no timestamp**.
- **Week 6**: GPT-5.2 skills vs baseline experiment. Files have **`_bare_` + timestamp**.

---

## Scripts

| Script | Purpose | Output dirs |
|--------|---------|-------------|
| `scripts/smoke_test.sh` | Sanity check — 10 samples, mazenav | `outputs_smoke_test/`, `eval_summary_smoke_test/` |
| `scripts/run_experiment.sh` | Main experiment — 100 samples, 3 tasks | `outputs/`, `eval_summary/` |
| `scripts/run_experiment_rounds.sh` | Rounds 2 & 3 for mean ± std stats | `outputs_round{N}/`, `eval_summary_round{N}/` |
| `scripts/deprecated/` | Old multi-model scripts | Kept for history; do not use |

**All scripts are run from the `spatial_eval/` directory**, not the project root.

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
- Don't add intermediate dev output folders (`outputs_foo/`, `eval_summary_bar/`) without
  documenting them here and adding gitignore rules.
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

## Security Notes

- API keys are loaded exclusively from `.env` via `config.py` — never hardcode them.
- `.env` is gitignored; never commit it.
- The AIO Sandbox container runs with `seccomp:unconfined` for browser automation — do not
  expose the sandbox port (8080) to external networks.
- JSONL output files may contain base64-encoded images — treat them as sensitive data.
- When evaluating model outputs, use the regex extraction pipeline; do not `eval()` model
  responses as code.
