# MS Paint Reasoning Evaluation

Evaluates visual reasoning on MS Paint-style images with varying image transformations (blur, greyscale, inversion) using GPT-4o, GPT-5.1, and GPT-5.2. Measures how image degradation and DeepAgent **spatial skills** affect model accuracy.

---

## Folder Structure

```
MS_Paint_Reasoning_Evaluation/
├── Eval_script.py                        # Batch evaluation entry point
├── tiny_test_eval.py                     # Single image/question test
├── llm_check_answer.py                   # Automated answer checking
├── process_images.py                     # Generate blurred/greyscale/inverted images
├── json_results_to_df.py                 # Parse Results/dashboard_data/ → DataFrame
├── dashboard.py                          # Plotly Dash dashboard (served via Docker)
├── docker-compose.yml                    # Dashboard container
├── Dockerfile                            # Dashboard image
├── requirements-dashboard.txt            # Dashboard Python deps
├── TESTING_GUIDE.md                      # Step-by-step CLI reference
├── CONVENTIONS.md                        # Naming conventions, dos/don'ts
├── skills/                               # DeepAgent skill files
│   └── master-skill/
│       ├── SKILL.md                      # Master skill (routing authority)
│       ├── colored-images/SKILL.md
│       ├── grayscale-images/SKILL.md
│       ├── inverted-grayscale-images/SKILL.md
│       └── recognizing-shapes/SKILL.md   # + shape example images
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
├── Results/                              # All experiment output (gitignored except dashboard_data/)
│   ├── {image_type}_{blur_level}_{reasoning_effort}/   # Per-run results
│   │   └── imgN/qN/
│   │       ├── answer_{model}_skills.txt
│   │       └── answer_{model}_noskills.txt
│   ├── dashboard_data/                   # Aggregated JSON for dashboard (committed)
│   ├── res_vis/                          # Generated plots (gitignored)
│   └── legacy/
│       └── week3_no_skills/              # Historical results (old naming scheme, no skills)
├── logs/                                 # Debug logs from tiny_test_eval.py (gitignored)
├── viz/                                  # All visualization scripts
│   ├── plot_accuracy_heatmap.py          # Per-run heatmap + accuracy bar chart
│   ├── plot_accuracy_by_blur.py          # Accuracy comparison across blur levels
│   ├── plot_accuracy_all_conditions.py   # All models × all blur levels bar chart
│   ├── plot_accuracy_heavy_blur_high.py  # Heavy blur: compare two reasoning modes
│   └── plot_token_time_stats.py          # Token usage + elapsed time
└── tests/
    └── test_deepagents_tools.py          # Dev debug script for DeepAgent tool testing
```

Data flow:
**Images** → `Eval_script.py` → `tiny_test_eval.py` (per image/question) → answer `.txt` files  
→ `llm_check_answer.py` → correctness scores → `json_results_to_df.py` → `Results/dashboard_data/`  
→ `viz/plot_*.py` → PNGs in `Results/res_vis/`  
→ `dashboard.py` (Dash UI)

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

## Quick Start

```bash
# 1. Install dependencies (from project root)
uv sync

# 2. Smoke test — img1 only, color, no blur, low effort, skills on
uv run python MS_Paint_Reasoning_Evaluation/Eval_script.py \
  --image-types color --blur-levels no_blur \
  --reasoning-effort low --skills yes --models gpt-5.2 --smoke

# 3. Single image/question inspection
uv run python MS_Paint_Reasoning_Evaluation/tiny_test_eval.py \
  color no_blur 1 1 gpt-5.2 medium --skills yes

# 4. Full batch run (cross-product of selections)
uv run python MS_Paint_Reasoning_Evaluation/Eval_script.py \
  --image-types color greyscale inverted_greyscale \
  --blur-levels no_blur med_blur heavy_blur \
  --reasoning-effort medium \
  --skills yes no \
  --models gpt-4o gpt-5.1 gpt-5.2

# 5. Visualise results
uv run python MS_Paint_Reasoning_Evaluation/MSP_results.py \
  --image-type color --blur-level heavy_blur --reasoning-mode medium

# 6. Launch dashboard
cd MS_Paint_Reasoning_Evaluation && docker-compose up -d
# → http://localhost:8050
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for full CLI reference.

---

## Result File Naming

| File | Convention |
|------|-----------|
| Results directory | `Results/{image_type}_{blur_level}_{reasoning_effort}/` |
| Answer file (skills) | `answer_{model}_skills.txt` |
| Answer file (no skills) | `answer_{model}_noskills.txt` |
| Plot output | `Results/res_vis/{image_type}_{blur_level}_{reasoning_mode}/` |
| Aggregated JSON | `Results/dashboard_data/llm_results_{image_type}_{blur_level}_{models}_{effort}_{skills_mode}.json` |

---

## Skills System

Skills are Markdown files with YAML frontmatter located in `skills/master-skill/`. The master skill routes to the appropriate sub-skill based on image type and question content:

- `colored-images/SKILL.md` — colour image analysis
- `grayscale-images/SKILL.md` — greyscale image analysis
- `inverted-grayscale-images/SKILL.md` — inverted greyscale analysis
- `recognizing-shapes/SKILL.md` — shape identification (applied alongside any image type skill)

The agent reads `skills/master-skill/SKILL.md` first on every run.

---

## Legacy Data

`Results/legacy/week3_no_skills/` contains historical results from the week 3 experiment using the old naming scheme (no `{image_type}_` prefix, no skills). Do not delete — kept for historical reference.
