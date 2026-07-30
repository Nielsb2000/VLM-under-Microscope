# Case study 1 visualization pipeline

This folder creates paper-style figures for comparing the two case-study-1 prompting strategies:

- **Direct optimization**: `exploratory == false`
- **Exploratory + optimization**: `exploratory == true`

The code loads both per-run `run_manifest.json` files and `multi_run_summary_*.json` files from an `outputs/case_study_1` directory, normalizes them into one tidy table, detects matched direct/exploratory cases with the same sample/randomization seeds and randomized filters, and exports both plots and CSV summaries.

## Installation

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/make_case1_figures.py \
  --input /home/nielsbroekhuizen/projects/my-vscode-project/outputs/case_study_1 \
  --output /home/nielsbroekhuizen/projects/my-vscode-project/outputs/case_study_1/figures_case1
```

The script writes `.pdf`, `.svg`, and `.png` versions of each figure. Use PDF/SVG for LaTeX or Overleaf whenever possible.

## Outputs

### CSVs

- `case1_all_runs_tidy.csv`: all loaded runs after normalization.
- `case1_matched_pairs_long.csv`: only runs belonging to direct/exploratory matched groups.
- `case1_matched_pairs_wide.csv`: one row per matched case with direct/exploratory columns and deltas.
- `case1_summary_statistics.csv`: method-level summary statistics and matched-pair delta summary.

### Figures

- `fig01_score_distributions`: unpaired final-score distribution per method.
- `fig02_matched_paired_scores`: paired direct-vs-exploratory comparison for identical starts. This is the main causal-comparison figure.
- `fig03_matched_score_delta`: distribution of paired score differences. Negative values favor exploratory prompting because lower score is better.
- `fig04_improvement_vs_start`: how much each method recovers from the randomized starting score.
- `fig05_cost_vs_quality`: final score versus number of filter adjustments.
- `fig06_filter_trajectories`: brightness/contrast trajectories for representative runs, if `actions/filter_trajectory.csv` exists.
- `fig07_best_run_contact_sheet`: randomized start, final result, and hidden reference thumbnails for top runs, if run image files exist.

## Interpretation notes

The code assumes `final_score`/`score` is a histogram-distance metric where **lower is better**, based on your manifest where `absolute_improvement = randomized_score - final_score`. If this changes, update axis labels in `caseviz/plots.py`.

The matched-pair key is intentionally conservative. It uses these fields when available:

```text
sample_seed, randomization_seed, sample_index, random_brightness, random_contrast
```

If too few of those exist, it falls back to:

```text
image, randomization_seed, random_brightness, random_contrast
```

## Recommended paper narrative

Use the figures in this order:

1. Show `fig02_matched_paired_scores` to explain that the same image and randomized starting point were run under both prompts.
2. Show `fig03_matched_score_delta` to summarize the effect size across matched cases.
3. Use `fig04_improvement_vs_start` to show whether exploration helps especially on harder randomized starts.
4. Use `fig05_cost_vs_quality` to discuss whether extra exploration costs more filter adjustments or VLM snapshots.
5. Use `fig07_best_run_contact_sheet` sparingly as a qualitative example, not as the main evidence.

## Extending

If you later add more case studies, the code can be reused by pointing `--input` to a different output root as long as the same manifest fields exist.
