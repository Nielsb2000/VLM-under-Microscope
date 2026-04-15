# Plotting Guide

This guide defines the visual standards for all plots in this project and
documents exactly what each script produces, what every plot must contain, and
what it must not do. Follow this guide whenever adding or editing a plot.

---

## 1. Shared Style Constants

### 1.1 Okabe-Ito Colorblind-Safe Palette

All semantic colors come from the Okabe-Ito palette. **Never** use matplotlib
defaults (`tab:blue`, `#1f77b4`, `#ff7f0e`, `#2ca02c`, `#d62728`, `tab10`, etc.)
for bar, line, or patch colors.

| Role | Hex | Name |
|------|-----|------|
| Baseline / No-skills | `#555555` | Dark grey |
| Skill variant – light / n-small | `#56B4E9` | Sky blue |
| Skill variant – mid / n-medium | `#0072B2` | Deep blue |
| Skill variant – deep / n-large | `#332288` | Indigo |
| Biased skill (Img+Q&A) | `#E69F00` | Orange |
| Unbiased / context skill | `#009E73` | Teal |
| Contamination / warning | `#D55E00` | Vermillion |
| Contamination secondary | `#CC79A7` | Reddish purple |
| **Δ positive annotation** | `#009E73` | Teal |
| **Δ negative annotation** | `#D55E00` | Vermillion |

> The Δ annotation colors are always `#009E73` / `#D55E00` across **all** scripts.
> Never use `#2ca02c`, `#d62728`, or `#1a7a1a` for delta text.

### N-range sequential blues

Whenever a plot sweeps over N example images (n=3, 10, 30, …), **every distinct
N value must have its own bar** and the bars must progress from lighter to
darker blue as N grows:

| N | Hex | Name |
|---|-----|------|
| n=3 (smallest) | `#56B4E9` | Sky blue |
| n=10 | `#0072B2` | Deep blue |
| n=30 | `#332288` | Indigo |
| n=50 | `#084594` | Very dark blue |
| n=100 (largest) | `#08306b` | Darkest blue |

Each N-value condition **must appear in the legend** with its color and explicit
`n=X` label (e.g. `"Img-Only (n=10)"`). Never collapse them into a single bar
or omit any from the legend.

### 1.2 MS Paint Model Colors (viz/ scripts only)

All `MS_Paint_Reasoning_Evaluation/viz/` scripts use the same model-color mapping.
Define it once per file as:

```python
MODEL_COLORS = {
    "gpt-4o":  "#56B4E9",   # sky blue
    "gpt-5.1": "#E69F00",   # orange
    "gpt-5.2": "#009E73",   # teal
}
```

Use `MODEL_COLORS.get(model, "#0072B2")` for any unexpected fourth model.

### 1.3 Token-chart type colors (plot_token_time_stats.py only)

```python
TOKEN_COLORS = [
    ("Input",  "#56B4E9"),   # sky blue
    ("Output", "#E69F00"),   # orange
    ("Total",  "#009E73"),   # teal
]
```

---

## 1.4 Smoke-Test Exclusion

A **smoke-test run** is any JSONL output file with fewer than
`MIN_STAT_ITEMS = 20` evaluated items, or any file that lacks the Monte Carlo
tag (`_mc[0-9]{2}s[0-9]+_`) when the plot is an MC-aggregation plot.

Rules for every plot script:

- **JSONL-reading scripts (MC plots)**: call `_is_mc_file(fname)` and skip
  files where it returns `False`. This already excludes all smoke-test and
  one-off runs whose filenames have no `_mc` tag.
- **JSONL-reading scripts that intentionally accept single runs** (e.g.
  contamination-validation, preload scaling): add
  `_item_count(path) < MIN_STAT_ITEMS → continue` after the file is
  selected, before computing accuracy.
- **CSV-reading scripts**: use exact trailing-underscore patterns in
  `classify()` (e.g. `"_skills_img-only_"` not `"_skills_img-only"`) so
  n-variant single-run entries are never misclassified into the base variant.

Define `MIN_STAT_ITEMS = 20` as a module-level constant and
`_item_count(path)` as a helper in every script that accepts non-MC files.

---

## 2. Figure Settings

| Property | Rule |
|----------|------|
| **Backend** | `matplotlib.use("Agg")` — always, at module level, before `import matplotlib.pyplot` |
| **DPI** | `dpi=150` — always on `plt.savefig()` |
| **Close** | `plt.close()` — always immediately after `savefig()` |
| **Layout** | `plt.tight_layout()` — always before `savefig()` |
| **Single accuracy panel** | `figsize=(9, 5.5)` |
| **Single wide panel (6+ bars)** | `figsize=(11, 5.5)` |
| **3-panel combined** | `figsize=(18, 5.5)` or `figsize=(22, 5.5)` for wide tasks |
| **MS Paint heatmap** | `figsize=(len(qs)+2, len(imgs))` |
| **Token/time stats** | `figsize=(max(14, n*1.2), 12)` |

---

## 3. Axes Styling

Apply these to every axis:

```python
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, 120)          # 120 leaves headroom for labels + Δ annotations
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
```

- Y-axis label is always `"Accuracy (%)"` for accuracy plots.
- Y-axis range is `(0, 120)` for most plots; use `(0, 150)` only for
  contamination-validation plots where bars can reach ~100% + need Δ labels.
- Token / time plots use their own y-labels and ranges.
- **All accuracy values must be on a 0–100 scale** (multiply 0-1 fractions by 100
  before passing to bar/plot). Never mix a 0–1 bar scale with %-style text labels.

---

## 4. Titles

```
{Task Display} — {Short Description} ({model}, {mode}, {n} samples)
```

- Max ~80 characters per line. Two lines (via `\n`) are permitted.
- `fontsize=12, fontweight="bold"` for single-panel `ax.set_title`.
- `fontsize=13, fontweight="bold"` for `fig.suptitle` on multi-panel figures.
- **Do not** embed raw argparse values like `reasoning=medium`; use human-readable
  strings or omit.
- **Do not** leave unresolved `{{}}` placeholders — remove or fill all variables in
  f-strings.

---

## 5. Bar Charts

```python
ax.bar(x[i], value, width=0.55, color=color,
       edgecolor="white", linewidth=1.2, zorder=3)
```

| Property | Value |
|----------|-------|
| Width — 4 bars | `0.55` |
| Width — 5–6 bars | `0.6` |
| Edge color | `"white"` |
| Edge width | `1.2` |
| zorder | `3` (above grid) |

Value labels sit `+1.5 pp` above the bar top, `fontsize=10, fontweight="bold"`.

---

## 6. Error Bars (SD)

```python
ax.errorbar(x[i], mean * 100, yerr=sd * 100,
            fmt="none", ecolor="black", elinewidth=1.8, capsize=5, zorder=4)
```

Show a secondary `±{sd:.1f}%` text label `+4.5 pp` above the value label,
`fontsize=8.5, color="#555"`.

---

## 7. Delta (Δ) Annotations vs Baseline

```python
sign = "+" if delta >= 0 else ""
col  = "#009E73" if delta >= 0 else "#D55E00"
ax.text(x[i], bar_top + 10, f"Δ {sign}{delta*100:.1f}%",
        ha="center", va="bottom", fontsize=9, color=col, fontweight="bold")
```

- `bar_top` = `mean * 100 + (sd * 100 if sd else 0)`.
- Gap above bar_top: `10 pp` for single-panel; `5–6 pp` for multi-panel subplots.
- Skip the baseline bar itself.
- Colors are **always** `#009E73` (positive) / `#D55E00` (negative) — no exceptions.

---

## 8. Baseline Reference Line

```python
ax.axhline(baseline_mean * 100, color="#7f7f7f", linestyle="--",
           linewidth=1.2, alpha=0.6, zorder=2)
```

Draw on every plot that has a baseline condition. Placed at zorder=2 (below bars).

---

## 9. Legends

| Context | Rule |
|---------|------|
| Single-panel | `ax.legend(fontsize=9, loc="upper left", framealpha=0.85, edgecolor="#ccc")` |
| Multi-panel combined | `fig.legend(loc="lower center", ncol=N, fontsize=9, framealpha=0.85, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.06))` |
| Handle text | `.replace("\n", " ")` inside legend — no newlines in legend labels |
| No duplication | Do not repeat information already in the title |

Use `matplotlib.patches.Patch` for bar-chart legend handles.

---

## 10. Missing-Data Placeholder

When a condition has no data, draw a hatched empty bar and italicised "no data" label:

```python
ax.bar(x[i], 0, width=0.55, color=color, alpha=0.3,
       edgecolor=color, linewidth=1.2, hatch="//", zorder=3)
ax.text(x[i], 3, "no data", ha="center", va="bottom",
        fontsize=8, color="#888", fontstyle="italic")
```

---

## 11. Plot Catalog

### 11.1 SpatialEval scripts (`spatial_eval/eval_summary/`)

#### `plot_mc_results.py`
**Output:** `{task}_mc_skill_variants.png`, `all_tasks_mc_skill_variants.png`

Compares mean ± 1 SD accuracy of **4 image-skill variants** (Baseline, Img-Only,
Img+Q&A, Img+Context) across Monte Carlo subsets, for each task.

Must contain:
- 4 bars per panel, Okabe-Ito colors (grey, sky-blue, orange, teal)
- Error bars (SD) + `±SD` text above value label
- Δ vs baseline annotation + dashed baseline reference line
- Legend (upper-left single; bottom-center in combined figure)
- Title: `"{Task} — Image Skill Variants (GPT-5.2, VQA, {n} MC runs)"`

Must **not** contain:
- `{{}}` in title (remove the `× {{}}/q-type imgs` fragment)
- Non-CB delta colors (`#1a7a1a`, `#d62728`)

---

#### `plot_img_only_tool.py`
**Output:** `{task}_img_only_no_qa.png`, `all_tasks_img_only_no_qa.png`

Accuracy vs N example images for the **img-only (no Q&A)** approach (3, 10, 30 images),
all with `offset_k=30`. The tool returns images only — no answer information.

Must contain:
- 4 bars: Baseline `#555555` → n=3 `#56B4E9` → n=10 `#0072B2` → n=30 `#332288`
- Hatched placeholder for any missing condition
- Δ annotations, error bars, baseline reference line, legend
- Title must include "Images Only, No Q&A" to distinguish from preload

Must **not** contain:
- Conditions without a label explaining the N value
- "Img-Only-Tool" or bare "Img-Only" in titles without the "No Q&A" qualifier

---

#### `plot_img_only_range.py`
**Output:** `{task}_img_only_range.png`, `all_tasks_img_only_range.png`

Learning curve: accuracy as a function of N in-skill example images
(Baseline → 3 → 10 → 30 → 50 → 100).

Must contain:
- 6 bars with sequential blue palette:
  `#555555` (baseline), `#56B4E9`, `#0072B2`, `#332288`, `#084594`, `#08306b`
- Capitalized labels: `"Img-Only\n(n=X)"` (not lowercase `img-only`)
- Wider figure `(11, 5.5)` to accommodate 6 bars
- Missing `matplotlib.use("Agg")` must be added

Must **not** contain:
- Inconsistent label casing compared to other img-only scripts

---

#### `plot_preload_scaling.py`
**Output:** `{task}_img_qa_preload.png`, `all_tasks_img_qa_preload.png`

Accuracy vs N preloaded examples for the **Img+Q&A preload** architecture
(baseline, n=3, n=10, n=30). The tool returns images **with** Q&A answers.

Must contain:
- Sequential blue palette matching `plot_img_only_tool.py`
- Error bars, Δ annotations, baseline reference, legend
- Title must include "Img+Q&A Preload" to distinguish from img-only

---

#### `plot_validation_test.py`
**Output:** `{task}_img_qa_validation.png`, `all_tasks_img_qa_validation.png`

Img+Q&A contamination check: Baseline, Img+Q&A skill, Img+Q&A-val (same images —
expected ~100%), Preload-tool (same images — expected ~100%).

Must contain:
- 4 bars: grey, orange, vermillion, reddish-purple (Okabe-Ito)
- Y-axis range `(0, 150)` to show contaminated bars (~100%) with Δ annotations
- Title includes "Img+Q&A Contamination Validation"
- Subtitle clarifying which bars are contaminated (saw test images)
- `matplotlib.use("Agg")` (must be added)

Must **not** contain:
- Ambiguous bar ordering where contaminated bars appear without explanation
- "Validation Test" as the title — must specify "Img+Q&A Contamination"

---

#### `plot_image_skill_variants.py`
**Output:** `{task}_image_skill_variants.png`, `{task}_image_skill_variants_by_qtype.png`

Single-run (not MC) comparison of 4 skill variants for a specific date.
Optionally produces a per-question-type breakdown subplot.

Must contain:
- 4 bars using Okabe-Ito (grey, sky-blue, orange, teal)
- Δ annotations + baseline reference line

---

#### `plot_skills_comparison.py`
**Output:** `{task}_skills_comparison.png`

Skills vs Baseline for VQA and VTQA modes (grouped bars, 2×2). Uses CSV summaries.

Must contain:
- 2 bar groups (VQA, VTQA) with Baseline `#555555` and Skills `#56B4E9`
- Value labels on bars, Δ annotation per group, legend
- `plt.close()` after `savefig()` (must be added)

Must **not** contain:
- Non-CB colors `#5B8DB8` / `#F28C38` for the two bar types
- Non-CB delta colors `#2ca02c` / `#d62728`

---

#### `plot_skills_comparison_multi.py`
**Output:** `{task}_skills_comparison_multi.png`

Multi-round aggregation of skills vs baseline. Mean ± std across rounds, otherwise
identical layout to `plot_skills_comparison.py`.

Must contain:
- Error bars (std across rounds)
- Consistent bar colors with `plot_skills_comparison.py`

Must **not** contain:
- Non-CB delta or bar colors

---

#### `plot_results.py` (legacy — Week 3 gpt-4o / gpt-5.1 data)
**Output:** `{task}_{mode}_gpt.png`

Simple per-model bar chart from CSV summaries.

Must contain:
- Per-model colors from Okabe-Ito: gpt-4o `#56B4E9`, gpt-5.1 `#E69F00`
- Value labels, grid, no top/right spines, `dpi=150`

Must **not** contain:
- A single flat `tab:blue` applied to all bars regardless of model

---

### 11.2 MS Paint scripts (`MS_Paint_Reasoning_Evaluation/viz/`)

All scripts in this section use `MODEL_COLORS` as defined in §1.2 and must include:
- `matplotlib.use("Agg")` at module level
- `dpi=150` on every `savefig()`
- `ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)` + `ax.set_axisbelow(True)`
- `ax.spines[["top", "right"]].set_visible(False)`
- **0–100 scale** for all accuracy values (`ylim(0, 110)` or `(0, 115)`)
- `ylabel="Accuracy (%)"` (not `"Accuracy"`)

---

#### `plot_accuracy_heatmap.py`
**Output:** `model_accuracy.png`, `{model}_heatmap.png`

1. Bar chart: per-model accuracy for a specific run configuration.
2. Heatmap: per-image / per-question correctness grid for each model.

Must contain:
- Bar chart with `MODEL_COLORS`, `ylim(0, 110)`, `"Accuracy (%)"` ylabel
- Heatmap: green=correct, white=incorrect, black=missing; labelled axes

Must **not** contain:
- 0–1 scale on the bar chart (bars at height 0.65 with a "65.0%" label)
- `["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]` as bar colors

---

#### `plot_accuracy_by_blur.py`
**Output:** `accuracy_by_blur.png`

Per-model accuracy across blur levels for fixed image type and reasoning mode.
Red dotted lines separate blur-level groups.

Must contain:
- `MODEL_COLORS` — same color per model regardless of blur level
- Red dotted separator lines between groups
- Legend (`plt.legend`), grid, spines, `ylim(0, 115)`, `"Accuracy (%)"`

Must **not** contain:
- `plt.get_cmap("tab10")` — this assigns colors positionally, not by model identity
- `ylim(0, 1.15)` (old 0–1 scale)

---

#### `plot_accuracy_all_conditions.py`
**Output:** `accuracy_all_blur.png`

All models × all 3 blur levels in one chart.

Must contain:
- `MODEL_COLORS` — same color per model across all columns
- Red dotted separators, grid, `ylim(0, 115)`, `"Accuracy (%)"`

Must **not** contain:
- Hard-coded `["#1f77b4", "#ff7f0e", "#2ca02c"] * len(BLUR_LEVELS)` cycling colors
- `"Accuracy"` ylabel without `"(%)"`

---

#### `plot_accuracy_heavy_blur_high.py`
**Output:** `heavy_blur_reasoning_comparison.png`

Heavy blur: comparison between two reasoning modes for selected models.

Must contain:
- `MODEL_COLORS`, red separator between mode groups, grid, `ylim(0, 115)`

Must **not** contain:
- `{"gpt-5.1": "#ff7f0e", "gpt-5.2": "#2ca02c"}` (non-Okabe-Ito)

---

#### `plot_token_time_stats.py`
**Output:** `token_time_stats.png`

Two-panel figure: token usage (input/output/total) and elapsed time per
image/question, grouped by model.

Must contain:
- Token-type colors: Input `#56B4E9`, Output `#E69F00`, Total `#009E73`
- Cost annotations in EUR above total-token bars
- `dpi=150`, `plt.close()`

Must **not** contain:
- `["#1f77b4", "#ff7f0e", "#2ca02c"]` for token types

---

## 12. Anti-Pattern Checklist

Before committing any plot script, verify:

- [ ] `matplotlib.use("Agg")` present at module level
- [ ] `plt.close()` called after every `plt.savefig()`
- [ ] `dpi=150` on every `savefig()`
- [ ] All accuracy values on a 0–100 scale; ylabel is `"Accuracy (%)"`
- [ ] No `tab10`, `#2ca02c`, `#d62728`, `#1f77b4`, `#ff7f0e` used for semantic colors
- [ ] Delta annotations use `#009E73` (positive) and `#D55E00` (negative) **only**
- [ ] No broken f-string placeholders (`{{}}`)
- [ ] Title legible: ≤80 chars per line, no raw argparse-style parameter names
- [ ] Legend labels have no `\n` (replace with spaces inside the legend)
- [ ] Y-axis upper limit leaves ≥ 20 pp headroom above the tallest bar for labels
- [ ] Grid (`yticks`, dashed, alpha=0.4) + `set_axisbelow(True)` on every axes
- [ ] Top and right spines hidden on every axes
- [ ] Smoke-test runs excluded: MC plots use `_is_mc_file()` gate; single-run-accepting scripts add `_item_count() < MIN_STAT_ITEMS → skip`; CSV scripts use exact underscore-bounded `classify()` patterns
- [ ] N-range plots: sequential blues (sky→deep→indigo) as N grows; every N-value has its own bar and legend entry with explicit `n=X` label
