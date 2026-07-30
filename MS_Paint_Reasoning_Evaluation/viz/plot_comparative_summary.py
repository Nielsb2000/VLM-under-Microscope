"""
Compact comparative summary plots for the MS Paint Reasoning Evaluation.

This script intentionally creates a small number of high-information plots from
Results/dashboard_data/ instead of producing one plot per configuration.

Outputs are written to:
    Results/res_vis/comparative_summary/

Usage from project root:
    uv run python MS_Paint_Reasoning_Evaluation/viz/plot_comparative_summary.py

Usage from MS_Paint_Reasoning_Evaluation/evaluation:
    uv run ../viz/plot_comparative_summary.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from json_results_to_df import load_results_df

OUT_BASE = os.path.join(os.path.dirname(__file__), "..", "Results", "res_vis", "comparative_summary")

MODEL_ORDER = ["gpt-4o", "gpt-5.1", "gpt-5.2", "gpt-5.5"]
IMAGE_TYPE_ORDER = ["color", "greyscale", "inverted_greyscale"]
BLUR_ORDER = ["no_blur", "med_blur", "heavy_blur"]
REASONING_ORDER = ["low", "medium", "high"]
SKILLS_ORDER = ["no_skills", "skills"]

MODEL_COLORS = {
    "gpt-4o": "#56B4E9",
    "gpt-5.1": "#E69F00",
    "gpt-5.2": "#009E73",
    "gpt-5.5": "#CC79A7",
}
SKILL_MARKERS = {"no_skills": "o", "skills": "s"}
REASONING_MARKERS = {"low": "o", "medium": "^", "high": "D"}

FACTOR_TITLES = {
    "image_type": "Color treatment",
    "blur_level": "Gaussian blur level",
    "reasoning_mode": "Reasoning effort",
    "skills_mode": "Skill-use condition",
}

FACTOR_LABELS = {
    "image_type": {
        "color": "Color",
        "greyscale": "Grayscale",
        "inverted_greyscale": "Inverted grayscale",
    },
    "blur_level": {
        "no_blur": "No blur",
        "med_blur": "Medium blur\n(r = 5 px)",
        "heavy_blur": "Heavy blur\n(r = 15 px)",
    },
    "reasoning_mode": {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
    },
    "skills_mode": {
        "no_skills": "No skill-use",
        "skills": "Skill-use enabled",
    },
}

PANEL_LETTERS = ["A", "B", "C", "D"]

# One-column paper figure settings for the individual boxpoint panels.
# Most journals use roughly 3.3--3.6 inch single-column figures. 9 pt text
# stays readable after insertion while keeping all plot text visually consistent.
BOXPOINT_FIG_WIDTH = 3.5
BOXPOINT_FIG_HEIGHT = 4.05
BOXPOINT_FONT_SIZE = 9.0
BOXPOINT_LEGEND_FONT_SIZE = 9.0
BOXPOINT_XLABEL_FONT_SIZE = 9.0


def pretty_label(value: object, factor: str | None = None) -> str:
    """Return a publication-ready label for factor values."""
    value_str = str(value)
    if factor is not None and factor in FACTOR_LABELS:
        return FACTOR_LABELS[factor].get(value_str, value_str.replace("_", " ").title())
    for labels in FACTOR_LABELS.values():
        if value_str in labels:
            return labels[value_str]
    return value_str.replace("_", " ").title()


def compact_label(value: object, factor: str | None = None) -> str:
    """Compact labels for dense heatmaps."""
    label = pretty_label(value, factor=factor).replace("\n", " ")
    return label.replace("Gaussian ", "")


def format_delta(delta_pp: float) -> str:
    """Format an accuracy-point difference with a typographic sign."""
    sign = "+" if delta_pp > 0 else ""
    return f"{sign}{delta_pp:.1f} pp"


def paired_factor_effect_table(df: pd.DataFrame, factor: str, cats: list[str], baseline: str) -> pd.DataFrame:
    """Return paired condition-level effects relative to a baseline factor level.

    Each paired unit is Model x the other experimental factors. For example,
    when factor='blur_level', the paired unit is Model x color treatment x
    reasoning effort x skill-use condition. The response in each cell is mean
    accuracy over image-question rows.
    """
    other_factors = [
        col
        for col in ["image_type", "blur_level", "reasoning_mode", "skills_mode"]
        if col != factor
    ]
    grouped = (
        df.groupby(["Model"] + other_factors + [factor], dropna=False)["Correct"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["Model"] + other_factors,
        columns=factor,
        values="Correct",
        aggfunc="mean",
    )
    if baseline not in pivot.columns:
        return pd.DataFrame()

    rows = []
    for cat in cats:
        if cat == baseline or cat not in pivot.columns:
            continue
        paired = pivot[[baseline, cat]].dropna().reset_index()
        if paired.empty:
            continue
        paired["factor"] = factor
        paired["baseline_level"] = baseline
        paired["comparison_level"] = cat
        paired["baseline_accuracy"] = paired[baseline]
        paired["comparison_accuracy"] = paired[cat]
        paired["delta_pp"] = (paired[cat] - paired[baseline]) * 100
        rows.append(
            paired[[
                "factor",
                "baseline_level",
                "comparison_level",
                "Model",
                *other_factors,
                "baseline_accuracy",
                "comparison_accuracy",
                "delta_pp",
            ]]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def mean_model_effects(df: pd.DataFrame, factor: str, cats: list[str], baseline: str) -> list[tuple[str, float]]:
    """Average paired effects relative to the baseline with equal model weight."""
    effect_table = paired_factor_effect_table(df, factor, cats, baseline)
    if effect_table.empty:
        return []

    effects: list[tuple[str, float]] = []
    for cat, sub in effect_table.groupby("comparison_level", dropna=False):
        model_level = sub.groupby("Model", dropna=False)["delta_pp"].mean()
        if not model_level.empty:
            effects.append((str(cat), float(model_level.mean())))
    cat_order = {cat: idx for idx, cat in enumerate(cats)}
    return sorted(effects, key=lambda item: cat_order.get(item[0], 10_000))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for a family of p-values."""
    m = len(p_values)
    adjusted = [float("nan")] * m
    indexed = sorted((p, i) for i, p in enumerate(p_values) if np.isfinite(p))
    running_max = 0.0
    for rank, (p_val, original_idx) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p_val)
        running_max = max(running_max, adj)
        adjusted[original_idx] = running_max
    return adjusted


def welch_ttest_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | str]:
    """Return Welch unequal-variance t-test statistics for two samples.

    The test is applied to condition-level aggregated accuracies, i.e. the same
    kind of values shown as dots in the boxplots. It assumes that the plotted
    condition-level observations are approximately independent, that the two
    groups are measured on a continuous scale after aggregation, and that the
    sampling distribution of the group means is approximately normal. Welch's
    test does not assume equal variances. The test is not reported when either
    group has fewer than two observations or when both groups have zero sample
    variance, because the t statistic and Welch-Satterthwaite degrees of
    freedom are then not meaningful.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n_x = int(len(x))
    n_y = int(len(y))

    result: dict[str, float | int | str] = {
        "n_baseline": n_x,
        "n_comparison": n_y,
        "mean_baseline": float(np.mean(x)) if n_x else float("nan"),
        "mean_comparison": float(np.mean(y)) if n_y else float("nan"),
        "mean_difference_pp": float("nan"),
        "t_statistic": float("nan"),
        "welch_df": float("nan"),
        "p_raw": float("nan"),
        "p_holm": float("nan"),
        "test": "not reported",
        "note": "Welch test requires at least two observations per group and non-zero variance in at least one group",
    }

    if n_x == 0 or n_y == 0:
        result["note"] = "missing baseline or comparison group"
        return result

    result["mean_difference_pp"] = (float(np.mean(y)) - float(np.mean(x))) * 100
    if n_x < 2 or n_y < 2:
        return result

    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    se_sq = var_x / n_x + var_y / n_y
    if not np.isfinite(se_sq) or se_sq <= 0:
        result["note"] = "both groups have zero variance after aggregation"
        return result

    t_stat = (float(np.mean(y)) - float(np.mean(x))) / np.sqrt(se_sq)
    numerator = se_sq ** 2
    denominator = 0.0
    if n_x > 1 and var_x > 0:
        denominator += (var_x / n_x) ** 2 / (n_x - 1)
    if n_y > 1 and var_y > 0:
        denominator += (var_y / n_y) ** 2 / (n_y - 1)
    if denominator <= 0:
        result["note"] = "Welch-Satterthwaite degrees of freedom could not be computed"
        return result

    df_welch = numerator / denominator
    p_raw = float(2 * stats.t.sf(abs(t_stat), df_welch))
    result.update({
        "t_statistic": float(t_stat),
        "welch_df": float(df_welch),
        "p_raw": p_raw,
        "test": "Welch unequal-variance t-test",
        "note": "condition-level aggregated accuracies; unequal variances allowed; Holm adjusted within panel",
    })
    return result


def factor_significance_tests(df: pd.DataFrame, factor: str, cats: list[str], baseline: str) -> dict[str, dict[str, float | int | str]]:
    """Run Welch unequal-variance t-tests against the baseline level.

    The samples are condition-level aggregated accuracies, matching the values
    shown as dots in the boxplots. For example, when factor='blur_level', the
    baseline sample contains all plotted condition accuracies for No blur and
    the comparison sample contains all plotted condition accuracies for Medium
    blur or Heavy blur. Welch's test is used because equal variances across
    factor levels should not be assumed.
    """
    points = make_variant_accuracy_points(df, factor)
    if points.empty or factor not in points.columns:
        return {}
    if baseline not in set(points[factor].astype(str)):
        return {}

    baseline_vals = points[points[factor].astype(str) == str(baseline)]["accuracy"].dropna().to_numpy(dtype=float)
    raw_results = []
    for cat in [cat for cat in cats if cat != baseline]:
        comparison_vals = points[points[factor].astype(str) == str(cat)]["accuracy"].dropna().to_numpy(dtype=float)
        result = welch_ttest_summary(baseline_vals, comparison_vals)
        result["comparison_level"] = cat
        raw_results.append(result)

    adjusted = holm_adjust([float(r["p_raw"]) for r in raw_results])
    for r, p_adj in zip(raw_results, adjusted):
        r["p_holm"] = p_adj

    return {str(r["comparison_level"]): r for r in raw_results}

def format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p n/a"
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def format_df(df_value: float) -> str:
    if not np.isfinite(df_value):
        return "df n/a"
    return f"df={df_value:.1f}"


def add_effect_annotation(ax: plt.Axes, df: pd.DataFrame, factor: str, cats: list[str]) -> None:
    """Annotate a panel with average effects relative to the baseline level."""
    if not cats:
        return

    baseline = cats[0]
    effects = mean_model_effects(df, factor, cats, baseline)
    if not effects:
        return

    baseline_label = pretty_label(baseline, factor).replace("\n", " ")

    parts = []
    for cat, delta in effects:
        cat_label = pretty_label(cat, factor).replace("\n", " ")
        parts.append(f"{cat_label}: {format_delta(delta)}")

    text = "Mean effect vs " + baseline_label + "\n" + "\n".join(parts)

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "white",
            "edgecolor": "0.55",
            "alpha": 0.92,
        },
    )


def ordered_unique(values: Iterable[str], preferred: list[str]) -> list[str]:
    observed = [v for v in preferred if v in set(values)]
    observed.extend(sorted(v for v in set(values) if v not in preferred))
    return observed


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Correct", "Input Tokens", "Output Tokens", "Elapsed Time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def scenario_label(image_type: str, blur_level: str, reasoning_mode: str, skills_mode: str) -> str:
    return f"{image_type}\n{blur_level}\n{reasoning_mode}\n{skills_mode}"


def scenario_short_label(row: pd.Series) -> str:
    img = compact_label(row["image_type"], "image_type")
    blur = compact_label(row["blur_level"], "blur_level")
    reasoning = compact_label(row["reasoning_mode"], "reasoning_mode")
    skills = compact_label(row["skills_mode"], "skills_mode")
    return f"{img}\n{blur}\n{reasoning}\n{skills}"


def prepare_df(drop_empty_failed: bool) -> pd.DataFrame:
    df = load_results_df()
    if df.empty:
        return df

    df = ensure_numeric(df)
    needed = ["Model", "image_type", "blur_level", "reasoning_mode", "skills_mode", "image_num", "question_num", "Correct"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"load_results_df() is missing required columns: {missing}")

    # Remove exact duplicates while keeping the last loaded result for an image/question/model/config.
    dedup_cols = ["Model", "image_type", "blur_level", "reasoning_mode", "skills_mode", "image_num", "question_num"]
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

    if drop_empty_failed:
        metric_cols = [c for c in ["Input Tokens", "Output Tokens", "Elapsed Time"] if c in df.columns]
        if metric_cols:
            empty_metrics = df[metric_cols].isna().all(axis=1)
            likely_failed_empty_answer = (df["Correct"].fillna(0) == 0) & empty_metrics
            removed = int(likely_failed_empty_answer.sum())
            if removed:
                print(f"[INFO] Dropping {removed} likely failed/interrupted rows with no token/time metadata.")
            df = df[~likely_failed_empty_answer].copy()

    return df


def summarize_condition_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Model", "image_type", "blur_level", "reasoning_mode", "skills_mode"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            accuracy=("Correct", "mean"),
            n=("Correct", "count"),
            input_tokens=("Input Tokens", "mean") if "Input Tokens" in df.columns else ("Correct", "size"),
            output_tokens=("Output Tokens", "mean") if "Output Tokens" in df.columns else ("Correct", "size"),
            elapsed_time=("Elapsed Time", "mean") if "Elapsed Time" in df.columns else ("Correct", "size"),
        )
        .reset_index()
    )
    if "Input Tokens" in df.columns and "Output Tokens" in df.columns:
        summary["total_tokens"] = summary["input_tokens"].fillna(0) + summary["output_tokens"].fillna(0)
    else:
        summary["total_tokens"] = np.nan
    summary["scenario"] = summary.apply(scenario_short_label, axis=1)
    return summary


def plot_condition_accuracy_matrix(summary: pd.DataFrame, out_dir: str) -> None:
    models = ordered_unique(summary["Model"].dropna().astype(str), MODEL_ORDER)

    scenario_rows = (
        summary[["image_type", "blur_level", "reasoning_mode", "skills_mode", "scenario"]]
        .drop_duplicates()
        .copy()
    )
    scenario_rows["image_type"] = pd.Categorical(scenario_rows["image_type"], IMAGE_TYPE_ORDER, ordered=True)
    scenario_rows["blur_level"] = pd.Categorical(scenario_rows["blur_level"], BLUR_ORDER, ordered=True)
    scenario_rows["reasoning_mode"] = pd.Categorical(scenario_rows["reasoning_mode"], REASONING_ORDER, ordered=True)
    scenario_rows["skills_mode"] = pd.Categorical(scenario_rows["skills_mode"], SKILLS_ORDER, ordered=True)
    scenario_rows = scenario_rows.sort_values(["image_type", "blur_level", "reasoning_mode", "skills_mode"])
    scenarios = scenario_rows["scenario"].tolist()

    pivot = summary.pivot_table(index="Model", columns="scenario", values="accuracy", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=scenarios)

    fig_width = max(18, len(scenarios) * 0.42)
    fig_height = max(4.8, len(models) * 0.8 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    mat = pivot.to_numpy(dtype=float) * 100
    masked = np.ma.masked_invalid(mat)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#D9D9D9")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_title("Hand-crafted VQA diagnostic dataset: accuracy across experimental conditions", fontsize=15, fontweight="bold")
    ax.set_ylabel("Model")
    ax.set_xlabel("Color treatment × Gaussian blur level × reasoning effort × skill-use condition")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=90, fontsize=7)

    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if not np.isnan(mat[y, x]):
                ax.text(x, y, f"{mat[y, x]:.0f}", ha="center", va="center", fontsize=6, color="white" if mat[y, x] < 55 else "black")

    # Separator lines between image type groups.
    last_img = None
    for idx, (_, row) in enumerate(scenario_rows.iterrows()):
        img = row["image_type"]
        if last_img is not None and img != last_img:
            ax.axvline(idx - 0.5, color="white", linewidth=2.0)
        last_img = img

    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label("Accuracy (%)")
    fig.tight_layout()
    path = os.path.join(out_dir, "01_accuracy_matrix_all_conditions.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_factor_summary(df: pd.DataFrame, out_dir: str) -> None:
    """Plot mean accuracy per factor level and model.

    Important interpretation note: each bar is the row-level mean over all
    image/question results matching that model and factor level. Therefore the
    bar does not show how much the other experimental variants differ from each
    other. Use plot_factor_summary_boxpoints() for that internal variation.
    """
    factors = [
        ("image_type", IMAGE_TYPE_ORDER, FACTOR_TITLES["image_type"]),
        ("blur_level", BLUR_ORDER, FACTOR_TITLES["blur_level"]),
        ("reasoning_mode", REASONING_ORDER, FACTOR_TITLES["reasoning_mode"]),
        ("skills_mode", SKILLS_ORDER, FACTOR_TITLES["skills_mode"]),
    ]
    models = ordered_unique(df["Model"].dropna().astype(str), MODEL_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    for ax, (factor, order, title) in zip(axes, factors):
        sub = df.groupby(["Model", factor], dropna=False)["Correct"].mean().reset_index()
        cats = ordered_unique(sub[factor].dropna().astype(str), order)
        x = np.arange(len(cats))
        width = min(0.18, 0.8 / max(1, len(models)))
        offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
        for i, model in enumerate(models):
            vals = []
            for cat in cats:
                row = sub[(sub["Model"] == model) & (sub[factor].astype(str) == str(cat))]
                vals.append(float(row["Correct"].iloc[0]) * 100 if not row.empty else np.nan)
            ax.bar(x + offsets[i], np.nan_to_num(vals), width=width, label=model, color=MODEL_COLORS.get(model, "#777777"), alpha=0.88)
            for xi, val in zip(x + offsets[i], vals):
                if not np.isnan(val):
                    ax.text(xi, val + 1.0, f"{val:.0f}", ha="center", fontsize=7, rotation=90)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 110)
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_label(cat, factor) for cat in cats], rotation=18, ha="right")
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(title="Model", fontsize=9)
    fig.suptitle("Hand-crafted VQA diagnostic dataset: mean accuracy by experimental factor", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "02_accuracy_by_major_factors.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def make_variant_accuracy_points(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Create one point per held-out variant combination for a factor.

    Example for factor='reasoning_mode': each plotted point is the accuracy for
    one Model x reasoning_mode x image_type x blur_level x skills_mode
    combination, averaged over image/question rows. The box for 'medium' then
    summarizes variation across those variant points, instead of hiding it in a
    single mean bar.
    """
    variant_cols = ["Model", factor]
    for col in ["image_type", "blur_level", "reasoning_mode", "skills_mode"]:
        if col != factor:
            variant_cols.append(col)

    return (
        df.groupby(variant_cols, dropna=False)
        .agg(accuracy=("Correct", "mean"), n=("Correct", "count"))
        .reset_index()
    )


def add_delta_markers_for_factor(ax: plt.Axes, df: pd.DataFrame, factor: str, cats: list[str]) -> None:
    """Add mean delta markers at y=99 relative to the first factor level.

    The delta is the equal-model-weighted mean percentage-point difference
    used in the exported factor_effect_significance.csv file. Positive deltas
    are green, negative deltas are red, and zero/baseline deltas are neutral.
    """
    if not cats:
        return

    effects = dict(mean_model_effects(df, factor, cats, cats[0]))
    y_marker = 99.0

    for cat_idx, cat in enumerate(cats):
        delta = 0.0 if cat == cats[0] else effects.get(cat, np.nan)
        if not np.isfinite(delta):
            continue

        if delta > 0:
            color = "#008000"
        elif delta < 0:
            color = "#C00000"
        else:
            color = "0.35"

        ax.text(
            cat_idx,
            y_marker,
            f"Δ {format_delta(delta)}",
            ha="center",
            va="top",
            fontsize=BOXPOINT_FONT_SIZE,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.92,
            },
            zorder=6,
        )


def make_boxpoint_model_handles(models: list[str]) -> list[Patch]:
    """Legend handles for model colors in the individual boxpoint panels."""
    return [
        Patch(facecolor=MODEL_COLORS.get(m, "#777777"), alpha=0.45, label=m)
        for m in models
    ]


def make_boxpoint_encoding_handles() -> list[Line2D]:
    """Legend handles for point encoding in the individual boxpoint panels."""
    reasoning_handles = [
        Line2D(
            [0], [0],
            marker=REASONING_MARKERS.get(label, "o"),
            color="w",
            label=pretty_label(label, "reasoning_mode"),
            markerfacecolor="0.72",
            markeredgecolor="black",
            markeredgewidth=0.75,
            linestyle="None",
            markersize=6.2,
        )
        for label in REASONING_ORDER
    ]
    skill_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="No skill-use",
            markerfacecolor="0.72",
            markeredgecolor="0.65",
            markeredgewidth=0.45,
            linestyle="None",
            markersize=6.2,
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Skill-use enabled",
            markerfacecolor="0.72",
            markeredgecolor="black",
            markeredgewidth=1.45,
            linestyle="None",
            markersize=6.2,
        ),
    ]
    return reasoning_handles + skill_handles


def save_factor_boxpoint_legend(models: list[str], out_dir: str) -> None:
    """Save a standalone two-part legend for the individual boxpoint panels."""
    model_handles = make_boxpoint_model_handles(models)
    encoding_handles = make_boxpoint_encoding_handles()

    fig, ax = plt.subplots(figsize=(BOXPOINT_FIG_WIDTH, 1.55))
    ax.axis("off")
    model_legend = fig.legend(
        handles=model_handles,
        title="Model",
        fontsize=BOXPOINT_LEGEND_FONT_SIZE,
        title_fontsize=BOXPOINT_LEGEND_FONT_SIZE,
        loc="center left",
        bbox_to_anchor=(0.00, 0.50),
        ncol=1,
        frameon=True,
        handlelength=1.15,
        columnspacing=0.85,
        handletextpad=0.40,
        borderpad=0.35,
        labelspacing=0.35,
    )
    model_legend.get_title().set_ha("center")

    point_legend = fig.legend(
        handles=encoding_handles,
        title="Point encoding",
        fontsize=BOXPOINT_LEGEND_FONT_SIZE,
        title_fontsize=BOXPOINT_LEGEND_FONT_SIZE,
        loc="center right",
        bbox_to_anchor=(1.00, 0.50),
        ncol=1,
        frameon=True,
        handlelength=1.05,
        columnspacing=0.75,
        handletextpad=0.38,
        borderpad=0.35,
        labelspacing=0.30,
    )
    point_legend.get_title().set_ha("center")

    path = os.path.join(out_dir, "02b_accuracy_by_major_factors_boxpoints_legend.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def plot_factor_summary_boxpoints(df: pd.DataFrame, out_dir: str) -> None:
    """Create one boxplot-plus-points figure per major factor.

    The boxplots show the distribution of condition-level accuracies. Points are
    individual aggregated variants. Marker shape indicates reasoning effort;
    point outline indicates skill-use condition. Exact point definitions are
    also exported to boxplot_point_definitions.csv.
    """
    factors = [
        ("image_type", IMAGE_TYPE_ORDER, FACTOR_TITLES["image_type"]),
        ("blur_level", BLUR_ORDER, FACTOR_TITLES["blur_level"]),
        ("reasoning_mode", REASONING_ORDER, FACTOR_TITLES["reasoning_mode"]),
        ("skills_mode", SKILLS_ORDER, FACTOR_TITLES["skills_mode"]),
    ]
    models = ordered_unique(df["Model"].dropna().astype(str), MODEL_ORDER)

    rng = np.random.default_rng(7)
    for factor_idx, (factor, order, title) in enumerate(factors):
        points = make_variant_accuracy_points(df, factor)
        cats = ordered_unique(points[factor].dropna().astype(str), order)
        if not cats:
            continue

        x = np.arange(len(cats))
        width = min(0.16, 0.78 / max(1, len(models)))
        offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

        fig, ax = plt.subplots(figsize=(BOXPOINT_FIG_WIDTH, BOXPOINT_FIG_HEIGHT))

        for i, model in enumerate(models):
            model_color = MODEL_COLORS.get(model, "#777777")
            positions = x + offsets[i]
            data = []
            for cat in cats:
                vals = points[(points["Model"] == model) & (points[factor].astype(str) == str(cat))]["accuracy"].dropna().to_numpy() * 100
                data.append(vals)

            non_empty_positions = [pos for pos, vals in zip(positions, data) if len(vals) > 0]
            non_empty_data = [vals for vals in data if len(vals) > 0]
            if non_empty_data:
                bp = ax.boxplot(
                    non_empty_data,
                    positions=non_empty_positions,
                    widths=width * 0.78,
                    patch_artist=True,
                    showfliers=False,
                    # Hide the boxplot median line so the only horizontal black
                    # bar in each box is the explicit condition mean marker below.
                    medianprops={"color": "none", "linewidth": 0.0},
                    boxprops={"facecolor": model_color, "alpha": 0.30, "edgecolor": model_color, "linewidth": 1.0},
                    whiskerprops={"color": model_color, "linewidth": 1.0},
                    capprops={"color": model_color, "linewidth": 1.0},
                )
                _ = bp

            for cat_idx, cat in enumerate(cats):
                sub = points[(points["Model"] == model) & (points[factor].astype(str) == str(cat))].copy()
                if sub.empty:
                    continue
                if "skills_mode" not in sub.columns:
                    sub["skills_mode"] = str(cat)
                if "reasoning_mode" not in sub.columns:
                    sub["reasoning_mode"] = str(cat)

                for reasoning_value, marker in REASONING_MARKERS.items():
                    reasoning_sub = sub[sub["reasoning_mode"].astype(str) == reasoning_value]
                    if reasoning_sub.empty:
                        continue
                    for skill_value, skill_sub in reasoning_sub.groupby("skills_mode", dropna=False):
                        if skill_sub.empty:
                            continue
                        skill_value_str = str(skill_value)
                        skill_enabled = skill_value_str == "skills"
                        jitter = rng.uniform(-width * 0.24, width * 0.24, size=len(skill_sub))
                        ax.scatter(
                            np.full(len(skill_sub), positions[cat_idx]) + jitter,
                            skill_sub["accuracy"].to_numpy() * 100,
                            marker=marker,
                            s=20 if skill_enabled else 17,
                            color=model_color,
                            edgecolor="black" if skill_enabled else "0.65",
                            linewidth=0.65 if skill_enabled else 0.25,
                            alpha=0.80,
                            zorder=3,
                        )

            means = [np.nanmean(vals) if len(vals) else np.nan for vals in data]
            for pos, mean_val in zip(positions, means):
                if not np.isnan(mean_val):
                    ax.scatter(pos, mean_val, marker="_", s=135, color="black", linewidth=1.35, zorder=4)

        ax.set_ylabel("Accuracy (%)", fontsize=BOXPOINT_FONT_SIZE, labelpad=5)
        # Keep the top of the plot close to 100%; delta labels are fixed at y=99.
        ax.set_ylim(40, 104)
        ax.set_xticks(x)
        x_tick_labels = [compact_label(cat, factor) for cat in cats]
        if factor == "blur_level":
            x_tick_labels = [
                label.replace("Medium blur", "Med. blur").replace("Heavy blur", "Heavy blur")
                for label in x_tick_labels
            ]
        ax.set_xticklabels(
            x_tick_labels,
            rotation=18 if len(cats) <= 3 else 25,
            ha="right",
            fontsize=BOXPOINT_XLABEL_FONT_SIZE,
        )
        ax.tick_params(axis="y", labelsize=BOXPOINT_FONT_SIZE)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        add_delta_markers_for_factor(ax, df, factor, cats)

        # Put the model legend on the left and the point-encoding legend on the right.
        # Both legends sit below the x-axis labels so the one-column plot stays readable.
        model_handles = make_boxpoint_model_handles(models)
        encoding_handles = make_boxpoint_encoding_handles()
        model_legend = ax.legend(
            handles=model_handles,
            title="Model",
            fontsize=BOXPOINT_LEGEND_FONT_SIZE,
            title_fontsize=BOXPOINT_LEGEND_FONT_SIZE,
            loc="upper left",
            bbox_to_anchor=(-0.02, -0.39),
            ncol=1,
            frameon=True,
            handlelength=1.15,
            columnspacing=0.85,
            handletextpad=0.40,
            borderpad=0.35,
            labelspacing=0.35,
        )
        model_legend.get_title().set_ha("center")
        ax.add_artist(model_legend)

        point_legend = ax.legend(
            handles=encoding_handles,
            title="Point encoding",
            fontsize=BOXPOINT_LEGEND_FONT_SIZE,
            title_fontsize=BOXPOINT_LEGEND_FONT_SIZE,
            loc="upper right",
            bbox_to_anchor=(1.02, -0.39),
            ncol=1,
            frameon=True,
            handlelength=1.05,
            columnspacing=0.75,
            handletextpad=0.38,
            borderpad=0.35,
            labelspacing=0.30,
        )
        point_legend.get_title().set_ha("center")

        fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.52)
        path = os.path.join(out_dir, f"02b_accuracy_by_major_factors_boxpoints_{factor}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    save_factor_boxpoint_legend(models, out_dir)

def plot_skills_delta_heatmap(summary: pd.DataFrame, out_dir: str) -> None:
    key = ["Model", "image_type", "blur_level", "reasoning_mode"]
    p = summary.pivot_table(index=key, columns="skills_mode", values="accuracy", aggfunc="mean").reset_index()
    if "skills" not in p.columns or "no_skills" not in p.columns:
        print("[WARN] Skipping skills delta heatmap: both skills and no_skills are required.")
        return
    p["delta"] = (p["skills"] - p["no_skills"]) * 100
    p["scenario"] = p.apply(lambda r: f"{compact_label(r['image_type'], 'image_type')}\n{compact_label(r['blur_level'], 'blur_level')}\n{compact_label(r['reasoning_mode'], 'reasoning_mode')}", axis=1)

    models = ordered_unique(p["Model"].dropna().astype(str), MODEL_ORDER)
    scenario_rows = p[["image_type", "blur_level", "reasoning_mode", "scenario"]].drop_duplicates().copy()
    scenario_rows["image_type"] = pd.Categorical(scenario_rows["image_type"], IMAGE_TYPE_ORDER, ordered=True)
    scenario_rows["blur_level"] = pd.Categorical(scenario_rows["blur_level"], BLUR_ORDER, ordered=True)
    scenario_rows["reasoning_mode"] = pd.Categorical(scenario_rows["reasoning_mode"], REASONING_ORDER, ordered=True)
    scenario_rows = scenario_rows.sort_values(["image_type", "blur_level", "reasoning_mode"])
    scenarios = scenario_rows["scenario"].tolist()

    pivot = p.pivot_table(index="Model", columns="scenario", values="delta", aggfunc="mean").reindex(index=models, columns=scenarios)
    vmax = max(5, np.nanmax(np.abs(pivot.to_numpy(dtype=float))) if not pivot.empty else 5)
    vmax = min(max(vmax, 10), 60)

    fig, ax = plt.subplots(figsize=(max(14, len(scenarios) * 0.5), max(4.5, len(models) * 0.8 + 2)))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#D9D9D9")
    mat = pivot.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title("Skill-use effect: Skill-use enabled - No skill-use", fontsize=15, fontweight="bold")
    ax.set_ylabel("Model")
    ax.set_xlabel("Image type × blur × reasoning effort")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=90, fontsize=8)
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if not np.isnan(mat[y, x]):
                ax.text(x, y, f"{mat[y, x]:+.0f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Accuracy-point effect of enabling skill-use")
    fig.tight_layout()
    path = os.path.join(out_dir, "03_skills_lift_heatmap.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_reasoning_delta_heatmap(summary: pd.DataFrame, out_dir: str) -> None:
    # Reasoning effort is meaningful for GPT-5.x models. Keep gpt-4o out of the high-low comparison.
    s = summary[summary["Model"] != "gpt-4o"].copy()
    key = ["Model", "image_type", "blur_level", "skills_mode"]
    p = s.pivot_table(index=key, columns="reasoning_mode", values="accuracy", aggfunc="mean").reset_index()
    if "high" not in p.columns or "low" not in p.columns:
        print("[WARN] Skipping reasoning delta heatmap: both low and high reasoning are required.")
        return
    p["delta"] = (p["high"] - p["low"]) * 100
    p["scenario"] = p.apply(lambda r: f"{compact_label(r['image_type'], 'image_type')}\n{compact_label(r['blur_level'], 'blur_level')}\n{compact_label(r['skills_mode'], 'skills_mode')}", axis=1)

    models = ordered_unique(p["Model"].dropna().astype(str), MODEL_ORDER)
    scenario_rows = p[["image_type", "blur_level", "skills_mode", "scenario"]].drop_duplicates().copy()
    scenario_rows["image_type"] = pd.Categorical(scenario_rows["image_type"], IMAGE_TYPE_ORDER, ordered=True)
    scenario_rows["blur_level"] = pd.Categorical(scenario_rows["blur_level"], BLUR_ORDER, ordered=True)
    scenario_rows["skills_mode"] = pd.Categorical(scenario_rows["skills_mode"], SKILLS_ORDER, ordered=True)
    scenario_rows = scenario_rows.sort_values(["image_type", "blur_level", "skills_mode"])
    scenarios = scenario_rows["scenario"].tolist()

    pivot = p.pivot_table(index="Model", columns="scenario", values="delta", aggfunc="mean").reindex(index=models, columns=scenarios)
    vmax = max(5, np.nanmax(np.abs(pivot.to_numpy(dtype=float))) if not pivot.empty else 5)
    vmax = min(max(vmax, 10), 60)

    fig, ax = plt.subplots(figsize=(max(14, len(scenarios) * 0.55), max(4.5, len(models) * 0.8 + 2)))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#D9D9D9")
    mat = pivot.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title("Reasoning-effort effect: High − Low", fontsize=15, fontweight="bold")
    ax.set_ylabel("Model")
    ax.set_xlabel("Image type × blur × skills mode")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=90, fontsize=8)
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if not np.isnan(mat[y, x]):
                ax.text(x, y, f"{mat[y, x]:+.0f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Accuracy-point effect of high reasoning")
    fig.tight_layout()
    path = os.path.join(out_dir, "04_reasoning_high_minus_low_heatmap.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_efficiency_frontier(summary: pd.DataFrame, out_dir: str) -> None:
    """Plot accuracy against average token use.

    This panel is intended as a compact publication-style efficiency summary:
    color encodes model, marker shape encodes reasoning effort, and point
    outline encodes skill-use. Latency is intentionally not encoded, so the
    panel focuses only on the accuracy--token-use trade-off.
    """
    needed = {"total_tokens", "accuracy"}
    if not needed.issubset(summary.columns):
        print("[WARN] Skipping efficiency frontier: token columns are missing.")
        return

    eff = (
        summary.dropna(subset=["total_tokens", "accuracy"])
        .groupby(["Model", "reasoning_mode", "skills_mode"], dropna=False)
        .agg(
            accuracy=("accuracy", "mean"),
            total_tokens=("total_tokens", "mean"),
            n=("n", "sum"),
        )
        .reset_index()
    )
    if eff.empty:
        print("[WARN] Skipping efficiency frontier: no rows with token metadata.")
        return

    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    point_size = 78

    # Add a light connecting line per model and skill-use condition to reveal
    # the low/medium/high reasoning trajectory without adding text labels.
    reasoning_rank = {name: idx for idx, name in enumerate(REASONING_ORDER)}
    for (model, skills), sub in eff.groupby(["Model", "skills_mode"], dropna=False):
        sub = sub.copy()
        sub["reasoning_rank"] = sub["reasoning_mode"].map(reasoning_rank)
        sub = sub.sort_values("reasoning_rank")
        if len(sub) < 2:
            continue
        ax.plot(
            sub["total_tokens"],
            sub["accuracy"] * 100,
            color=MODEL_COLORS.get(str(model), "#777777"),
            linewidth=0.9,
            alpha=0.28 if str(skills) == "no_skills" else 0.45,
            zorder=2,
        )

    for _, row in eff.iterrows():
        model = str(row["Model"])
        reasoning = str(row["reasoning_mode"])
        skills = str(row["skills_mode"])
        skill_enabled = skills == "skills"
        ax.scatter(
            row["total_tokens"],
            row["accuracy"] * 100,
            s=point_size,
            color=MODEL_COLORS.get(model, "#777777"),
            marker=REASONING_MARKERS.get(reasoning, "o"),
            alpha=0.82,
            edgecolor="black" if skill_enabled else "0.65",
            linewidth=1.0 if skill_enabled else 0.40,
            zorder=3,
        )

    ax.set_title("Accuracy--token-use trade-off", fontsize=15, fontweight="bold")
    ax.set_xlabel("Mean total tokens per answer")
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xscale("linear")

    token_min = float(eff["total_tokens"].min())
    token_max = float(eff["total_tokens"].max())
    if np.isfinite(token_min) and np.isfinite(token_max) and token_max > token_min:
        pad = 0.06 * (token_max - token_min)
        ax.set_xlim(max(0, token_min - pad), token_max + pad)

    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.28)
    ax.xaxis.grid(True, linestyle="--", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    model_handles = [
        Patch(facecolor=MODEL_COLORS.get(m, "#777777"), label=m)
        for m in ordered_unique(eff["Model"].astype(str), MODEL_ORDER)
    ]
    reasoning_handles = [
        Line2D([0], [0], marker=marker, color="w", label=pretty_label(label, "reasoning_mode"), markerfacecolor="gray", markeredgecolor="black", markersize=8)
        for label, marker in REASONING_MARKERS.items()
        if label in set(eff["reasoning_mode"].astype(str))
    ]
    skill_handles = [
        Line2D([0], [0], marker="o", color="w", label="No skill-use", markerfacecolor="gray", markeredgecolor="0.65", markeredgewidth=0.4, markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Skill-use enabled", markerfacecolor="gray", markeredgecolor="black", markeredgewidth=1.2, markersize=8),
    ]

    legend1 = ax.legend(handles=model_handles, title="Model", loc="upper left", bbox_to_anchor=(1.02, 1.00), fontsize=8.5, frameon=True)
    ax.add_artist(legend1)
    ax.legend(handles=reasoning_handles + skill_handles, title="Point encoding", loc="center left", bbox_to_anchor=(1.02, 0.50), fontsize=8.5, frameon=True)

    fig.tight_layout(rect=[0, 0, 0.80, 1])
    path = os.path.join(out_dir, "05_accuracy_vs_tokens.png")
    fig.savefig(path, dpi=180)
    legacy_path = os.path.join(out_dir, "05_accuracy_vs_tokens_latency.png")
    fig.savefig(legacy_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"Saved: {legacy_path}")


def plot_coverage_matrix(summary: pd.DataFrame, out_dir: str) -> None:
    models = ordered_unique(summary["Model"].dropna().astype(str), MODEL_ORDER)
    scenario_rows = summary[["image_type", "blur_level", "reasoning_mode", "skills_mode", "scenario"]].drop_duplicates().copy()
    scenario_rows["image_type"] = pd.Categorical(scenario_rows["image_type"], IMAGE_TYPE_ORDER, ordered=True)
    scenario_rows["blur_level"] = pd.Categorical(scenario_rows["blur_level"], BLUR_ORDER, ordered=True)
    scenario_rows["reasoning_mode"] = pd.Categorical(scenario_rows["reasoning_mode"], REASONING_ORDER, ordered=True)
    scenario_rows["skills_mode"] = pd.Categorical(scenario_rows["skills_mode"], SKILLS_ORDER, ordered=True)
    scenario_rows = scenario_rows.sort_values(["image_type", "blur_level", "reasoning_mode", "skills_mode"])
    scenarios = scenario_rows["scenario"].tolist()

    pivot = summary.pivot_table(index="Model", columns="scenario", values="n", aggfunc="sum").reindex(index=models, columns=scenarios)
    fig, ax = plt.subplots(figsize=(max(18, len(scenarios) * 0.42), max(4.8, len(models) * 0.8 + 2.5)))
    mat = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(mat) if not np.isnan(mat).all() else 1
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#D9D9D9")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title("Data coverage: image-question rows per condition", fontsize=15, fontweight="bold")
    ax.set_ylabel("Model")
    ax.set_xlabel("Color treatment × Gaussian blur level × reasoning effort × skill-use condition")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=90, fontsize=7)
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if not np.isnan(mat[y, x]):
                ax.text(x, y, f"{int(mat[y, x])}", ha="center", va="center", fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label("Rows")
    fig.tight_layout()
    path = os.path.join(out_dir, "06_data_coverage_matrix.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def save_boxplot_point_definitions(df: pd.DataFrame, out_dir: str) -> None:
    """Export the exact aggregated points used in the boxpoint panels."""
    factors = [
        ("image_type", IMAGE_TYPE_ORDER, FACTOR_TITLES["image_type"]),
        ("blur_level", BLUR_ORDER, FACTOR_TITLES["blur_level"]),
        ("reasoning_mode", REASONING_ORDER, FACTOR_TITLES["reasoning_mode"]),
        ("skills_mode", SKILLS_ORDER, FACTOR_TITLES["skills_mode"]),
    ]
    rows = []
    for factor, order, title in factors:
        points = make_variant_accuracy_points(df, factor).copy()
        if points.empty:
            continue
        points["panel_factor"] = factor
        points["panel_title"] = title
        points["x_level"] = points[factor]
        points["x_label"] = points[factor].map(lambda value: pretty_label(value, factor).replace("\n", " "))
        points["accuracy_percent"] = points["accuracy"] * 100
        for col in ["image_type", "blur_level", "reasoning_mode", "skills_mode"]:
            if col not in points.columns:
                points[col] = points["x_level"]
            points[col + "_label"] = points[col].map(lambda value, c=col: pretty_label(value, c).replace("\n", " "))
        rows.append(points)

    if not rows:
        return
    out = pd.concat(rows, ignore_index=True)
    cols = [
        "panel_factor",
        "panel_title",
        "Model",
        "x_level",
        "x_label",
        "image_type_label",
        "blur_level_label",
        "reasoning_mode_label",
        "skills_mode_label",
        "accuracy_percent",
        "n",
    ]
    path = os.path.join(out_dir, "boxplot_point_definitions.csv")
    out[cols].to_csv(path, index=False)
    print(f"Saved: {path}")


def save_factor_significance_summary(df: pd.DataFrame, out_dir: str) -> None:
    """Export effect estimates and Welch-test results used in annotations."""
    factors = [
        ("image_type", IMAGE_TYPE_ORDER, FACTOR_TITLES["image_type"]),
        ("blur_level", BLUR_ORDER, FACTOR_TITLES["blur_level"]),
        ("reasoning_mode", REASONING_ORDER, FACTOR_TITLES["reasoning_mode"]),
        ("skills_mode", SKILLS_ORDER, FACTOR_TITLES["skills_mode"]),
    ]
    rows = []
    for factor, order, title in factors:
        cats = ordered_unique(df[factor].dropna().astype(str), order)
        if not cats:
            continue
        baseline = cats[0]
        effects = dict(mean_model_effects(df, factor, cats, baseline))
        tests = factor_significance_tests(df, factor, cats, baseline)
        for cat in cats:
            if cat == baseline:
                continue
            test = tests.get(str(cat), {})
            rows.append({
                "factor": factor,
                "factor_title": title,
                "baseline_level": baseline,
                "baseline_label": pretty_label(baseline, factor).replace("\n", " "),
                "comparison_level": cat,
                "comparison_label": pretty_label(cat, factor).replace("\n", " "),
                "mean_effect_pp_equal_model_weight": effects.get(cat, np.nan),
                "welch_mean_difference_pp": test.get("mean_difference_pp", np.nan),
                "test": test.get("test", "not reported"),
                "t_statistic": test.get("t_statistic", np.nan),
                "welch_df": test.get("welch_df", np.nan),
                "p_raw": test.get("p_raw", np.nan),
                "p_holm_within_panel": test.get("p_holm", np.nan),
                "n_baseline": test.get("n_baseline", 0),
                "n_comparison": test.get("n_comparison", 0),
                "condition_check_note": test.get("note", "no Welch test result"),
                "test_unit": "condition-level aggregated accuracy points used in the boxplot",
                "assumptions": "independent condition-level observations; approximately normal sampling distribution of group means after aggregation; unequal variances allowed by Welch test",
            })
    if not rows:
        return
    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "factor_effect_significance.csv")
    out.to_csv(path, index=False)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact comparative summary plots for MS Paint results.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: Results/res_vis/comparative_summary")
    parser.add_argument(
        "--include-empty-failed",
        action="store_true",
        help="Include rows that look like failed/interrupted answer generations: Correct=0 with no token/time metadata.",
    )
    args = parser.parse_args()

    out_dir = os.path.normpath(args.output_dir or OUT_BASE)
    os.makedirs(out_dir, exist_ok=True)

    df = prepare_df(drop_empty_failed=not args.include_empty_failed)
    if df.empty:
        print("No data found in Results/dashboard_data/.")
        return

    summary = summarize_condition_accuracy(df)
    summary_path = os.path.join(out_dir, "condition_accuracy_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    factor_summary = (
        df.groupby(["Model", "image_type", "blur_level", "reasoning_mode", "skills_mode"], dropna=False)
        .agg(accuracy=("Correct", "mean"), n=("Correct", "count"))
        .reset_index()
    )
    factor_path = os.path.join(out_dir, "condition_accuracy_counts.csv")
    factor_summary.to_csv(factor_path, index=False)
    print(f"Saved: {factor_path}")

    save_boxplot_point_definitions(df, out_dir)
    save_factor_significance_summary(df, out_dir)

    plot_condition_accuracy_matrix(summary, out_dir)
    plot_factor_summary(df, out_dir)
    plot_factor_summary_boxpoints(df, out_dir)
    plot_skills_delta_heatmap(summary, out_dir)
    plot_reasoning_delta_heatmap(summary, out_dir)
    plot_efficiency_frontier(summary, out_dir)
    plot_coverage_matrix(summary, out_dir)

    print(f"\n[INFO] Comparative summary plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
