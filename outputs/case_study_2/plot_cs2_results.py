"""
Scientific comparative plots for Case Study 2 SEM counting results.

Purpose
-------
This script creates compact, publication-style visual summaries focused on:
  1. Comparing the seven CS2 experimental variants.
  2. Checking whether the larger Grid Scan Paper set aligns with the two
     experimental validation sets.
  3. Producing one clear overall-effect figure using existing metrics only.

Important: the overall figure ranks variants by MAE only. No composite score is
created. Lower MAE is better.

Reads existing CS2 run manifests from:
    outputs/case_study_2/runs/

Writes cleaned data and figures to:
    outputs/case_study_2/
    outputs/figures/case_study_2/

Usage from project root:
    uv run python outputs/case_study_2/plot_cs2_results.py

Usage also works when launched from outputs/case_study_2/ because paths are
resolved relative to this file location.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Publication-oriented plotting defaults.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 9,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "patch.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

CM_TO_INCH = 1.0 / 2.54
SINGLE_COL_WIDTH_CM = 8.9
SINGLE_COL_WIDTH_IN = SINGLE_COL_WIDTH_CM * CM_TO_INCH


def single_col_figsize(height_cm: float) -> tuple[float, float]:
    """Return a column-friendly figure size in inches."""
    return (SINGLE_COL_WIDTH_IN, height_cm * CM_TO_INCH)

# Robust when this file is saved as outputs/case_study_2/plot_cs2_results.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_OUT = PROJECT_ROOT / "outputs" / "case_study_2"
FIG_OUT = PROJECT_ROOT / "outputs" / "figures" / "case_study_2"
RUNS = DATA_OUT / "runs"
FIG_DIR = FIG_OUT
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "grid_scan_paper",
    "validation_unc_luca",
    "validation_andrea_grid_scan_mirror",
]

LARGE_DATASET = "grid_scan_paper"
EXPERIMENTAL_DATASETS = [
    "validation_unc_luca",
    "validation_andrea_grid_scan_mirror",
]

VARIANTS = [
    "E1_raw_vlm",
    "E2_sam2_deterministic",
    "E3_imggen_deterministic",
    "E4_sam2_overlay_vlm",
    "E5_imggen_overlay_vlm",
    "E6_sam3_deterministic",
    "E7_sam3_overlay_vlm",
]

VARIANT_LABELS = {
    "E1_raw_vlm": "VLM",
    "E2_sam2_deterministic": "SAM2 mask",
    "E3_imggen_deterministic": "ImgGen mask",
    "E4_sam2_overlay_vlm": "SAM2 overlay",
    "E5_imggen_overlay_vlm": "ImgGen overlay",
    "E6_sam3_deterministic": "SAM3 mask",
    "E7_sam3_overlay_vlm": "SAM3 overlay",
}

DATASET_LABELS = {
    "grid_scan_paper": "Grid Scan Paper",
    "validation_unc_luca": "Rettenberger Uncertainty",
    "validation_andrea_grid_scan_mirror": "TFS Mirror",
}

# Colorblind-friendly palette, comparable in spirit to the MS Paint summary.
VARIANT_COLORS = {
    # Baseline
    "E1_raw_vlm": "#999999",              # grey

    # SAM2: light blue -> dark blue
    "E2_sam2_deterministic": "#56B4E9",   # light blue
    "E4_sam2_overlay_vlm": "#0072B2",     # dark blue

    # SAM3: light green -> dark green
    "E6_sam3_deterministic": "#66C2A5",   # light green
    "E7_sam3_overlay_vlm": "#009E73",     # dark green

    # ImgGen: light orange -> dark orange
    "E3_imggen_deterministic": "#E69F00", # light orange
    "E5_imggen_overlay_vlm": "#D55E00",   # dark orange
}

DATASET_MARKERS = {
    "grid_scan_paper": "o",
    "validation_unc_luca": "s",
    "validation_andrea_grid_scan_mirror": "D",
}

PANEL_LETTERS = ["A", "B", "C", "D"]

# For boxplots, keep the two grid-like datasets adjacent and put the harder Rettenberger Uncertainty set last.
BOXPLOT_DATASET_ORDER = [
    "grid_scan_paper",
    "validation_andrea_grid_scan_mirror",
    "validation_unc_luca",
]

# Presentation order for method-comparison plots:
# baseline first, then SAM2/SAM3 mask-overlay pairs, then ImgGen mask-overlay.
METHOD_COMPARISON_ORDER = [
    "E1_raw_vlm",
    "E2_sam2_deterministic",
    "E4_sam2_overlay_vlm",
    "E6_sam3_deterministic",
    "E7_sam3_overlay_vlm",
    "E3_imggen_deterministic",
    "E5_imggen_overlay_vlm",
]


def pretty_variant(variant: str) -> str:
    return VARIANT_LABELS.get(str(variant), str(variant).replace("_", " ").title())


def pretty_dataset(dataset: str) -> str:
    return DATASET_LABELS.get(str(dataset), str(dataset).replace("_", " ").title())


def safe_name(label: object) -> str:
    return str(label).lower().replace(" ", "_").replace("/", "_")


def dataset_name(manifest: dict) -> str:
    src = manifest.get("sample_source") or {}
    return src.get("dataset_name") or f"{src.get('source', 'unknown')}:{src.get('category', 'unknown')}"


def ordered_unique(values: Iterable[str], preferred: list[str]) -> list[str]:
    observed = [v for v in preferred if v in set(values)]
    observed.extend(sorted(v for v in set(values) if v not in preferred))
    return observed


def load_clean_latest_results() -> pd.DataFrame:
    """Load latest completed comparable result per dataset/sample/variant."""
    best: dict[tuple[str, str, str], dict] = {}

    for mf in sorted(RUNS.glob("*/*/run_manifest.json")):
        try:
            manifest = json.loads(mf.read_text())
        except Exception:
            continue

        ds = dataset_name(manifest)
        if ds not in DATASETS:
            continue

        sample = manifest.get("sample_id")
        started = manifest.get("started_at") or mf.parent.name
        run_dir = mf.parent
        per_variant = manifest.get("metrics_summary", {}).get("per_variant", {})

        for variant in VARIANTS:
            metrics = per_variant.get(variant)
            if not metrics or not metrics.get("comparable"):
                continue

            pred_path = run_dir / "predictions" / f"{variant}.json"
            if not pred_path.exists():
                continue

            try:
                pred = json.loads(pred_path.read_text())
            except Exception:
                continue

            if pred.get("status") != "completed":
                continue

            gt_interval = metrics.get("gt_interval") or [None, None]
            key = (ds, sample, variant)
            old = best.get(key)
            if old is None or started > old["started"]:
                best[key] = {
                    "dataset": ds,
                    "dataset_label": pretty_dataset(ds),
                    "sample_id": sample,
                    "variant": variant,
                    "variant_label": pretty_variant(variant),
                    "started": started,
                    "run_dir": str(run_dir),
                    "predicted_count": metrics.get("predicted_count"),
                    "gt_count": metrics.get("gt_count"),
                    "gt_count_mode": metrics.get("gt_count_mode"),
                    "gt_interval_low": gt_interval[0],
                    "gt_interval_high": gt_interval[1],
                    "absolute_error": metrics.get("absolute_error"),
                    "relative_error": metrics.get("relative_error"),
                    "exact_match": bool(metrics.get("exact_match")),
                    "inside_gt_interval": bool(metrics.get("inside_gt_interval")),
                    "within_1": bool(metrics.get("within_1")),
                    "within_2": bool(metrics.get("within_2")),
                    "within_3": bool(metrics.get("within_3")),
                    "within_5_pct": bool(metrics.get("within_5_pct")),
                    "within_10_pct": bool(metrics.get("within_10_pct")),
                }

    df = pd.DataFrame(best.values())
    if df.empty:
        raise SystemExit(f"No completed comparable CS2 results found under {RUNS}")

    df["variant"] = pd.Categorical(df["variant"], categories=VARIANTS, ordered=True)
    df["variant_label"] = pd.Categorical(
        df["variant_label"], categories=[VARIANT_LABELS[v] for v in VARIANTS], ordered=True
    )
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASETS, ordered=True)
    df["dataset_label"] = pd.Categorical(
        df["dataset_label"], categories=[DATASET_LABELS[d] for d in DATASETS], ordered=True
    )
    numeric_cols = ["absolute_error", "relative_error", "predicted_count", "gt_count", "gt_interval_low", "gt_interval_high"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["gt_value"] = np.where(
        df["gt_interval_low"].notna() & df["gt_interval_high"].notna(),
        (df["gt_interval_low"] + df["gt_interval_high"]) / 2.0,
        df["gt_count"],
    )
    df["normalized_absolute_error_pct"] = np.where(
        df["gt_value"].notna() & (df["gt_value"] > 0),
        (df["absolute_error"] / df["gt_value"]) * 100.0,
        np.nan,
    )
    df["signed_error"] = df["predicted_count"] - df["gt_count"]
    return df


def summarize_by_dataset_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize only the existing CS2 metrics."""
    summary = (
        df.groupby(["dataset", "dataset_label", "variant", "variant_label"], observed=True)
        .agg(
            n=("sample_id", "count"),
            mae=("absolute_error", "mean"),
            median_abs_error=("absolute_error", "median"),
            exact_pct=("exact_match", lambda x: x.mean() * 100),
            inside_interval_pct=("inside_gt_interval", lambda x: x.mean() * 100),
            within_1_pct=("within_1", lambda x: x.mean() * 100),
            within_2_pct=("within_2", lambda x: x.mean() * 100),
            within_3_pct=("within_3", lambda x: x.mean() * 100),
            within_10_pct=("within_10_pct", lambda x: x.mean() * 100),
            normalized_mae_pct=("normalized_absolute_error_pct", "mean"),
            mean_signed_error=("signed_error", "mean"),
            median_signed_error=("signed_error", "median"),
        )
        .reset_index()
    )
    summary["variant"] = pd.Categorical(summary["variant"], categories=VARIANTS, ordered=True)
    summary["dataset"] = pd.Categorical(summary["dataset"], categories=DATASETS, ordered=True)
    return summary.sort_values(["dataset", "variant"]).reset_index(drop=True)


def compute_overall_mae_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank variants by plain MAE averaged equally across datasets."""
    overall = (
        summary.groupby(["variant", "variant_label"], observed=True)
        .agg(
            equal_dataset_mae=("mae", "mean"),
            sample_weighted_mae=("mae", lambda _: np.nan),
            mean_exact_pct=("exact_pct", "mean"),
            mean_within_1_pct=("within_1_pct", "mean"),
            mean_within_10_pct=("within_10_pct", "mean"),
            n_datasets=("dataset", "nunique"),
        )
        .reset_index()
    )
    # Add sample-weighted MAE from dataset-level summary using n weights.
    weighted_rows = []
    for variant, sub in summary.groupby("variant", observed=True):
        weights = sub["n"].to_numpy(dtype=float)
        vals = sub["mae"].to_numpy(dtype=float)
        weighted = float(np.average(vals, weights=weights)) if np.isfinite(vals).any() else np.nan
        weighted_rows.append({"variant": variant, "sample_weighted_mae": weighted})
    weighted_df = pd.DataFrame(weighted_rows)
    overall = overall.drop(columns=["sample_weighted_mae"]).merge(weighted_df, on="variant", how="left")
    overall["rank"] = overall["equal_dataset_mae"].rank(ascending=True, method="min").astype(int)
    overall["variant"] = pd.Categorical(overall["variant"], categories=VARIANTS, ordered=True)
    return overall.sort_values(["rank", "variant"]).reset_index(drop=True)


def compute_overall_normalized_mae_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank variants by normalized MAE averaged equally across datasets."""
    overall = (
        summary.groupby(["variant", "variant_label"], observed=True)
        .agg(
            equal_dataset_normalized_mae=("normalized_mae_pct", "mean"),
            sample_weighted_normalized_mae=("normalized_mae_pct", lambda _: np.nan),
            mean_exact_pct=("exact_pct", "mean"),
            mean_within_1_pct=("within_1_pct", "mean"),
            mean_within_10_pct=("within_10_pct", "mean"),
            n_datasets=("dataset", "nunique"),
        )
        .reset_index()
    )
    weighted_rows = []
    for variant, sub in summary.groupby("variant", observed=True):
        weights = sub["n"].to_numpy(dtype=float)
        vals = sub["normalized_mae_pct"].to_numpy(dtype=float)
        weighted = float(np.average(vals, weights=weights)) if np.isfinite(vals).any() else np.nan
        weighted_rows.append({"variant": variant, "sample_weighted_normalized_mae": weighted})
    weighted_df = pd.DataFrame(weighted_rows)
    overall = overall.drop(columns=["sample_weighted_normalized_mae"]).merge(weighted_df, on="variant", how="left")
    overall["rank"] = overall["equal_dataset_normalized_mae"].rank(ascending=True, method="min").astype(int)
    overall["variant"] = pd.Categorical(overall["variant"], categories=VARIANTS, ordered=True)
    return overall.sort_values(["rank", "variant"]).reset_index(drop=True)


def _apply_clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, linestyle="--", alpha=0.28)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, linestyle="--", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def _finite_values(values: Iterable[object]) -> np.ndarray:
    """Flatten numeric values and keep only finite entries."""
    arrays = []
    for value in values:
        if value is None:
            continue
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return np.array([], dtype=float)
    out = np.concatenate(arrays)
    return out[np.isfinite(out)]


def _nice_upper(values: Iterable[object], *, floor: float = 1.0, pad_frac: float = 0.07) -> float:
    """Return a padded upper bound for shared y/x limits."""
    vals = _finite_values(values)
    if vals.size == 0:
        return floor
    upper = max(float(np.nanmax(vals)), floor)
    if upper <= 0:
        return floor
    return upper * (1.0 + pad_frac)


def _set_shared_ylim(axes: Iterable[plt.Axes], values: Iterable[object], *, lower: float = 0.0, pad_frac: float = 0.07) -> None:
    """Apply the same y-axis limits to a group of comparable panels."""
    upper = _nice_upper(values, floor=max(1.0, lower + 1.0), pad_frac=pad_frac)
    for ax in axes:
        ax.set_ylim(lower, upper)


def _set_symmetric_shared_ylim(axes: Iterable[plt.Axes], values: Iterable[object], *, pad_frac: float = 0.10) -> None:
    """Apply symmetric y-limits around zero, useful for signed-error panels."""
    vals = _finite_values(values)
    limit = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
    limit = max(limit * (1.0 + pad_frac), 1.0)
    for ax in axes:
        ax.set_ylim(-limit, limit)


def _legend_handles_for_variants() -> list[Patch]:
    return [Patch(facecolor=VARIANT_COLORS[v], alpha=0.65, label=VARIANT_LABELS[v]) for v in VARIANTS]


def _legend_handles_for_datasets() -> list[Line2D]:
    return [
        Line2D([0], [0], marker=DATASET_MARKERS[d], color="w", label=DATASET_LABELS[d],
               markerfacecolor="white", markeredgecolor="black", markersize=8)
        for d in DATASETS
    ]


def _dataset_gt_values(frame: pd.DataFrame, dataset: str) -> np.ndarray:
    """Return the GT values used for distribution plots."""
    sub = frame[frame["dataset"].astype(str) == dataset]
    if "gt_value" in sub.columns:
        values = sub["gt_value"].dropna().to_numpy(dtype=float)
    else:
        values = sub["gt_count"].dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _tukey_outlier_bounds(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return (np.nan, np.nan)
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return (np.nanmin(values), np.nanmax(values))
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def _annotate_mean_median(ax: plt.Axes, mean_val: float, median_val: float, *, axis: str = "x") -> None:
    if axis == "x":
        ax.axvline(mean_val, linestyle="--", color="0.35", linewidth=1.0, zorder=2)
        ax.axvline(median_val, linestyle="-", color="black", linewidth=1.15, zorder=2)
        ax.text(
            0.985,
            0.95,
            f"Median = {median_val:.1f}\nMean = {mean_val:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.55", "alpha": 0.94},
        )
    else:
        ax.axhline(mean_val, linestyle="--", color="0.35", linewidth=1.0, zorder=2)
        ax.axhline(median_val, linestyle="-", color="black", linewidth=1.15, zorder=2)
        ax.text(
            0.985,
            0.95,
            f"Median = {median_val:.1f}\nMean = {mean_val:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.55", "alpha": 0.94},
        )


def plot_dataset_difficulty_performance_reversal(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Show dataset difficulty and how all methods change across datasets.

    Left panel: dataset difficulty using GT count / GT interval midpoint.
    Right panel: MAE across datasets for all seven methods.
    """
    # One row per dataset/sample for dataset difficulty.
    sample_df = (
        df[["dataset", "sample_id", "gt_count", "gt_interval_low", "gt_interval_high", "gt_value"]]
        .drop_duplicates(["dataset", "sample_id"])
        .copy()
    )

    sample_df["gt_interval_midpoint"] = sample_df["gt_value"]

    sample_df["gt_interval_width"] = np.where(
        sample_df["gt_interval_low"].notna() & sample_df["gt_interval_high"].notna(),
        sample_df["gt_interval_high"] - sample_df["gt_interval_low"],
        0.0,
    )

    difficulty = (
        sample_df.groupby("dataset", observed=True)
        .agg(
            n_samples=("sample_id", "count"),
            median_gt=("gt_interval_midpoint", "median"),
            mean_gt=("gt_interval_midpoint", "mean"),
            median_uncertainty_width=("gt_interval_width", "median"),
            mean_uncertainty_width=("gt_interval_width", "mean"),
        )
        .reset_index()
    )

    dataset_order = difficulty.sort_values("median_gt")["dataset"].astype(str).tolist()
    dataset_labels = [DATASET_LABELS.get(d, d) for d in dataset_order]
    perf = summary.copy().sort_values(["dataset", "variant"])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.0, 6.3),
        gridspec_kw={"width_ratios": [1.0, 1.45]},
    )

    # A. Dataset difficulty
    ax = axes[0]
    rng = np.random.default_rng(7)
    box_data = [
        sample_df[sample_df["dataset"].astype(str) == dataset]["gt_interval_midpoint"]
        .dropna()
        .to_numpy(dtype=float)
        for dataset in dataset_order
    ]

    ax.boxplot(
        box_data,
        positions=np.arange(len(dataset_order)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.3},
        boxprops={"facecolor": "#D9D9D9", "edgecolor": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )

    for x_pos, dataset, vals in zip(np.arange(len(dataset_order)), dataset_order, box_data):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.17, 0.17, size=len(vals))
        ax.scatter(
            np.full(len(vals), x_pos) + jitter,
            vals,
            s=18,
            color="white",
            edgecolor="black",
            linewidth=0.45,
            alpha=0.65,
            zorder=3,
        )
        row = difficulty[difficulty["dataset"].astype(str) == dataset].iloc[0]
        ax.text(
            x_pos,
            np.nanmax(vals) * 1.03,
            f"n={int(row.n_samples)}\nmedian={row.median_gt:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("A. Ground-truth particle counts", loc="left", fontweight="bold")
    ax.set_ylabel("Ground-truth count\n(or interval midpoint)")
    ax.set_xticks(np.arange(len(dataset_order)))
    ax.set_xticklabels(dataset_labels, rotation=25, ha="right")
    ax.set_ylim(0, _nice_upper(box_data, floor=1.0, pad_frac=0.12))
    _apply_clean_axes(ax)

    # B. All methods across datasets ordered by difficulty
    ax = axes[1]
    x = np.arange(len(dataset_order))
    for variant in VARIANTS:
        vals = []
        for dataset in dataset_order:
            row = perf[
                (perf["dataset"].astype(str) == dataset)
                & (perf["variant"].astype(str) == variant)
            ]
            vals.append(float(row["mae"].iloc[0]) if not row.empty else np.nan)

        ax.plot(
            x,
            vals,
            marker="o",
            linewidth=1.8,
            markersize=5.5,
            color=VARIANT_COLORS.get(variant, "0.3"),
            label=VARIANT_LABELS.get(variant, variant),
        )

    # Circle the lowest-MAE method per dataset.
    for xi, dataset in enumerate(dataset_order):
        sub = perf[perf["dataset"].astype(str) == dataset].copy()
        if sub.empty:
            continue
        winner = sub.loc[sub["mae"].idxmin()]
        ax.scatter(
            xi,
            winner["mae"],
            s=155,
            facecolor="none",
            edgecolor="black",
            linewidth=1.8,
            zorder=5,
        )
        ax.text(
            xi,
            winner["mae"],
            "\nlowest",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title("B. Mean absolute error across datasets", loc="left", fontweight="bold")
    ax.set_ylabel("Mean absolute error (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, rotation=25, ha="right")
    ax.set_ylim(0, _nice_upper([perf["mae"].to_numpy(dtype=float)], floor=1.0, pad_frac=0.12))
    _apply_clean_axes(ax)
    ax.legend(title="Method", fontsize=8.0, title_fontsize=9, ncol=1, frameon=True)

    fig.suptitle("Dataset number of GT particles and method performance", fontsize=16, fontweight="bold")


    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    path = FIG_DIR / "08_dataset_difficulty_performance_reversal.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")

def plot_overall_mae_ranking(overall: pd.DataFrame, summary: pd.DataFrame) -> None:
    """One clear overall-effect figure using only MAE."""
    ranking = overall.sort_values("equal_dataset_mae", ascending=False).copy()
    y = np.arange(len(ranking))
    colors = [VARIANT_COLORS[str(v)] for v in ranking["variant"]]

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.barh(
        y,
        ranking["equal_dataset_mae"],
        color=colors,
        alpha=0.78,
        edgecolor="black",
        linewidth=0.8,
        label="Mean dataset MAE",
    )

    # Overlay dataset-level MAE values. This keeps the overall bar simple but transparent.
    for i, row in enumerate(ranking.itertuples()):
        sub = summary[summary["variant"].astype(str) == str(row.variant)]
        for ds_row in sub.itertuples():
            marker = DATASET_MARKERS.get(str(ds_row.dataset), "o")
            ax.scatter(
                ds_row.mae,
                i,
                marker=marker,
                s=58,
                color="white",
                edgecolor="black",
                linewidth=0.95,
                zorder=4,
            )

    labels = [f"#{int(r.rank)}  {r.variant_label}" for r in ranking.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean absolute error (lower is better)")
    ax.set_title("Overall particle-counting error", fontsize=15, fontweight="bold")
    ax.text(
        0.98,
        0.03,
        "Bars: mean across datasets\nMarkers: individual datasets",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "edgecolor": "0.55", "alpha": 0.92},
    )
    max_x = _nice_upper([ranking["equal_dataset_mae"].to_numpy(dtype=float), summary["mae"].to_numpy(dtype=float)], floor=1.0, pad_frac=0.08)
    ax.set_xlim(0, max_x)
    _apply_clean_axes(ax, grid_axis="x")
    ax.legend(handles=_legend_handles_for_datasets(), title="Dataset", loc="upper right", fontsize=8.5, frameon=True)

    fig.tight_layout()
    path = FIG_DIR / "01_overall_mae_ranking.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


BOXPLOT_FOCUSED_DATASET_ORDER = [
    "grid_scan_paper",
    "validation_unc_luca",
]


def _absolute_error_log_limits(
    df: pd.DataFrame,
    datasets_for_limits: list[str],
    *,
    pad_frac: float = 0.22,
) -> tuple[float, float, float]:
    """Return shared log-scale y-limits and the display floor for zero errors.

    A true log axis cannot display zero. Exact-zero errors are plotted at a
    small positive floor so exact matches remain visible instead of vanishing.
    """
    mask = df["dataset"].astype(str).isin(datasets_for_limits)
    values = pd.to_numeric(df.loc[mask, "absolute_error"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    positive = values[values > 0]
    if positive.size:
        zero_floor = min(0.5, max(float(np.nanmin(positive)) * 0.70, 1e-3))
    else:
        zero_floor = 0.5

    display_values = np.where(values <= 0, zero_floor, values)
    if display_values.size == 0:
        return zero_floor, zero_floor * 10.0, zero_floor

    y_min = zero_floor
    y_max = max(float(np.nanmax(display_values)) * (1.0 + pad_frac), zero_floor * 10.0)
    return y_min, y_max, zero_floor


def _values_for_log_plot(values: np.ndarray, zero_floor: float) -> np.ndarray:
    """Convert finite values to display values for a log-scaled axis."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    return np.where(values <= 0, zero_floor, values)


def _plot_error_boxpoints_by_dataset_subset(
    df: pd.DataFrame,
    datasets_to_plot: list[str],
    *,
    filename: str,
    y_limit_datasets: list[str] | None = None,
) -> None:
    """Boxplots with sample points for sample-level absolute counting error.

    The y-axis is log-scaled. All panels use the same y-axis limits. When this
    helper is used for the two-dataset view, the limits are still computed from
    the full boxplot dataset set so the two saved figures remain comparable.

    The black horizontal line inside each box is the median. The numeric label
    next to each box is that same median value. No separate mean marker is drawn,
    to avoid showing two competing horizontal markers on each boxplot.
    """
    available = set(df["dataset"].astype(str))
    datasets = [d for d in datasets_to_plot if d in available]
    if not datasets:
        print(f"[WARN] Skipping {filename}: none of the requested datasets are present.")
        return

    y_limit_datasets = y_limit_datasets or datasets
    y_min, y_max, zero_floor = _absolute_error_log_limits(df, y_limit_datasets)

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.9 * len(datasets), 6.3),
        sharey=True,
    )
    if len(datasets) == 1:
        axes = [axes]

    rng = np.random.default_rng(7)

    for panel_idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        sub = df[df["dataset"].astype(str) == dataset].copy()
        variant_order = [v for v in METHOD_COMPARISON_ORDER if v in set(sub["variant"].astype(str))]
        positions = np.arange(len(variant_order))

        raw_data = [
            sub[sub["variant"].astype(str) == v]["absolute_error"].dropna().to_numpy(dtype=float)
            for v in variant_order
        ]
        plot_data = [_values_for_log_plot(vals, zero_floor) for vals in raw_data]

        non_empty_positions = [pos for pos, vals in zip(positions, plot_data) if len(vals) > 0]
        non_empty_data = [vals for vals in plot_data if len(vals) > 0]

        if non_empty_data:
            bp = ax.boxplot(
                non_empty_data,
                positions=non_empty_positions,
                widths=0.58,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 2.0},
                boxprops={"facecolor": "white", "alpha": 0.68, "edgecolor": "black", "linewidth": 1.0},
                whiskerprops={"color": "black", "linewidth": 0.95},
                capprops={"color": "black", "linewidth": 0.95},
            )
            for patch, variant in zip(bp["boxes"], [v for v, vals in zip(variant_order, plot_data) if len(vals) > 0]):
                patch.set_facecolor(VARIANT_COLORS[variant])
                patch.set_alpha(0.26)
                patch.set_edgecolor(VARIANT_COLORS[variant])

        for pos, variant, raw_vals, display_vals in zip(positions, variant_order, raw_data, plot_data):
            if len(display_vals) == 0:
                continue

            jitter = rng.uniform(-0.19, 0.19, size=len(display_vals))
            ax.scatter(
                np.full(len(display_vals), pos) + jitter,
                display_vals,
                s=16,
                color=VARIANT_COLORS[variant],
                edgecolor="black",
                linewidth=0.28,
                alpha=0.50,
                zorder=3,
            )

            median_val = float(np.median(raw_vals))
            median_display_val = max(median_val, zero_floor)
            ax.text(
                pos + 0.35,
                median_display_val,
                f"{median_val:.1f}",
                fontsize=8.8,
                fontweight="bold",
                ha="left",
                va="center",
                color="black",
                zorder=5,
            )

        ax.set_title(
            f"{PANEL_LETTERS[panel_idx]}. {pretty_dataset(dataset)}",
            fontweight="bold",
            fontsize=13,
            loc="left",
        )
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)

        if panel_idx == 0:
            ax.set_ylabel(
                "Absolute Error (log scale)",
                fontsize=13,
                fontweight="bold",
            )
        else:
            ax.set_ylabel("")

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [VARIANT_LABELS[v] for v in variant_order],
            rotation=35,
            ha="right",
            fontsize=10,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=10)
        _apply_clean_axes(ax)

    dot_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="gray",
        markeredgecolor="black",
        label="Sample",
        markersize=6.5,
    )
    median_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=2.0,
        label="Median",
    )
    fig.legend(
        handles=[dot_handle, median_handle],
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        handlelength=1.8,
        columnspacing=1.6,
    )

    # No large suptitle: each panel keeps only its dataset subtitle.
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    path = FIG_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_error_boxpoints_by_dataset(df: pd.DataFrame) -> None:
    """Main CS2 absolute-error boxplot with all three datasets."""
    _plot_error_boxpoints_by_dataset_subset(
        df,
        BOXPLOT_DATASET_ORDER,
        filename="02_absolute_error_boxpoints_by_dataset.png",
        y_limit_datasets=BOXPLOT_DATASET_ORDER,
    )


def plot_error_boxpoints_grid_scan_rettenberger(df: pd.DataFrame) -> None:
    """Focused CS2 absolute-error boxplot for Grid Scan Paper and Rettenberger."""
    _plot_error_boxpoints_by_dataset_subset(
        df,
        BOXPLOT_FOCUSED_DATASET_ORDER,
        filename="02b_absolute_error_boxpoints_grid_scan_rettenberger.png",
        y_limit_datasets=BOXPLOT_DATASET_ORDER,
    )


def plot_predicted_count_boxpoints_by_dataset(df: pd.DataFrame) -> None:
    """Boxplots with sample points for predicted counts on the Rettenberger dataset.

    This version keeps the original scientific styling but uses a linear scale so
    the counts stay easier to read in the column layout.
    """
    dataset = "validation_unc_luca"
    sub = df[df["dataset"].astype(str) == dataset].copy()
    if sub.empty:
        print("[WARN] Skipping predicted-count boxplot: no Rettenberger data found.")
        return

    order = ["gt"] + list(VARIANTS)
    labels = ["GT"] + [VARIANT_LABELS[v] for v in VARIANTS]
    colors = ["#666666"] + [VARIANT_COLORS[v] for v in VARIANTS]

    rng = np.random.default_rng(7)
    positions = np.arange(len(order))
    data = []
    for item in order:
        if item == "gt":
            vals = sub["gt_count"].dropna().to_numpy(dtype=float)
        else:
            vals = sub[sub["variant"].astype(str) == item]["predicted_count"].dropna().to_numpy(dtype=float)
        data.append(vals)

    fig, ax = plt.subplots(figsize=single_col_figsize(5.6))

    non_empty_positions = [pos for pos, vals in zip(positions, data) if len(vals) > 0]
    non_empty_data = [vals for vals in data if len(vals) > 0]
    bp = ax.boxplot(
        non_empty_data,
        positions=non_empty_positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "white", "alpha": 0.65, "edgecolor": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
    )

    box_items = [item for item, vals in zip(order, data) if len(vals) > 0]
    for patch, item in zip(bp["boxes"], box_items):
        patch.set_facecolor("#666666" if item == "gt" else VARIANT_COLORS[item])
        patch.set_alpha(0.20 if item == "gt" else 0.24)
        patch.set_edgecolor("#666666" if item == "gt" else VARIANT_COLORS[item])

    for pos, item, vals, color in zip(positions, order, data, colors):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.19, 0.19, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=12,
            color=color,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.45,
            zorder=3,
        )
        mean_val = float(np.mean(vals))
        ax.scatter(pos, mean_val, marker="_", s=250, color="black", linewidth=1.8, zorder=4)
        ax.text(pos, mean_val, f" {mean_val:.1f}", fontsize=7, ha="left", va="center")

    ax.set_title("9. Rettenberger Uncertainty: predicted counts vs GT", fontweight="bold", loc="left")
    ax.set_ylabel("Count")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, _nice_upper(data, floor=1.0, pad_frac=0.12))
    _apply_clean_axes(ax)

    fig.suptitle("Rettenberger counts: GT and model predictions", fontsize=16, fontweight="bold")
    mean_handle = Line2D([0], [0], marker="_", color="black", linestyle="None", label="mean", markersize=13, markeredgewidth=1.8)
    dot_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", label="one sample", markersize=5)
    gt_handle = Patch(facecolor="#666666", alpha=0.20, label="GT")
    variant_handles = [Patch(facecolor=VARIANT_COLORS[v], alpha=0.24, label=VARIANT_LABELS[v]) for v in VARIANTS]
    fig.legend(handles=[gt_handle] + variant_handles + [mean_handle, dot_handle], loc="lower center", ncol=4, fontsize=8.2, frameon=True)
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    path = FIG_DIR / "09_predicted_count_boxpoints_by_dataset.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_gt_distribution_three_datasets(df: pd.DataFrame) -> None:
    """Compare GT value distributions for Grid Scan Paper and Rettenberger."""
    datasets = ["grid_scan_paper", "validation_unc_luca"]
    present = [d for d in datasets if d in set(df["dataset"].astype(str))]
    if len(present) < 2:
        print("[WARN] Skipping GT comparison plot: need Grid Scan and Rettenberger data.")
        return

    short_labels = {"grid_scan_paper": "Grid Scan", "validation_unc_luca": "Rettenberger"}
    fig, ax = plt.subplots(figsize=single_col_figsize(6.2))
    rng = np.random.default_rng(7)
    positions = np.arange(len(present))
    data = [_dataset_gt_values(df, dataset) for dataset in present]

    violin = ax.violinplot(
        data,
        positions=positions,
        widths=0.74,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#C8C8C8")
        body.set_edgecolor("black")
        body.set_alpha(0.42)
        body.set_linewidth(0.9)

    box_data = [vals for vals in data if len(vals) > 0]
    box_positions = [pos for pos, vals in zip(positions, data) if len(vals) > 0]
    ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.15},
        boxprops={"facecolor": "white", "alpha": 0.76, "edgecolor": "black", "linewidth": 0.9},
        whiskerprops={"color": "black", "linewidth": 0.85},
        capprops={"color": "black", "linewidth": 0.85},
    )

    for pos, dataset, vals in zip(positions, present, data):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=13,
            color="white" if dataset != LARGE_DATASET else "#5A5A5A",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.62,
            zorder=3,
        )
        median_val = float(np.nanmedian(vals))
        mean_val = float(np.nanmean(vals))
        text_y = np.nanmax(vals) * 1.18
        ax.text(
            pos,
            text_y,
            f"med={median_val:.1f}\nmean={mean_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "0.65", "alpha": 0.90},
        )

    ax.set_title("GT value distribution across datasets", loc="left", fontweight="bold")
    ax.set_ylabel("GT count")
    ax.set_xticks(positions)
    ax.set_xticklabels([short_labels[d] for d in present], rotation=0, ha="center")
    ax.set_yscale("log")
    upper = _nice_upper(data, floor=1.0, pad_frac=0.25)
    ax.set_ylim(bottom=max(1.0, float(np.nanmin(_finite_values(data))) * 0.85 if _finite_values(data).size else 1.0), top=upper)
    ax.set_xlabel("")
    _apply_clean_axes(ax)

    handles = [
        Line2D([0], [0], marker="o", color="w", label="samples", markerfacecolor="white", markeredgecolor="black", markersize=5),
        Patch(facecolor="white", alpha=0.55, edgecolor="black", label="boxplot"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.1, frameon=True)
    fig.tight_layout()
    path = FIG_DIR / "05_gt_value_distribution_two_datasets.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_single_dataset_gt_distribution(df: pd.DataFrame, dataset: str, *, title: str, path: Path) -> None:
    values = _dataset_gt_values(df, dataset)
    if values.size == 0:
        print(f"[WARN] Skipping GT distribution plot: no data for {dataset}.")
        return

    fig, ax = plt.subplots(figsize=single_col_figsize(5.9))
    rng = np.random.default_rng(7)

    ax.violinplot(
        [values],
        positions=[0],
        widths=0.62,
        vert=False,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in ax.collections:
        try:
            body.set_facecolor("#C8C8C8")
            body.set_edgecolor("black")
            body.set_alpha(0.42)
            body.set_linewidth(0.9)
        except Exception:
            pass

    ax.boxplot(
        [values],
        positions=[0],
        widths=0.22,
        vert=False,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "white", "alpha": 0.76, "edgecolor": "black", "linewidth": 0.9},
        whiskerprops={"color": "black", "linewidth": 0.85},
        capprops={"color": "black", "linewidth": 0.85},
    )

    jitter = rng.uniform(-0.085, 0.085, size=len(values))
    ax.scatter(values, jitter, s=14, color="white", edgecolor="black", linewidth=0.35, alpha=0.62, zorder=3)

    lower, upper = _tukey_outlier_bounds(values)
    outliers = values[(values < lower) | (values > upper)] if np.isfinite(lower) and np.isfinite(upper) else np.array([])
    if outliers.size:
        ax.scatter(outliers, np.zeros_like(outliers), s=26, marker="D", facecolor="none", edgecolor="black", linewidth=1.0, zorder=4)
        offset_cycle = [10, -14, 18, -22, 26, -30]
        for idx, val in enumerate(np.sort(outliers)):
            offset = offset_cycle[idx % len(offset_cycle)]
            ax.annotate(
                f"{val:.0f}",
                xy=(val, 0),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if offset >= 0 else "top",
                fontsize=7.0,
                color="black",
                arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.7, "shrinkA": 0, "shrinkB": 0},
            )

    mean_val = float(np.nanmean(values))
    median_val = float(np.nanmedian(values))
    ax.axvline(mean_val, linestyle="--", color="0.35", linewidth=1.0, zorder=2)
    ax.axvline(median_val, linestyle="-", color="black", linewidth=1.15, zorder=2)
    ax.text(
        0.985,
        0.94,
        f"Median = {median_val:.1f}\nMean = {mean_val:.1f}\nOutliers = {len(outliers)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.94},
    )

    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("GT count")
    ax.set_yticks([])
    ax.set_ylim(-0.40, 0.40)
    ax.set_xlim(left=max(0.0, float(np.nanmin(values)) * 0.92))
    ax.text(
        0.02,
        0.10,
        f"n = {len(values)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.92},
    )
    _apply_clean_axes(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_gt_distribution_grid_scan(df: pd.DataFrame) -> None:
    _plot_single_dataset_gt_distribution(
        df,
        "grid_scan_paper",
        title="Grid Scan Paper: GT distribution",
        path=FIG_DIR / "06_gt_distribution_grid_scan_paper.png",
    )


def plot_gt_distribution_rettenberger(df: pd.DataFrame) -> None:
    _plot_single_dataset_gt_distribution(
        df,
        "validation_unc_luca",
        title="Rettenberger Uncertainty: GT distribution",
        path=FIG_DIR / "07_gt_distribution_rettenberger.png",
    )

def _rettenberger_case_study_label(sample_id: str) -> str:
    tail = str(sample_id).split("_")[-1]
    try:
        num = int(tail)
        return f"Rettenberger low mag. {num}"
    except Exception:
        return f"Rettenberger low mag. {tail}"


def plot_rettenberger_case_study_examples(df: pd.DataFrame) -> None:
    """Show three Rettenberger samples with GT intervals and all method counts.

    The figure focuses on the samples that drive the high ImgGen MAE. It uses a
    compact three-row layout with the GT interval as a gray band and one marker
    per model prediction.
    """
    dataset = "validation_unc_luca"
    target_ids = [
        "luca_unc_low_mag_020",
        "luca_unc_low_mag_022",
        "luca_unc_low_mag_035",
    ]
    sub = df[df["dataset"].astype(str) == dataset].copy()
    if sub.empty:
        print("[WARN] Skipping case-study figure: no Rettenberger data found.")
        return

    available_ids = set(sub["sample_id"].astype(str))
    selected_ids = [sid for sid in target_ids if sid in available_ids]

    if len(selected_ids) < len(target_ids):
        missing = [sid for sid in target_ids if sid not in available_ids]
        print(f"[WARN] Case-study figure: missing expected low-mag samples: {missing}")
        if not selected_ids:
            return

    panel_data = []
    for sample_id in selected_ids:
        rows = sub[sub["sample_id"].astype(str) == sample_id].sort_values("variant")
        if rows.empty:
            continue
        row0 = rows.iloc[0]
        panel_data.append((sample_id, row0, rows))

    if len(panel_data) < 1:
        print(f"[WARN] Skipping case-study figure: incomplete sample rows. Found: {[p[0] for p in panel_data]}")
        return

    fig, axes = plt.subplots(
        len(panel_data),
        1,
        figsize=(6.9, max(4.2, 1.45 * len(panel_data))),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    if len(panel_data) == 1:
        axes = [axes]

    x_max = 0.0
    x_min = np.inf
    for _, row0, rows in panel_data:
        if pd.notna(row0.get("gt_interval_low")):
            x_min = min(x_min, float(row0["gt_interval_low"]))
        if pd.notna(row0.get("gt_interval_high")):
            x_max = max(x_max, float(row0["gt_interval_high"]))
        if rows["predicted_count"].notna().any():
            x_min = min(x_min, float(np.nanmin(rows["predicted_count"].to_numpy(dtype=float))))
            x_max = max(x_max, float(np.nanmax(rows["predicted_count"].to_numpy(dtype=float))))

    if not np.isfinite(x_min):
        x_min = 0.0
    if not np.isfinite(x_max):
        x_max = 1.0
    x_min = max(0.0, x_min * 0.94)
    x_max = max(1.0, x_max * 1.06)

    variant_offsets = {
        variant: offset
        for variant, offset in zip(VARIANTS, np.linspace(-0.14, 0.14, len(VARIANTS)))
    }

    for ax, (sample_id, row0, rows) in zip(axes, panel_data):
        low = float(row0["gt_interval_low"]) if pd.notna(row0["gt_interval_low"]) else np.nan
        high = float(row0["gt_interval_high"]) if pd.notna(row0["gt_interval_high"]) else np.nan
        y0 = 0.0

        if np.isfinite(low) and np.isfinite(high):
            ax.broken_barh(
                [(low, max(high - low, 0.0))],
                (y0 - 0.12, 0.24),
                facecolors="#BFBFBF",
                edgecolors="#666666",
                linewidth=0.9,
                zorder=1,
            )
            ax.vlines([low, high], y0 - 0.15, y0 + 0.15, colors="black", linewidth=1.0, zorder=2)

        for _, r in rows.iterrows():
            variant = str(r["variant"])
            pred = r["predicted_count"]
            if pd.isna(pred):
                continue
            ax.scatter(
                float(pred),
                y0 + variant_offsets.get(variant, 0.0),
                s=36 if variant in {"E3_imggen_deterministic", "E5_imggen_overlay_vlm"} else 28,
                color=VARIANT_COLORS.get(variant, "0.3"),
                edgecolor="black",
                linewidth=0.45,
                zorder=4,
            )

        sample_label = _rettenberger_case_study_label(sample_id)
        ax.set_yticks([0])
        ax.set_yticklabels([sample_label])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.38, 0.38)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_visible(False)
        _apply_clean_axes(ax, grid_axis="x")

    axes[-1].set_xlabel("Particle count")
    axes[-1].set_xlim(x_min, x_max)
    axes[0].set_title("Rettenberger case study: GT interval vs model counts", loc="left", fontweight="bold")

    legend_handles = [
        Patch(facecolor="#BFBFBF", edgecolor="#666666", label="GT interval"),
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VARIANT_COLORS[v], markeredgecolor="black", linestyle="None", markersize=6.2, label=VARIANT_LABELS[v])
        for v in VARIANTS
    ]

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.91, 0.5),
        frameon=True,
        fontsize=7.6,
        title="Legend",
    )

    fig.tight_layout(rect=[0.0, 0.0, 0.76, 1.0])
    path = FIG_DIR / "16_rettenberger_case_study_020_022_035.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_large_vs_experimental_mae_alignment(summary: pd.DataFrame) -> None:
    """Check whether large-set variant conclusions align with validation sets, using MAE."""
    large = summary[summary["dataset"].astype(str) == LARGE_DATASET][["variant", "mae"]].rename(columns={"mae": "large_mae"})
    exp = (
        summary[summary["dataset"].astype(str).isin(EXPERIMENTAL_DATASETS)]
        .groupby("variant", observed=True)
        .agg(experimental_mae=("mae", "mean"))
        .reset_index()
    )
    comp = large.merge(exp, on="variant", how="inner")
    if comp.empty:
        print("[WARN] Skipping alignment plot: no overlap between large and experimental datasets.")
        return

    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    for row in comp.itertuples():
        variant = str(row.variant)
        ax.scatter(
            row.large_mae,
            row.experimental_mae,
            s=92,
            color=VARIANT_COLORS[variant],
            edgecolor="black",
            linewidth=0.9,
            label=VARIANT_LABELS[variant],
            zorder=3,
        )
        ax.text(row.large_mae, row.experimental_mae, "  " + VARIANT_LABELS[variant], fontsize=8.4, va="center")

    max_val = float(np.nanmax([comp["large_mae"].max(), comp["experimental_mae"].max()]))
    max_val = max(max_val * 1.12, 1.0)
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="0.35", linewidth=1.0, label="equal MAE")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("MAE on large set\n(Grid Scan Paper)")
    ax.set_ylabel("Mean MAE on experimental sets\n(Rettenberger Uncertainty + TFS Mirror)")
    ax.set_title("Large-set and validation-set alignment", fontsize=15, fontweight="bold")
    ax.text(
        0.02,
        0.98,
        "Lower-left is better.\nDiagonal = similar error.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "edgecolor": "0.55", "alpha": 0.92},
    )
    _apply_clean_axes(ax, grid_axis="both")
    fig.tight_layout()
    path = FIG_DIR / "03_large_vs_experimental_mae_alignment.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_mae_heatmap(summary: pd.DataFrame) -> None:
    """Compact heatmap for comparing MAE across datasets and variants."""
    pivot = summary.pivot_table(index="dataset_label", columns="variant_label", values="mae", observed=True)
    pivot = pivot.reindex(index=[DATASET_LABELS[d] for d in DATASETS], columns=[VARIANT_LABELS[v] for v in VARIANTS])
    mat = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12.2, 4.4))
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color="#D9D9D9")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap)
    ax.set_title("Mean absolute error by dataset and method", fontsize=15, fontweight="bold")
    ax.set_xlabel("Method")
    ax.set_ylabel("Dataset")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(str), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    finite_vals = mat[np.isfinite(mat)]
    threshold = np.nanmedian(finite_vals) if len(finite_vals) else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > threshold else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Mean absolute error (lower is better)")
    fig.tight_layout()
    path = FIG_DIR / "04_mae_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_threshold_metric_splits(summary: pd.DataFrame) -> None:
    """Split threshold metrics into matched, slide-ready percentage bar charts."""
    variant_order = [v for v in METHOD_COMPARISON_ORDER if v in set(summary["variant"].astype(str))]
    if not variant_order:
        print("[WARN] Skipping threshold plots: no known variants found.")
        return

    # Two-line tick labels keep the model names readable without clipping.
    threshold_variant_labels = {
        "E1_raw_vlm": "VLM",
        "E2_sam2_deterministic": "SAM2\nmask",
        "E4_sam2_overlay_vlm": "SAM2\noverlay",
        "E6_sam3_deterministic": "SAM3\nmask",
        "E7_sam3_overlay_vlm": "SAM3\noverlay",
        "E3_imggen_deterministic": "ImgGen\nmask",
        "E5_imggen_overlay_vlm": "ImgGen\noverlay",
    }

    # Same canvas, margins, fonts, and axis range for all threshold plots.
    figsize = (6.2, 4.15)
    axis_label_fontsize = 14
    tick_fontsize = 10
    value_fontsize = 9

    def _bar_panel(sub: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
        if sub.empty:
            print(f"[WARN] Skipping threshold plot: no data for {path.name}.")
            return

        sub = sub.copy()
        sub["variant_str"] = sub["variant"].astype(str)
        sub = sub.set_index("variant_str").reindex(variant_order).reset_index()

        x = np.arange(len(variant_order))
        values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        colors = [VARIANT_COLORS[v] for v in variant_order]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            x,
            values,
            color=colors,
            alpha=0.82,
            edgecolor="black",
            linewidth=0.65,
        )

        for rect, val in zip(bars, values):
            if not np.isfinite(val):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                min(val + 2.0, 97.0),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
                fontweight="bold",
            )

        ax.set_ylim(0, 100)
        ax.set_xlim(-0.55, len(variant_order) - 0.45)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [threshold_variant_labels.get(v, VARIANT_LABELS[v]) for v in variant_order],
            rotation=0,
            ha="center",
            fontsize=tick_fontsize,
            fontweight="bold",
            linespacing=0.95,
        )
        ax.set_ylabel(
            ylabel,
            fontsize=axis_label_fontsize,
            fontweight="bold",
            labelpad=8,
        )
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.tick_params(axis="x", pad=7)
        _apply_clean_axes(ax)

        # Fixed margins and no title so figures 13 and 14 have identical layout.
        # Extra left/bottom space prevents y-axis and model labels from being cut off.
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.22, top=0.96)
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(path, dpi=220)
        plt.close(fig)
        print(f"Saved: {path}")

    rettenberger = summary[summary["dataset"].astype(str) == "validation_unc_luca"]
    _bar_panel(
        rettenberger,
        metric="inside_interval_pct",
        ylabel="Inside GT Interval (%)",
        path=FIG_DIR / "13_threshold_inside_gt_interval_rettenberger.png",
    )

    grid_scan = summary[summary["dataset"].astype(str) == "grid_scan_paper"]
    _bar_panel(
        grid_scan,
        metric="within_1_pct",
        ylabel="Within 1 Count (%)",
        path=FIG_DIR / "14_threshold_within1_grid_scan_paper.png",
    )

    tfs_mirror = summary[summary["dataset"].astype(str) == "validation_andrea_grid_scan_mirror"]
    _bar_panel(
        tfs_mirror,
        metric="within_1_pct",
        ylabel="Within 1 Count (%)",
        path=FIG_DIR / "15_threshold_within1_tfs_mirror.png",
    )

def plot_signed_bias_panel(summary: pd.DataFrame) -> None:
    """Show whether variants systematically overcount or undercount."""
    datasets = [d for d in DATASETS if d in set(summary["dataset"].astype(str))]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 5.2), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for panel_idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        sub = summary[summary["dataset"].astype(str) == dataset].copy()
        x = np.arange(len(sub))
        colors = [VARIANT_COLORS[str(v)] for v in sub["variant"]]
        ax.bar(x, sub["mean_signed_error"], color=colors, alpha=0.78, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["variant_label"].astype(str), rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{PANEL_LETTERS[panel_idx]}. {pretty_dataset(dataset)}", fontweight="bold", loc="left")
        if panel_idx == 0:
            ax.set_ylabel("Mean signed error\n(predicted - ground truth)")
        else:
            ax.set_ylabel("")
        _apply_clean_axes(ax)

    # Keep the signed-error y-axis identical and symmetric across panels.
    _set_symmetric_shared_ylim(axes, [summary["mean_signed_error"].to_numpy(dtype=float)], pad_frac=0.10)

    fig.suptitle("Mean signed counting error by method", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = FIG_DIR / "06_signed_error_bias_panel.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_coverage_matrix(df: pd.DataFrame) -> None:
    coverage = df.groupby(["dataset_label", "variant_label"], observed=True).size().unstack(fill_value=0)
    coverage = coverage.reindex(index=[DATASET_LABELS[d] for d in DATASETS], columns=[VARIANT_LABELS[v] for v in VARIANTS])
    mat = coverage.to_numpy(dtype=float)
    vmax = np.nanmax(mat) if not np.isnan(mat).all() else 1
    fig, ax = plt.subplots(figsize=(12.0, 4.0))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#D9D9D9")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title("Evaluation coverage by dataset and method", fontsize=15, fontweight="bold")
    ax.set_xlabel("Method")
    ax.set_ylabel("Dataset")
    ax.set_xticks(np.arange(len(coverage.columns)))
    ax.set_xticklabels(coverage.columns.astype(str), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(coverage.index)))
    ax.set_yticklabels(coverage.index.astype(str))
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            if np.isfinite(mat[y, x]):
                ax.text(x, y, f"{int(mat[y, x])}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Completed comparable datapoints")
    fig.tight_layout()
    path = FIG_DIR / "07_coverage_matrix.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def save_plot_data(df: pd.DataFrame, summary: pd.DataFrame, overall: pd.DataFrame, normalized_overall: pd.DataFrame) -> None:
    clean_path = DATA_OUT / "cs2_clean_latest_results.csv"
    df.sort_values(["dataset", "sample_id", "variant"]).to_csv(clean_path, index=False)
    print(f"Saved: {clean_path}")

    summary_path = DATA_OUT / "cs2_clean_latest_summary_by_dataset_variant.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    overall_path = DATA_OUT / "cs2_overall_mae_ranking.csv"
    overall.to_csv(overall_path, index=False)
    print(f"Saved: {overall_path}")

    norm_overall_path = DATA_OUT / "cs2_overall_normalized_mae_ranking.csv"
    normalized_overall.to_csv(norm_overall_path, index=False)
    print(f"Saved: {norm_overall_path}")

    boxpoint_path = DATA_OUT / "cs2_boxpoint_sample_errors.csv"
    df[[
        "dataset", "dataset_label", "sample_id", "variant", "variant_label",
        "absolute_error", "relative_error", "signed_error", "predicted_count", "gt_count", "gt_value",
        "inside_gt_interval", "within_1", "within_10_pct",
    ]].sort_values(["dataset", "variant", "sample_id"]).to_csv(boxpoint_path, index=False)
    print(f"Saved: {boxpoint_path}")


def print_console_summary(df: pd.DataFrame, summary: pd.DataFrame, overall: pd.DataFrame, normalized_overall: pd.DataFrame) -> None:
    print("\nCoverage:")
    coverage = df.groupby(["dataset_label", "variant_label"], observed=True).size().unstack(fill_value=0)
    print(coverage)

    print("\nOverall variant ranking by equal-dataset MAE:")
    cols = ["rank", "variant_label", "equal_dataset_mae", "sample_weighted_mae", "mean_exact_pct", "mean_within_1_pct", "mean_within_10_pct"]
    print(overall[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nDataset-level summary:")
    compact = summary[["dataset_label", "variant_label", "n", "mae", "exact_pct", "within_1_pct", "within_10_pct"]]
    print(compact.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_latest_results()
    summary = summarize_by_dataset_variant(df)
    overall = compute_overall_mae_ranking(summary)
    normalized_overall = compute_overall_normalized_mae_ranking(summary)

    save_plot_data(df, summary, overall, normalized_overall)
    print_console_summary(df, summary, overall, normalized_overall)

    plot_overall_mae_ranking(overall, summary)
    plot_error_boxpoints_by_dataset(df)
    plot_error_boxpoints_grid_scan_rettenberger(df)
    plot_gt_distribution_three_datasets(df)
    plot_gt_distribution_grid_scan(df)
    plot_gt_distribution_rettenberger(df)
    plot_rettenberger_case_study_examples(df)
    plot_large_vs_experimental_mae_alignment(summary)
    plot_mae_heatmap(summary)
    plot_threshold_metric_splits(summary)
    plot_signed_bias_panel(summary)
    plot_coverage_matrix(df)
    plot_dataset_difficulty_performance_reversal(df, summary)

    print(f"\n[INFO] CS2 figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
