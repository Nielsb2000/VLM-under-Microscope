#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from caseviz.io import load_runs
from caseviz.matching import matched_pairs
from caseviz.plots import (
    plot_delta_histogram,
    plot_filter_trajectories,
    plot_image_contact_sheet,
    plot_improvement_vs_start,
)

# ---------------------------------------------------------------------
# Publication-style constants
# ---------------------------------------------------------------------

METHOD_ORDER = ["direct", "non_exploratory", "non-exploratory", "exploratory"]

METHOD_LABELS = {
    "direct": "Direct refinement",
    "non_exploratory": "Direct refinement",
    "non-exploratory": "Direct refinement",
    "exploratory": "Exploration + refinement",
}

METHOD_COLORS = {
    "direct": "#0072B2",
    "non_exploratory": "#0072B2",
    "non-exploratory": "#0072B2",
    "exploratory": "#D55E00",
}

NEUTRAL = "0.35"


def set_publication_style() -> None:
    """Apply a compact scientific-journal style for all figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def completed_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Return completed runs with valid final scores."""
    required = {"status", "final_score"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    out = df[(df["status"] == "completed") & df["final_score"].notna()].copy()
    out["method_label"] = out["method"].map(lambda m: METHOD_LABELS.get(str(m), str(m).replace("_", " ").title()))
    return out


def ordered_methods(values: Iterable[object]) -> list[str]:
    observed = [str(v) for v in values]
    ordered = [m for m in METHOD_ORDER if m in observed]
    ordered.extend(sorted(m for m in set(observed) if m not in ordered))
    return ordered


def bootstrap_ci(values: np.ndarray, statistic=np.median, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI for a one-dimensional statistic."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = statistic(values[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def savefig(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Plot 1: final restoration score by method
# ---------------------------------------------------------------------

def plot_final_score_estimation(long_pairs: pd.DataFrame, wide_pairs: pd.DataFrame, out_dir: Path) -> None:
    """Plot 1: final score on the same matched pairs as Figure 2."""
    if wide_pairs is None or wide_pairs.empty:
        print("[WARN] Skipping plot 1: wide_pairs is empty.")
        return

    def find_col(columns: Iterable[str], candidates: list[str]) -> str | None:
        colset = set(columns)
        for candidate in candidates:
            if candidate in colset:
                return candidate
        return None

    columns = wide_pairs.columns
    direct_final_col = find_col(
        columns,
        [
            "final_score_direct",
            "direct_final_score",
            "score_direct",
            "direct_score",
            "final_score_non_exploratory",
            "non_exploratory_final_score",
            "score_non_exploratory",
            "non_exploratory_score",
            "final_score_non-exploratory",
            "score_non-exploratory",
        ],
    )
    exploratory_final_col = find_col(
        columns,
        [
            "final_score_exploratory",
            "exploratory_final_score",
            "score_exploratory",
            "exploratory_score",
        ],
    )

    if direct_final_col is None or exploratory_final_col is None:
        print("[WARN] Skipping plot 1: could not find direct/exploratory final-score columns in wide_pairs.")
        print(list(wide_pairs.columns))
        return

    pair_table = pd.DataFrame(
        {
            "direct": pd.to_numeric(wide_pairs[direct_final_col], errors="coerce"),
            "exploratory": pd.to_numeric(wide_pairs[exploratory_final_col], errors="coerce"),
        }
    ).dropna(subset=["direct", "exploratory"])

    if pair_table.empty:
        print("[WARN] Skipping plot 1: no complete matched pairs with both final scores.")
        return

    n_pairs = len(pair_table)
    methods = ["direct", "exploratory"]

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    rng = np.random.default_rng(7)

    for x_idx, method in enumerate(methods):
        y = pair_table[method].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if len(y) == 0:
            continue

        color = METHOD_COLORS.get(method, "0.35")
        jitter = rng.uniform(-0.08, 0.08, size=len(y))

        # show the actual matched-pair points
        ax.scatter(
            np.full(len(y), x_idx) + jitter,
            y,
            s=28,
            color=color,
            alpha=0.65,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

        median = float(np.median(y))
        lo, hi = bootstrap_ci(y, np.median)
        ax.errorbar(
            x_idx,
            median,
            yerr=[[median - lo], [hi - median]],
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
            markersize=5,
            zorder=4,
        )

    ax.set_title("Final restoration score on matched pairs", loc="left", fontweight="bold")
    ax.set_ylabel("Final histogram score")
    ax.set_xlabel("")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([METHOD_LABELS.get(m, m.replace("_", " ").title()) for m in methods], rotation=12, ha="right")

    y_all = pair_table.to_numpy(dtype=float).ravel()
    y_all = y_all[np.isfinite(y_all)]
    if len(y_all):
        ymin = max(0.0, float(np.min(y_all)) * 0.90)
        ymax = float(np.max(y_all)) * 1.12
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            ax.set_ylim(ymin, ymax)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markeredgecolor="black", markersize=6, label="Median + 95% CI"),
        Line2D([0], [0], color="none", label=f"Matched-pairs n = {n_pairs}"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        handlelength=1.2,
    )

    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True, linestyle="--", alpha=0.28)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    savefig(fig, out_dir, "01_final_restoration_error_by_method.png")


# ---------------------------------------------------------------------
# Pair extraction for plot 2
# ---------------------------------------------------------------------

def _find_first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    colset = set(columns)
    for candidate in candidates:
        if candidate in colset:
            return candidate
    return None


def extract_direct_exploratory_pairs(long_pairs: pd.DataFrame, wide_pairs: pd.DataFrame) -> pd.DataFrame:
    """Extract matched direct/exploratory final scores from wide or long pairs.

    The exact column names may differ depending on the matching implementation,
    so this function tries the common names first and falls back to a long-form
    pivot when possible.
    """
    if wide_pairs is not None and not wide_pairs.empty:
        direct_col = _find_first_existing(
            wide_pairs.columns,
            [
                "final_score_direct",
                "direct_final_score",
                "score_direct",
                "direct_score",
                "final_score_non_exploratory",
                "non_exploratory_final_score",
                "score_non_exploratory",
                "non_exploratory_score",
                "final_score_non-exploratory",
                "score_non-exploratory",
            ],
        )
        exploratory_col = _find_first_existing(
            wide_pairs.columns,
            [
                "final_score_exploratory",
                "exploratory_final_score",
                "score_exploratory",
                "exploratory_score",
            ],
        )

        if direct_col and exploratory_col:
            out = wide_pairs[[direct_col, exploratory_col]].copy()
            out = out.rename(columns={direct_col: "direct_score", exploratory_col: "exploratory_score"})
            out = out.dropna(subset=["direct_score", "exploratory_score"])
            out["delta_exploratory_minus_direct"] = out["exploratory_score"] - out["direct_score"]
            return out.reset_index(drop=True)

    if long_pairs is not None and not long_pairs.empty and {"method", "final_score"}.issubset(long_pairs.columns):
        id_candidates = [
            "pair_id",
            "matched_id",
            "case_id",
            "image_id",
            "seed",
            "run_seed",
            "degradation_seed",
            "input_id",
        ]
        id_cols = [c for c in id_candidates if c in long_pairs.columns]
        if not id_cols:
            ignored = {
                "method",
                "final_score",
                "status",
                "absolute_improvement",
                "filter_adjustments",
                "vlm_snapshots",
            }
            id_cols = [c for c in long_pairs.columns if c not in ignored]

        temp = long_pairs.copy()
        temp["method_norm"] = temp["method"].astype(str).replace(
            {
                "non_exploratory": "direct",
                "non-exploratory": "direct",
            }
        )

        pivot = temp.pivot_table(
            index=id_cols,
            columns="method_norm",
            values="final_score",
            aggfunc="mean",
        ).reset_index()

        if {"direct", "exploratory"}.issubset(pivot.columns):
            out = pivot[["direct", "exploratory"]].copy()
            out = out.rename(columns={"direct": "direct_score", "exploratory": "exploratory_score"})
            out = out.dropna(subset=["direct_score", "exploratory_score"])
            out["delta_exploratory_minus_direct"] = out["exploratory_score"] - out["direct_score"]
            return out.reset_index(drop=True)

    return pd.DataFrame()


def paired_test_direct_vs_exploratory(pairs: pd.DataFrame) -> dict[str, float | str | bool]:
    """Paired t-test for direct vs exploratory scores.

    The direct and exploratory scores are matched one-to-one by seed/case, so
    the test is applied to the within-pair differences. Lower histogram score
    is better.
    """
    direct = pairs["direct_score"].to_numpy(dtype=float)
    exploratory = pairs["exploratory_score"].to_numpy(dtype=float)
    mask = np.isfinite(direct) & np.isfinite(exploratory)
    direct = direct[mask]
    exploratory = exploratory[mask]

    out: dict[str, float | str | bool] = {
        "n_pairs": len(direct),
        "mean_direct": float(np.mean(direct)) if len(direct) else np.nan,
        "mean_exploratory": float(np.mean(exploratory)) if len(exploratory) else np.nan,
        "mean_difference_exploratory_minus_direct": float(np.mean(exploratory - direct)) if len(direct) else np.nan,
        "t_statistic": np.nan,
        "df": np.nan,
        "p_value": np.nan,
        "significant": False,
        "winner": "not tested",
    }

    if len(direct) < 2:
        out["winner"] = "insufficient n"
        return out

    t_stat, p_value = stats.ttest_rel(exploratory, direct, nan_policy="omit")
    if not np.isfinite(t_stat) or not np.isfinite(p_value):
        out["winner"] = "test unavailable"
        return out

    df_val = len(direct) - 1
    significant = bool(p_value < 0.05)

    if not significant:
        winner = "no significant difference"
    elif float(np.mean(exploratory)) < float(np.mean(direct)):
        winner = "exploration lower score"
    else:
        winner = "direct refinement lower score"

    out.update(
        {
            "t_statistic": float(t_stat),
            "df": float(df_val),
            "p_value": float(p_value),
            "significant": significant,
            "winner": winner,
        }
    )
    return out


# ---------------------------------------------------------------------
# Plot 2: paired direct vs exploratory comparison
# ---------------------------------------------------------------------

def plot_paired_slope_scores_scientific(long_pairs: pd.DataFrame, wide_pairs: pd.DataFrame, out_dir: Path) -> None:
    """Plot 2: matched direct-vs-exploratory comparison.

    The plot uses a compact paired design:
    - thin gray lines connect matched cases;
    - colored dots show individual final scores;
    - a paired t-test is reported in the legend.

    Lower histogram score is better.
    """
    pairs = extract_direct_exploratory_pairs(long_pairs, wide_pairs)
    if pairs.empty:
        print("[WARN] Skipping plot 2: no matched direct/exploratory score pairs found.")
        return

    pairs = pairs.sort_values("direct_score").reset_index(drop=True)
    test = paired_test_direct_vs_exploratory(pairs)

    direct_x = 0.0
    exploratory_x = 0.75

    fig, ax = plt.subplots(figsize=(7.6, 4.3))

    # Matched case lines.
    for _, row in pairs.iterrows():
        ax.plot(
            [direct_x, exploratory_x],
            [float(row["direct_score"]), float(row["exploratory_score"])],
            color="0.78",
            alpha=0.55,
            linewidth=0.7,
            zorder=1,
        )

    # Individual observations.
    ax.scatter(
        np.full(len(pairs), direct_x),
        pairs["direct_score"],
        s=28,
        color=METHOD_COLORS.get("direct", "#0072B2"),
        edgecolor="white",
        linewidth=0.45,
        alpha=0.88,
        zorder=3,
    )
    ax.scatter(
        np.full(len(pairs), exploratory_x),
        pairs["exploratory_score"],
        s=28,
        color=METHOD_COLORS.get("exploratory", "#D55E00"),
        edgecolor="white",
        linewidth=0.45,
        alpha=0.88,
        zorder=3,
    )

    p_value = float(test["p_value"])
    p_text = "p n/a" if not np.isfinite(p_value) else ("p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}")
    df_value = float(test["df"])
    df_text = "df n/a" if not np.isfinite(df_value) else f"df = {df_value:.1f}"
    t_value = float(test["t_statistic"])
    t_text = "t n/a" if not np.isfinite(t_value) else f"t = {t_value:.2f}"

    n_direct = int(test.get("n_direct", len(pairs)))
    n_text = f"n = {n_direct} matched pairs"


    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=METHOD_COLORS.get("direct", "#0072B2"),
            markeredgecolor="white",
            markersize=7,
            label="Direct refinement",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=METHOD_COLORS.get("exploratory", "#D55E00"),
            markeredgecolor="white",
            markersize=7,
            label="Exploration + refinement",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            label="Paired t-test: equal mean\nwithin-pair difference",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            label=f"{n_text}, {t_text}, {df_text}, {p_text}",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        title="Method and test",
        loc="center left",
        bbox_to_anchor=(1.02, 0.50),
        frameon=True,
        framealpha=0.95,
        borderpad=0.8,
        handlelength=1.4,
    )

    ax.set_title("Matched seed pairs: paired t-test on final scores", loc="left", fontweight="bold")
    ax.set_ylabel("Final histogram score")
    ax.set_xticks([direct_x, exploratory_x])
    ax.set_xticklabels(["Direct\nrefinement", "Exploration +\nrefinement"])
    ax.set_xlim(-0.22, 0.97)

    ymin = max(0.0, min(pairs["direct_score"].min(), pairs["exploratory_score"].min()) * 0.92)
    ymax = max(pairs["direct_score"].max(), pairs["exploratory_score"].max()) * 1.10
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        ax.set_ylim(ymin, ymax)

    ax.tick_params(axis="x", pad=8)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=[0, 0, 0.74, 1])
    savefig(fig, out_dir, "02_matched_direct_vs_exploratory_slopeplot.png")



def select_contact_sheet_case(df: pd.DataFrame, paired_df: pd.DataFrame) -> pd.DataFrame:
    """Pick the worst matched case and return the full rows needed for plotting."""
    full = completed_runs(df).copy()
    paired = completed_runs(paired_df).copy()
    if full.empty or paired.empty:
        return pd.DataFrame()

    id_candidates = [
        "pair_id",
        "matched_id",
        "case_id",
        "image_id",
        "seed",
        "run_seed",
        "degradation_seed",
        "input_id",
    ]
    shared_id_col = next((c for c in id_candidates if c in full.columns and c in paired.columns), None)
    if shared_id_col is None:
        raise SystemExit("No shared identifier column found between df and paired_df.")

    # Rank cases using the paired table.
    case_scores = (
        paired[paired[shared_id_col].notna()]
        .groupby(shared_id_col)["final_score"]
        .agg(case_worst_score="max", case_mean_score="mean", n_rows="size")
        .reset_index()
        .sort_values(
            ["case_worst_score", "case_mean_score", "n_rows"],
            ascending=[False, False, False],
            kind="mergesort",
        )
    )

    if case_scores.empty:
        return pd.DataFrame()

    worst_case_id = case_scores.iloc[0][shared_id_col]

    # Pull the actual plot rows from the full table so filename/image columns survive.
    out = full[
        (full[shared_id_col] == worst_case_id)
        & (full["method"].astype(str).isin(["direct", "exploratory"]))
    ].copy()

    if out.empty:
        return pd.DataFrame()

    # Keep the single worst row per method within the selected case.
    out = out.sort_values(["method", "final_score"], ascending=[True, False], kind="mergesort")
    out = out.drop_duplicates(subset=["method"], keep="first").copy()

    out["method"] = pd.Categorical(
        out["method"].astype(str),
        categories=["direct", "exploratory"],
        ordered=True,
    )
    out = out.sort_values("method", kind="mergesort").reset_index(drop=True)

    print(f"[INFO] Selected worst contact-sheet case: {shared_id_col}={worst_case_id}")
    cols = [
        c for c in [shared_id_col, "method", "final_score", "initial_score", "absolute_improvement"]
        if c in out.columns
    ]
    print("[INFO] Contact-sheet rows:")
    print(out[cols].to_string(index=False) if cols else out.to_string(index=False))

    return out



def print_selected_contact_sheet_datapoints(df: pd.DataFrame) -> None:
    """Print the datapoints used in the contact sheet."""
    preferred_cols = [
        "pair_id",
        "matched_id",
        "case_id",
        "image_id",
        "seed",
        "run_seed",
        "degradation_seed",
        "input_id",
        "method",
        "initial_score",
        "final_score",
        "score_delta",
        "absolute_improvement",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if not cols:
        print("[INFO] Selected contact-sheet rows:")
        print(df.to_string(index=False))
        return

    print("[INFO] Worst contact-sheet datapoints:")
    print(df[cols].to_string(index=False))

def print_selected_contact_sheet_datapoints(df: pd.DataFrame) -> None:
    """Print the datapoints used in the contact sheet."""
    preferred_cols = [
        "pair_id", "matched_id", "case_id", "image_id", "seed",
        "run_seed", "degradation_seed", "input_id", "method",
        "initial_score", "final_score", "score_delta", "absolute_improvement",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if not cols:
        print("[INFO] Selected contact-sheet rows:")
        print(df.to_string(index=False))
        return

    print("[INFO] Worst contact-sheet datapoints:")
    print(df[cols].to_string(index=False))


# ---------------------------------------------------------------------
# Figure 5: cost vs quality without boxplots
# ---------------------------------------------------------------------

def plot_effort_quality_scatter(long_pairs: pd.DataFrame, wide_pairs: pd.DataFrame, out_dir: Path) -> None:
    """Figure 5: cost-effectiveness on the same matched pairs as Figure 2."""
    if wide_pairs is None or wide_pairs.empty:
        print("[WARN] Skipping figure 5: wide_pairs is empty.")
        return

    def find_col(columns: Iterable[str], candidates: list[str]) -> str | None:
        colset = set(columns)
        for candidate in candidates:
            if candidate in colset:
                return candidate
        return None

    columns = wide_pairs.columns

    direct_final_col = find_col(
        columns,
        [
            "final_score_direct",
            "direct_final_score",
            "score_direct",
            "direct_score",
            "final_score_non_exploratory",
            "non_exploratory_final_score",
            "score_non_exploratory",
            "non_exploratory_score",
            "final_score_non-exploratory",
            "score_non-exploratory",
        ],
    )
    exploratory_final_col = find_col(
        columns,
        [
            "final_score_exploratory",
            "exploratory_final_score",
            "score_exploratory",
            "exploratory_score",
        ],
    )

    randomized_direct_col = find_col(
        columns,
        [
            "randomized_score_direct",
            "direct_randomized_score",
            "randomized_histogram_score_direct",
        ],
    )
    randomized_exploratory_col = find_col(
        columns,
        [
            "randomized_score_exploratory",
            "exploratory_randomized_score",
            "randomized_histogram_score_exploratory",
        ],
    )

    if direct_final_col is None or exploratory_final_col is None:
        print("[WARN] Skipping figure 5: could not find final-score columns in wide_pairs.")
        print(list(wide_pairs.columns))
        return
    if randomized_direct_col is None or randomized_exploratory_col is None:
        print("[WARN] Skipping figure 5: could not find randomized score columns in wide_pairs.")
        print(list(wide_pairs.columns))
        return

    pair_table = pd.DataFrame(
        {
            "randomized_direct": pd.to_numeric(wide_pairs[randomized_direct_col], errors="coerce"),
            "final_direct": pd.to_numeric(wide_pairs[direct_final_col], errors="coerce"),
            "randomized_exploratory": pd.to_numeric(wide_pairs[randomized_exploratory_col], errors="coerce"),
            "final_exploratory": pd.to_numeric(wide_pairs[exploratory_final_col], errors="coerce"),
        }
    ).dropna(subset=["randomized_direct", "final_direct", "randomized_exploratory", "final_exploratory"])

    pair_table = pair_table[
        (pair_table["randomized_direct"] > 0)
        & (pair_table["final_direct"] > 0)
        & (pair_table["randomized_exploratory"] > 0)
        & (pair_table["final_exploratory"] > 0)
    ].copy()

    if pair_table.empty:
        print("[WARN] Skipping figure 5: no complete matched pairs after filtering.")
        return

    n_pairs = len(pair_table)

    plot_data = pd.DataFrame(
        [
            {"method": "direct", "gain": row["randomized_direct"] - row["final_direct"]}
            for _, row in pair_table.iterrows()
        ]
        + [
            {"method": "exploratory", "gain": row["randomized_exploratory"] - row["final_exploratory"]}
            for _, row in pair_table.iterrows()
        ]
    )

    methods = ["direct", "exploratory"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rng = np.random.default_rng(17)
    x_positions = np.arange(len(methods), dtype=float)
    width = 0.34

    for x_idx, method in zip(x_positions, methods):
        y = plot_data.loc[plot_data["method"] == method, "gain"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if len(y) == 0:
            continue

        color = METHOD_COLORS.get(method, "0.35")
        jitter = rng.uniform(-width * 0.42, width * 0.42, size=len(y))

        parts = ax.violinplot(
            [y],
            positions=[x_idx],
            widths=width * 1.35,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.18)

        ax.scatter(
            np.full(len(y), x_idx) + jitter,
            y,
            s=28,
            color=color,
            alpha=0.70,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

        median = float(np.median(y))
        lo, hi = bootstrap_ci(y, np.median)
        ax.errorbar(
            x_idx,
            median,
            yerr=[[median - lo], [hi - median]],
            fmt="D",
            color="black",
            ecolor="black",
            elinewidth=1.1,
            capsize=4,
            markersize=5.2,
            zorder=4,
        )

        ax.text(
            x_idx,
            float(np.nanmax(y)),
            f"n={len(y)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(0, color="0.35", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_title("Histogram score improvement for both prompting methods on 57 matched pairs", loc="left", fontweight="bold")
    ax.set_ylabel("Histogram Score Decrease (positive = improvement)")
    ax.set_xlabel("")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS.get(m, m.replace("_", " ").title()) for m in methods], rotation=10, ha="right")

    y_all = plot_data["gain"].to_numpy(dtype=float)
    y_all = y_all[np.isfinite(y_all)]
    if len(y_all):
        lo, hi = np.percentile(y_all, [1, 99])
        span = hi - lo
        if np.isfinite(span) and span > 0:
            ax.set_ylim(lo - 0.12 * span, hi + 0.18 * span)

    handles = [
        Line2D([0], [0], marker="o", color="w", label="Direct refinement",
               markerfacecolor=METHOD_COLORS.get("direct", "#0072B2"),
               markeredgecolor="white", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Exploration + refinement",
               markerfacecolor=METHOD_COLORS.get("exploratory", "#D55E00"),
               markeredgecolor="white", markersize=7),
    ]
    ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=1,
        frameon=True,
        framealpha=0.95,
    )
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True, linestyle="--", alpha=0.28)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    savefig(fig, out_dir, "05_effectiveness_per_vlm_snapshot.png")




# ---------------------------------------------------------------------
# Figure 5b: cost vs quality (gain per VLM snapshot)
# ---------------------------------------------------------------------

def plot_cost_vs_quality(long_pairs: pd.DataFrame, out_dir: Path) -> None:
    """Plot histogram-score reduction per VLM observation for matched pairs."""
    required = {"method", "final_score", "randomized_score"}
    if long_pairs is None or long_pairs.empty or not required.issubset(long_pairs.columns):
        print("[WARN] Skipping cost-vs-quality plot: missing required columns.")
        return

    data = long_pairs[
        (long_pairs["status"] == "completed")
        & long_pairs["final_score"].notna()
        & long_pairs["randomized_score"].notna()
    ].copy()

    if data.empty:
        print("[WARN] Skipping cost-vs-quality plot: no valid rows.")
        return

    data["method_norm"] = data["method"].astype(str).replace(
        {"non_exploratory": "direct", "non-exploratory": "direct"}
    )

    # Use match_key (set by matched_pairs()) as the pivot index; fall back to other id columns.
    id_candidates = ["match_key", "pair_id", "matched_id", "case_id", "image_id", "seed", "run_seed", "degradation_seed", "input_id"]
    id_cols = [c for c in id_candidates if c in data.columns]
    if not id_cols:
        ignored = {"method", "method_norm", "final_score", "randomized_score", "status", "absolute_improvement", "filter_adjustments", "vlm_snapshots"}
        id_cols = [c for c in data.columns if c not in ignored]

    # Fill missing vlm_snapshots with 1 so we don't lose matched pairs.
    snapshots = data["vlm_snapshots"].fillna(1).astype(float).clip(lower=1) if "vlm_snapshots" in data.columns else pd.Series(1.0, index=data.index)

    # Compute gain per VLM (clip denominator at 1 to handle 0-snapshot runs).
    data["gain_per_vlm"] = (
        data["randomized_score"].astype(float) - data["final_score"].astype(float)
    ) / snapshots

    pivot = data.pivot_table(
        index=id_cols,
        columns="method_norm",
        values="gain_per_vlm",
        aggfunc="mean",
    ).reset_index()

    if not {"direct", "exploratory"}.issubset(pivot.columns):
        print("[WARN] Skipping cost-vs-quality plot: could not pivot to direct/exploratory columns.")
        print(f"[INFO] method_norm values: {data['method_norm'].unique().tolist()}")
        return

    pivot = pivot.dropna(subset=["direct", "exploratory"])
    if pivot.empty:
        print("[WARN] Skipping cost-vs-quality plot: no complete matched pairs after pivot.")
        return

    # Melt back to long form for plotting.
    data = pd.DataFrame(
        [{"method": "direct", "gain_per_vlm": float(row["direct"])} for _, row in pivot.iterrows()]
        + [{"method": "exploratory", "gain_per_vlm": float(row["exploratory"])} for _, row in pivot.iterrows()]
    )

    methods = ["direct", "exploratory"]
    n_pairs = len(pivot)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rng = np.random.default_rng(17)
    x_positions = np.arange(len(methods), dtype=float)
    width = 0.34

    for x_idx, method in zip(x_positions, methods):
        y = data.loc[data["method"] == method, "gain_per_vlm"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if len(y) == 0:
            continue

        color = METHOD_COLORS.get(method, "0.35")
        jitter = rng.uniform(-width * 0.42, width * 0.42, size=len(y))

        parts = ax.violinplot(
            [y],
            positions=[x_idx],
            widths=width * 1.35,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.20)

        ax.scatter(
            np.full(len(y), x_idx) + jitter,
            y,
            s=26,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

        median = float(np.median(y))
        lo, hi = bootstrap_ci(y, np.median)
        ax.errorbar(
            x_idx,
            median,
            yerr=[[median - lo], [hi - median]],
            fmt="D",
            color="black",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
            markersize=5.5,
            zorder=5,
        )

        ax.text(
            x_idx,
            float(np.nanmax(y)) + 0.004,
            f"n={len(y)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(0, color="0.40", linewidth=0.9, linestyle="--", zorder=1)

    ax.set_title("Cost-effectiveness on matched restoration pairs", loc="left", fontweight="bold")
    ax.set_ylabel("Histogram-score reduction per VLM observation\n(higher is better)")
    ax.set_xlabel("")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [METHOD_LABELS.get(m, m.replace("_", " ").title()) for m in methods],
        rotation=10, ha="right",
    )

    y_all = data["gain_per_vlm"].to_numpy(dtype=float)
    y_all = y_all[np.isfinite(y_all)]
    if len(y_all):
        lo_p, hi_p = np.percentile(y_all, [1, 99])
        span = hi_p - lo_p
        if np.isfinite(span) and span > 0:
            ax.set_ylim(lo_p - 0.15 * span, hi_p + 0.22 * span)

    handles = [
        Line2D([0], [0], marker="o", color="w", label="Direct refinement",
               markerfacecolor=METHOD_COLORS.get("direct", "#0072B2"),
               markeredgecolor="white", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Exploration + refinement",
               markerfacecolor=METHOD_COLORS.get("exploratory", "#D55E00"),
               markeredgecolor="white", markersize=7),
        Line2D([0], [0], marker="D", color="w", label="Median + 95% CI",
               markerfacecolor="black", markeredgecolor="black", markersize=6),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
        framealpha=0.95,
    )

    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True, linestyle="--", alpha=0.28)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    savefig(fig, out_dir, "05_cost_vs_quality.png")


# ---------------------------------------------------------------------
# Figure 7: degraded-run trajectories
# ---------------------------------------------------------------------

def find_initial_score_column(df: pd.DataFrame) -> str | None:
    """Find or reconstruct the initial histogram score column."""
    for candidate in [
        "initial_score",
        "start_score",
        "starting_score",
        "degraded_score",
        "baseline_score",
        "score_before",
        "initial_histogram_score",
    ]:
        if candidate in df.columns:
            return candidate

    if {"final_score", "absolute_improvement"}.issubset(df.columns):
        df["initial_score_estimated"] = df["final_score"] + df["absolute_improvement"]
        return "initial_score_estimated"

    return None


def plot_degraded_run_score_trajectories(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot runs where the final histogram score is worse than the initial score.

    A degraded run is defined as final_score > initial_score because lower
    histogram score is better. Each line shows one run from its initial score
    to its final score. Upward lines therefore indicate that the restoration
    worsened the image according to the histogram score.
    """
    data = completed_runs(df)
    if data.empty:
        print("[WARN] Skipping degraded-run plot: no completed runs with final_score.")
        return

    initial_col = find_initial_score_column(data)
    if initial_col is None:
        print("[WARN] Skipping degraded-run plot: no initial score or absolute_improvement column found.")
        return

    required = {initial_col, "final_score", "method"}
    if not required.issubset(data.columns):
        print("[WARN] Skipping degraded-run plot: required columns missing.")
        return

    keep_cols = [initial_col, "final_score", "method"]
    if "vlm_snapshots" in data.columns:
        keep_cols.append("vlm_snapshots")
    plot_data = data[keep_cols].dropna(subset=[initial_col, "final_score", "method"]).copy()
    plot_data = plot_data[plot_data[initial_col] > 0].copy()
    if plot_data.empty:
        print("[WARN] Skipping degraded-run plot: no valid rows with positive initial score.")
        return

    plot_data["score_delta"] = plot_data["final_score"] - plot_data[initial_col]
    plot_data["relative_change_percent"] = (plot_data["score_delta"] / plot_data[initial_col]) * 100
    degraded = plot_data[plot_data["score_delta"] > 0].copy()

    if degraded.empty:
        print("[INFO] No degraded runs found: all final scores are <= initial scores.")
        return

    methods = ordered_methods(degraded["method"])

    fig, ax = plt.subplots(figsize=(7.4, 4.5))

    x_start = 0.0
    x_final = 0.85

    for method in methods:
        sub = degraded[degraded["method"].astype(str) == method].copy()
        if sub.empty:
            continue

        color = METHOD_COLORS.get(method, "0.35")

        for _, row in sub.iterrows():
            ax.plot(
                [x_start, x_final],
                [float(row[initial_col]), float(row["final_score"])],
                color=color,
                alpha=0.40,
                linewidth=0.9,
                zorder=1,
            )

        ax.scatter(
            np.full(len(sub), x_start),
            sub[initial_col],
            s=30,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.scatter(
            np.full(len(sub), x_final),
            sub["final_score"],
            s=30,
            color=color,
            alpha=0.78,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )

    total_completed = len(plot_data)
    total_degraded = len(degraded)
    degraded_pct = 100 * total_degraded / total_completed if total_completed else np.nan

    method_lines = []
    for method in methods:
        label = METHOD_LABELS.get(method, method.replace("_", " ").title())
        n_method = int((plot_data["method"].astype(str) == method).sum())
        n_degraded = int((degraded["method"].astype(str) == method).sum())
        pct = 100 * n_degraded / n_method if n_method else np.nan
        method_lines.append(f"{label}: {n_degraded}/{n_method} ({pct:.1f}%)")

    text = f"Degraded runs: {total_degraded}/{total_completed} ({degraded_pct:.1f}%)\n" + "\n".join(method_lines)
    ax.text(
        1.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.95,
        },
    )

    handles = []
    labels = []
    for method in methods:
        label = METHOD_LABELS.get(method, method.replace("_", " ").title())
        if label in labels:
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS.get(method, "0.35"),
                marker="o",
                markerfacecolor=METHOD_COLORS.get(method, "0.35"),
                markeredgecolor="white",
                linewidth=1.2,
                markersize=6,
                label=label,
            )
        )
        labels.append(label)

    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="0.55",
                markeredgecolor="white",
                markersize=6,
                label="Initial score",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="0.55",
                markeredgecolor="black",
                markersize=6,
                label="Final score",
            ),
        ]
    )

    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        title="Run type",
    )

    ax.set_title("Runs where restoration worsened", loc="left", fontweight="bold")
    ax.set_ylabel("Histogram score\n(lower is better)")
    ax.set_xticks([x_start, x_final])
    ax.set_xticklabels(["Initial degraded\nimage", "Final restored\nimage"])
    ax.set_xlim(-0.20, 1.10)

    ymin = max(0.0, min(degraded[initial_col].min(), degraded["final_score"].min()) * 0.92)
    ymax = max(degraded[initial_col].max(), degraded["final_score"].max()) * 1.10
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        ax.set_ylim(ymin, ymax)

    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True, linestyle="--", alpha=0.28)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=[0, 0, 0.76, 1])
    savefig(fig, out_dir, "07_degraded_run_histogram_score_trajectories.png")





# ---------------------------------------------------------------------
# Summary CSVs
# ---------------------------------------------------------------------

def summarize(df: pd.DataFrame, long_pairs: pd.DataFrame, wide_pairs: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "case1_all_runs_tidy.csv", index=False)
    long_pairs.to_csv(out_dir / "case1_matched_pairs_long.csv", index=False)
    wide_pairs.to_csv(out_dir / "case1_matched_pairs_wide.csv", index=False)

    completed = df[(df["status"] == "completed") & df["final_score"].notna()].copy()
    summary = []
    for method, group in completed.groupby("method"):
        summary.append(
            {
                "method": method,
                "n_runs": len(group),
                "median_final_score": group["final_score"].median(),
                "mean_final_score": group["final_score"].mean(),
                "median_absolute_improvement": group["absolute_improvement"].median() if "absolute_improvement" in group else np.nan,
                "mean_absolute_improvement": group["absolute_improvement"].mean() if "absolute_improvement" in group else np.nan,
                "median_filter_adjustments": group["filter_adjustments"].median() if "filter_adjustments" in group else np.nan,
                "median_vlm_snapshots": group["vlm_snapshots"].median() if "vlm_snapshots" in group else np.nan,
            }
        )

    stats = pd.DataFrame(summary)

    pairs = extract_direct_exploratory_pairs(long_pairs, wide_pairs)
    if not pairs.empty:
        paired = {
            "method": "matched_delta_exploratory_minus_direct",
            "n_runs": len(pairs),
            "median_final_score": pairs["delta_exploratory_minus_direct"].median(),
            "mean_final_score": pairs["delta_exploratory_minus_direct"].mean(),
            "exploratory_better_fraction": (pairs["delta_exploratory_minus_direct"] < 0).mean(),
        }
        paired_df = pd.DataFrame([paired])
        stats = paired_df if stats.empty else pd.concat([stats, paired_df], ignore_index=True)

    stats.to_csv(out_dir / "case1_summary_statistics.csv", index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-oriented figures for case study 1.")
    parser.add_argument("--input", required=True, help="Path to outputs/case_study_1")
    parser.add_argument("--output", default=None, help="Output folder for figures and CSVs. Defaults to <input>/figures_case1")
    parser.add_argument("--max-trajectory-runs", type=int, default=12)
    parser.add_argument("--max-contact-sheet-runs", type=int, default=6)
    args = parser.parse_args()

    set_publication_style()

    input_root = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve() if args.output else input_root / "figures_case1"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(input_root)
    if df.empty:
        raise SystemExit(f"No run manifests or multi-run summaries found under {input_root}")

    # Use the project matching logic as the source of truth for paired data.
    # All plots below use only rows from long_pairs, so the displayed n should
    # be 1 row per method per matched case, for example 57 direct + 57 exploratory.
    long_pairs, wide_pairs = matched_pairs(df)
    if long_pairs.empty or wide_pairs.empty:
        raise SystemExit(
            "No matched direct/exploratory pairs found. "
            "Refusing to create plots from unpaired data."
        )

    paired_df = long_pairs.copy()
    print(
        f"[INFO] Using matched-pairs subset only: "
        f"{len(wide_pairs)} matched cases, {len(paired_df)} paired run rows."
    )
    if "method" in paired_df.columns:
        print("[INFO] Paired rows by method:")
        print(paired_df["method"].value_counts(dropna=False).to_string())

    summarize(paired_df, long_pairs, wide_pairs, out_dir)

    # Updated publication-oriented plots.
    plot_final_score_estimation(long_pairs, wide_pairs, out_dir)
    plot_paired_slope_scores_scientific(long_pairs, wide_pairs, out_dir)

    # Existing plots that are still useful.
    plot_delta_histogram(wide_pairs, out_dir)
    plot_improvement_vs_start(paired_df, out_dir)

    # Figure 5 and degraded-run diagnostic, both restricted to matched pairs.
    plot_effort_quality_scatter(long_pairs, wide_pairs, out_dir)
    plot_cost_vs_quality(long_pairs, out_dir)
    plot_degraded_run_score_trajectories(paired_df, out_dir)

    # Existing supporting/qualitative plots, also restricted to matched pairs.
    plot_filter_trajectories(paired_df, out_dir, max_runs=args.max_trajectory_runs)


    worst_contact_sheet_df = select_contact_sheet_case(long_pairs, paired_df)
    if worst_contact_sheet_df.empty:
        print("[WARN] Skipping contact sheet: no worst matched case could be selected.")
    else:
        print_selected_contact_sheet_datapoints(worst_contact_sheet_df)
        worst_contact_sheet_df.to_csv(
            out_dir / "case1_worst_contact_sheet_examples.csv",
            index=False,
        )
        plot_image_contact_sheet(
            worst_contact_sheet_df,
            out_dir,
            max_runs=2,
        )
        
    print(f"Wrote figures and summary CSVs to: {out_dir}")
    if not wide_pairs.empty:
        print(f"Matched direct/exploratory cases: {len(wide_pairs)}")
    else:
        print("No matched direct/exploratory pairs found.")


if __name__ == "__main__":
    main()