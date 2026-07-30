from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

from .io import load_filter_trajectory, resolve_run_path
from .style import METHOD_LABELS, PALETTE, save_figure, set_paper_style

START_COLOR = "#D62728"  # red
FINAL_COLOR = "#2CA02C"  # green
DELTA_GOOD_COLOR = "#2CA02C"  # score decreased
DELTA_BAD_COLOR = "#D62728"   # score increased


def _method_order(df: pd.DataFrame) -> list[str]:
    return [m for m in ["direct", "exploratory"] if m in set(df["method"])]


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[(df["status"] == "completed") & df["final_score"].notna()].copy()


def _best_runs(df: pd.DataFrame, max_runs: int) -> pd.DataFrame:
    rows = _completed(df)
    if rows.empty:
        return rows
    rows = rows[rows["run_dir"].notna()].copy()
    if rows.empty:
        return rows
    # Stable ordering shared by Figure 6 and Figure 7.
    sort_cols = [c for c in ["final_score", "randomized_score", "run_id"] if c in rows.columns]
    ascending = [True, False, True][: len(sort_cols)]
    return rows.sort_values(sort_cols, ascending=ascending, na_position="last").head(max_runs).copy()


def _short_name(row: pd.Series) -> str:
    for key in ("image", "sample_id", "dataset_image", "run_id"):
        value = row.get(key)
        if value is not None and not pd.isna(value):
            name = Path(str(value)).name
            return name or str(value)
    return "unknown"


def _fmt_score(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.3g}"
    except Exception:
        return "n/a"


def _rgba_cmap(color: str, name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [(1, 1, 1, 0), color])


def plot_score_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    data = _completed(df)
    if data.empty:
        return
    set_paper_style()
    methods = _method_order(data)
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    positions = np.arange(len(methods))
    values = [data.loc[data["method"] == m, "final_score"].dropna().values for m in methods]
    bp = ax.boxplot(values, positions=positions, widths=0.45, patch_artist=True, showfliers=False)
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(PALETTE[method])
        patch.set_alpha(0.25)
        patch.set_edgecolor(PALETTE[method])
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.2)
    rng = np.random.default_rng(7)
    for pos, vals, method in zip(positions, values, methods):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=14,
            alpha=0.75,
            color=PALETTE[method],
            edgecolor="white",
            linewidth=0.3,
        )
    ax.set_xticks(positions, [METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Optimization outcome by prompt strategy")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    save_figure(fig, out_dir, "fig01_score_distributions")
    plt.close(fig)


def plot_paired_scores(long_df: pd.DataFrame, wide_df: pd.DataFrame, out_dir: Path) -> None:
    if wide_df.empty or not {"final_score_direct", "final_score_exploratory"}.issubset(wide_df.columns):
        return

    set_paper_style()

    fig, ax = plt.subplots(figsize=(4.4, 2.6))

    for _, row in wide_df.iterrows():
        xs = [0, 1]
        ys = [row["final_score_direct"], row["final_score_exploratory"]]
        improved = row["final_score_exploratory"] < row["final_score_direct"]
        color = PALETTE["improvement"] if improved else "#999999"
        ax.plot(xs, ys, color=color, alpha=0.45, linewidth=0.9, zorder=1)

    ax.scatter(
        np.zeros(len(wide_df)),
        wide_df["final_score_direct"],
        color=PALETTE["direct"],
        s=18,
        zorder=2,
    )
    ax.scatter(
        np.ones(len(wide_df)),
        wide_df["final_score_exploratory"],
        color=PALETTE["exploratory"],
        s=18,
        zorder=2,
    )

    ax.set_xticks([0, 1], ["Direct", "Exploratory"], rotation=0)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("Score")
    ax.set_title("Matched cases isolating the effect of exploration")

    n = len(wide_df)
    better_pct = (
        100.0 * float(wide_df["exploratory_better"].mean())
        if "exploratory_better" in wide_df
        else np.nan
    )
    pct_label = f"{better_pct:.0f}%" if np.isfinite(better_pct) else "n/a"

    fig.text(
        0.9,
        0.55,
        f"Matched cases: n={n}\nExploratory better: {pct_label}",
        ha="left",
        va="center",
        fontsize=7,
    )

    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    save_figure(fig, out_dir, "fig02_matched_paired_scores")
    plt.close(fig)


def plot_delta_histogram(wide_df: pd.DataFrame, out_dir: Path) -> None:
    # Figure 3 is intentionally no longer generated.
    return


def plot_improvement_vs_start(df: pd.DataFrame, out_dir: Path) -> None:
    data = df[(df["status"] == "completed") & df["randomized_score"].notna() & df["absolute_improvement"].notna()].copy()
    if data.empty:
        return
    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    for method in _method_order(data):
        s = data[data["method"] == method]
        ax.scatter(
            s["randomized_score"],
            s["absolute_improvement"],
            s=18,
            color=PALETTE[method],
            alpha=0.8,
            label=METHOD_LABELS[method],
            edgecolor="white",
            linewidth=0.3,
        )
    x_hi = max(float(data["randomized_score"].max()), 0.0)
    y_hi = max(float(data["absolute_improvement"].max()), 0.0)
    hi = max(x_hi, y_hi)
    if hi > 0:
        ax.plot([0, hi], [0, hi], color="#666666", linestyle=":", linewidth=0.9, label="Perfect recovery")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Randomized start score")
    ax.set_ylabel("Absolute improvement")
    ax.set_title("Absolute improvement based on randomized score for exploratory and direct prompt")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.2, linewidth=0.6)
    save_figure(fig, out_dir, "fig04_improvement_vs_start")
    plt.close(fig)


def plot_cost_vs_quality(df: pd.DataFrame, out_dir: Path) -> None:
    data = _completed(df)
    if data.empty or "filter_adjustments" not in data or data["filter_adjustments"].isna().all():
        return
    data = data[data["filter_adjustments"].notna()].copy()
    if data.empty:
        return
    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    x_all = data["filter_adjustments"].astype(float)
    y_all = data["final_score"].astype(float)
    x_bins = np.arange(np.floor(x_all.min()) - 0.5, np.ceil(x_all.max()) + 1.5, 1.0)
    if len(x_bins) < 3:
        x_bins = np.linspace(x_all.min() - 0.5, x_all.max() + 0.5, 4)
    y_bins = np.linspace(max(0.0, y_all.min()), y_all.max(), min(12, max(5, int(np.sqrt(len(data))) + 3)))
    if len(np.unique(y_bins)) < 3:
        y_bins = np.linspace(max(0.0, y_all.min() - 0.01), y_all.max() + 0.01, 5)

    for method in _method_order(data):
        s = data[data["method"] == method]
        if s.empty:
            continue
        cmap = _rgba_cmap(PALETTE[method], f"case1_{method}_hist2d")
        ax.hist2d(
            s["filter_adjustments"].astype(float),
            s["final_score"].astype(float),
            bins=[x_bins, y_bins],
            cmap=cmap,
            alpha=0.58,
            cmin=1,
        )
    handles = [Patch(facecolor=PALETTE[m], alpha=0.58, label=METHOD_LABELS[m]) for m in _method_order(data)]
    if handles:
        ax.legend(handles=handles, frameon=False, loc="best")
    ax.set_xlabel("Number of filter adjustments")
    ax.set_ylabel("Score")
    ax.set_title("Score distribution by filter adjustment count")
    ax.grid(alpha=0.2, linewidth=0.6)
    save_figure(fig, out_dir, "fig05_cost_vs_quality")
    plt.close(fig)


def _trajectory_scores(row: pd.Series, traj: pd.DataFrame) -> list[float | None]:
    if "score" in traj.columns and traj["score"].notna().any():
        return [float(v) if not pd.isna(v) else None for v in traj["score"]]
    return [None for _ in range(len(traj))]


def _candidate_delta_points(row: pd.Series, traj: pd.DataFrame) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    scores = _trajectory_scores(row, traj)
    prev_score = row.get("randomized_score")
    if prev_score is None or pd.isna(prev_score):
        prev_score = None
    else:
        prev_score = float(prev_score)

    for (_, trow), score in zip(traj.iterrows(), scores):
        if score is None or prev_score is None:
            prev_score = score if score is not None else prev_score
            continue
        if pd.isna(trow.get("brightness")) or pd.isna(trow.get("contrast")):
            prev_score = score
            continue
        delta = float(score) - float(prev_score)
        points.append((float(trow["brightness"]), float(trow["contrast"]), delta))
        prev_score = score

    # If per-step scores are unavailable, at least annotate the overall score change at the final marker.
    if not points:
        randomized = row.get("randomized_score")
        final = row.get("final_score")
        fb = row.get("final_brightness")
        fc = row.get("final_contrast")
        if not any(pd.isna(v) for v in [randomized, final, fb, fc]):
            points.append((float(fb), float(fc), float(final) - float(randomized)))
    return points


def _annotate_deltas(ax: plt.Axes, points: Iterable[tuple[float, float, float]]) -> None:
    placed: list[tuple[float, float]] = []
    for x, y, delta in points:
        label = f"Δ{delta:+.3g}"
        color = DELTA_GOOD_COLOR if delta < 0 else DELTA_BAD_COLOR
        # Deterministic vertical staggering to reduce overlap without external dependencies.
        y_span = max(1.0, ax.get_ylim()[1] - ax.get_ylim()[0])
        x_span = max(1.0, ax.get_xlim()[1] - ax.get_xlim()[0])
        offset_y = 0.035 * y_span
        offset_x = 0.0
        for _ in range(20):
            candidate = (x + offset_x, y + offset_y)
            too_close = any(abs(candidate[0] - px) < 0.07 * x_span and abs(candidate[1] - py) < 0.06 * y_span for px, py in placed)
            if not too_close:
                break
            offset_y += 0.045 * y_span
            offset_x = -offset_x + (0.025 * x_span if offset_x <= 0 else 0)
        placed.append((x + offset_x, y + offset_y))
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + offset_x, y + offset_y),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=6,
            color=color,
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": color, "alpha": 0.75},
        )


def plot_filter_trajectories(df: pd.DataFrame, out_dir: Path, max_runs: int = 6) -> None:
    rows = _best_runs(df, max_runs=max_runs)
    if rows.empty:
        return
    set_paper_style(font_size=7.5)
    n = len(rows)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, max(2.2, 1.9 * nrows)), squeeze=False)
    plotted = 0
    for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
        traj = load_filter_trajectory(row)
        if traj is None or not {"brightness", "contrast"}.issubset(traj.columns):
            ax.set_visible(False)
            continue
        traj = traj.dropna(subset=["brightness", "contrast"]).copy()
        if traj.empty:
            ax.set_visible(False)
            continue
        x = traj["brightness"].astype(float)
        y = traj["contrast"].astype(float)
        method = row["method"]

        start_x = row.get("random_brightness")
        start_y = row.get("random_contrast")
        if not any(pd.isna(v) for v in [start_x, start_y]) and len(x) > 0:
            ax.plot([float(start_x), float(x.iloc[0])], [float(start_y), float(y.iloc[0])], color="#666666", linestyle="--", linewidth=0.8, alpha=0.8)

        ax.plot(x, y, marker="o", color=PALETTE[method], linewidth=1.0, markersize=3)
        if not any(pd.isna(v) for v in [start_x, start_y]):
            ax.scatter([float(start_x)], [float(start_y)], color=START_COLOR, s=34, marker="x", linewidths=1.4, label="Randomization start")
        fb = row.get("final_brightness")
        fc = row.get("final_contrast")
        if not any(pd.isna(v) for v in [fb, fc]):
            ax.scatter([float(fb)], [float(fc)], color=FINAL_COLOR, s=46, marker="*", label="Final result", zorder=4)

        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
        ax.set_xlabel("Brightness")
        ax.set_ylabel("Contrast")
        ax.set_title(f"{METHOD_LABELS[method]}\n{_short_name(row)}", fontsize=7)
        ax.grid(alpha=0.2, linewidth=0.5)
        _annotate_deltas(ax, _candidate_delta_points(row, traj))
        plotted += 1
    for ax in axes.ravel()[plotted:]:
        ax.set_visible(False)

    handles = [
        Line2D([0], [0], color=PALETTE["direct"], marker="o", linestyle="-", label="Direct"),
        Line2D([0], [0], color=PALETTE["exploratory"], marker="o", linestyle="-", label="Exploratory"),
        Line2D([0], [0], color=START_COLOR, marker="x", linestyle="None", markersize=6, label="Randomization start score"),
        Line2D([0], [0], color=FINAL_COLOR, marker="*", linestyle="None", markersize=8, label="Final score"),
        Line2D([0], [0], color="#666666", linestyle="--", label="Start to first adjustment"),
    ]
    fig.suptitle("Optimization gradient for best cases", fontsize=9, y=1.02)
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save_figure(fig, out_dir, "fig06_filter_trajectories")
    plt.close(fig)


def plot_image_contact_sheet(df: pd.DataFrame, out_dir: Path, max_runs: int = 6) -> None:
    import matplotlib.image as mpimg

    rows = _best_runs(df, max_runs=max_runs)
    if rows.empty:
        return
    columns = [
        ("image_randomized_start", "Randomized start"),
        ("image_final_result", "Final result"),
        ("image_reference_hidden", "Hidden reference"),
    ]
    set_paper_style(font_size=7)
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(6.9, 1.95 * len(rows)), squeeze=False)
    any_image = False
    for r, (_, row) in enumerate(rows.iterrows()):
        for c, (col, title) in enumerate(columns):
            ax = axes[r, c]
            path = resolve_run_path(row, row.get(col))
            ax.set_xticks([])
            ax.set_yticks([])
            if path is not None and path.exists():
                ax.imshow(mpimg.imread(path), cmap="gray")
                any_image = True
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                filename = _short_name(row)
                label = (
                    f"{filename}\n"
                    f"{METHOD_LABELS[row['method']]}\n"
                    f"score={_fmt_score(row.get('final_score'))}\n"
                    f"start={_fmt_score(row.get('randomized_score'))}"
                )
                ax.set_ylabel(label, rotation=0, ha="right", va="center")
    if any_image:
        save_figure(fig, out_dir, "fig07_best_run_contact_sheet")
    plt.close(fig)
