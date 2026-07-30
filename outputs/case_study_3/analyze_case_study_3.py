"""Analyze and visualize Case Study 3 batch/run outputs.

This script is intended to live in:
    /home/nielsbroekhuizen/projects/my-vscode-project/outputs/case_study_3/

It scans the Case Study 3 output tree, extracts metrics from batch manifests,
run manifests, and prediction JSON files, converts everything to pandas
DataFrames, recomputes aggregate statistics across runs, and saves CSV tables
plus matplotlib visualizations.

Typical usage
-------------
From the Case Study 3 output folder:
    python analyze_case_study_3.py

Analyze a specific batch:
    python analyze_case_study_3.py \
        --root /home/nielsbroekhuizen/projects/my-vscode-project/outputs/case_study_3 \
        --batch-id case3_batch_20260614_124012

Analyze all discovered batches/runs and write to a custom output directory:
    python analyze_case_study_3.py --all --out-dir analysis_outputs/all_case3

The output directory will contain:
    tables/*.csv
    tables/*.xlsx              (if openpyxl is installed)
    figures/*.png
    analysis_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


VARIANT_ORDER = [
    "L0_path_only",
    "L1_path_only",
    "L2_path_only",
    "L0_grid",
    "L1_grid",
    "L2_grid",
    "L0_grid_labels",
    "L1_grid_labels",
    "L2_grid_labels",
    # Backward compatibility with old CS3 outputs.
    "E1_path_only",
    "E2_path_grid",
    "E3_path_grid_labels",
]
VARIANT_LABELS = {
    "L0_path_only": "L0: No annotations",
    "L0_grid": "L0: Grid",
    "L0_grid_labels": "L0: Grid + labels",
    "L1_path_only": "L1: No annotations",
    "L1_grid": "L1: Grid",
    "L1_grid_labels": "L1: Grid + labels",
    "L2_path_only": "L2: No annotations",
    "L2_grid": "L2: Grid",
    "L2_grid_labels": "L2: Grid + labels",
    "E1_path_only": "E1: No annotations",
    "E2_path_grid": "E2: Grid",
    "E3_path_grid_labels": "E3: Grid + labels",
}

VISUAL_CONDITION_LABELS = {
    "path_only": "No annotations",
    "grid": "Grid",
    "grid_labels": "Grid + labels",
}

PLOT_VARIANT_ORDER = [
    "L0_path_only",
    "L0_grid",
    "L0_grid_labels",
    "L1_path_only",
    "L1_grid",
    "L1_grid_labels",
    "L2_path_only",
    "L2_grid",
    "L2_grid_labels",
    "E1_path_only",
    "E2_path_grid",
    "E3_path_grid_labels",
]

PRIMARY_PLOT_VARIANT_ORDER = [
    "L0_path_only",
    "L1_path_only",
    "L2_path_only",
    "L0_grid",
    "L1_grid",
    "L2_grid",
    "L0_grid_labels",
    "L1_grid_labels",
    "L2_grid_labels",
    "E1_path_only",
    "E2_path_grid",
    "E3_path_grid_labels",
]

DIFFICULTY_ORDER = ["straight", "easy", "medium", "hard", "very_hard", "unknown"]
DIFFICULTY_LABELS = {
    "straight": "Straight (0 turns)",
    "easy": "Easy (1 turn)",
    "medium": "Medium (2 turns)",
    "hard": "Hard (3 turns)",
    "very_hard": "Very hard (>3 turns)",
    "unknown": "Unknown",
}

# Metrics that are most useful for the paper/results section.
PRIMARY_METRICS = ["precision", "recall", "f1", "tile_accuracy", "exact_match"]
ERROR_METRICS = ["missed_tile_rate", "extra_tile_rate"]
ALL_METRICS = PRIMARY_METRICS + ERROR_METRICS

# Publication-oriented palette, aligned with the style used in CS1/CS2.
METRIC_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
DIFFICULTY_COLORS = {
    "straight": "#999999",
    "easy": "#0072B2",
    "medium": "#E69F00",
    "hard": "#009E73",
    "very_hard": "#D55E00",
    "unknown": "#999999",
}


def set_publication_style() -> None:
    """Apply a compact, publication-style matplotlib theme."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
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
            "grid.alpha": 0.28,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _pretty_variant_label(label: str) -> str:
    return str(label).replace(": ", "\n", 1)


def _variant_labels_for_plot(variants: list[str]) -> list[str]:
    return [_pretty_variant_label(VARIANT_LABELS.get(v, v)) for v in variants]


def _styled_legend(ax: plt.Axes, **kwargs):
    defaults = {
        "frameon": True,
        "fancybox": True,
        "framealpha": 1.0,
        "facecolor": "white",
        "edgecolor": "black",
        "borderpad": 0.7,
        "labelspacing": 0.45,
        "handlelength": 1.5,
        "handletextpad": 0.6,
        "columnspacing": 1.1,
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


def _style_bar_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.28)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _top_center_boxed_legend(ax: plt.Axes, *, title: str, ncol: int, bbox_to_anchor=(0.5, 1.18), **kwargs):
    """Place a boxed legend above the axes, keeping it separate from the data area."""
    return _styled_legend(
        ax,
        title=title,
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        borderaxespad=0.0,
        **kwargs,
    )


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path
    out_dir: Path
    tables_dir: Path
    figures_dir: Path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        if isinstance(value, str) and value.strip() == "":
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def normalize_metric_key(key: str) -> str:
    """Normalize metric names from run-level and aggregate-level JSON files."""
    if key == "tile_accuracy_percent":
        return "tile_accuracy"
    if key == "exact_match_rate":
        return "exact_match"
    return key


def variant_sort_key(variant: str) -> tuple[int, str]:
    try:
        return (VARIANT_ORDER.index(variant), variant)
    except ValueError:
        return (999, variant)


def difficulty_sort_key(difficulty: Any) -> tuple[int, str]:
    key = normalize_turn_difficulty(difficulty)
    try:
        return (DIFFICULTY_ORDER.index(key), key)
    except ValueError:
        return (999, key)


def normalize_turn_difficulty(value: Any, turn_count: Any = None) -> str:
    """Normalize path difficulty labels from manifests/predictions.

    Preferred source is the stored ``turn_difficulty`` field. If older outputs
    only contain ``turn_count``, infer the same categories used by the runner:
    0=straight, 1=easy, 2=medium, 3=hard, >3=very_hard.
    """
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "veryhard": "very_hard",
            "very_hard": "very_hard",
            "very_hard_(>3_turns)": "very_hard",
            "easy": "easy",
            "medium": "medium",
            "hard": "hard",
            "straight": "straight",
            "none": "unknown",
            "nan": "unknown",
            "": "unknown",
        }
        if text in aliases:
            return aliases[text]
        if text in DIFFICULTY_ORDER:
            return text

    tc = safe_float(turn_count)
    if tc is None:
        return "unknown"
    if tc == 0:
        return "straight"
    if tc == 1:
        return "easy"
    if tc == 2:
        return "medium"
    if tc == 3:
        return "hard"
    return "very_hard"


def extract_turn_metadata(*payloads: dict[str, Any] | None) -> dict[str, Any]:
    """Extract or infer original-cell turn metadata from available payloads."""
    turn_count = None
    turn_difficulty = None
    turn_tile_sequence_l0 = None
    turn_coordinate_system = None

    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if turn_count is None and payload.get("turn_count") is not None:
            turn_count = payload.get("turn_count")
        if turn_difficulty is None and payload.get("turn_difficulty") is not None:
            turn_difficulty = payload.get("turn_difficulty")
        if turn_tile_sequence_l0 is None and payload.get("turn_tile_sequence_l0") is not None:
            turn_tile_sequence_l0 = payload.get("turn_tile_sequence_l0")
        if turn_coordinate_system is None and payload.get("turn_coordinate_system") is not None:
            turn_coordinate_system = payload.get("turn_coordinate_system")

    difficulty_key = normalize_turn_difficulty(turn_difficulty, turn_count)
    return {
        "turn_count": safe_float(turn_count),
        "turn_difficulty": difficulty_key,
        "turn_difficulty_label": DIFFICULTY_LABELS.get(difficulty_key, difficulty_key),
        "turn_tile_sequence_l0": turn_tile_sequence_l0,
        "turn_coordinate_system": turn_coordinate_system,
    }


def infer_variant_metadata(variant: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Infer grid-level metadata from explicit payload fields or variant name."""
    payload = payload or {}
    grid_level = payload.get("grid_level")
    subdivision = payload.get("subdivision")
    visual_condition = payload.get("visual_condition")

    m = re.match(r"L(\d+)_(path_only|grid_labels|grid)$", str(variant))
    if m:
        if grid_level is None:
            grid_level = int(m.group(1))
        if visual_condition is None:
            visual_condition = m.group(2)

    if subdivision is None and grid_level is not None:
        try:
            subdivision = 2 ** int(grid_level)
        except Exception:
            subdivision = None

    return {
        "grid_level": safe_float(grid_level),
        "subdivision": safe_float(subdivision),
        "visual_condition": visual_condition,
        "visual_condition_label": VISUAL_CONDITION_LABELS.get(str(visual_condition), visual_condition),
    }

def discover_batch_dirs(root: Path, batch_id: str | None, include_all: bool) -> list[Path]:
    runs_dir = root / "runs"
    if batch_id:
        batch_dir = runs_dir / batch_id
        if not batch_dir.exists():
            raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
        return [batch_dir]

    batch_dirs = sorted(
        [p.parent for p in runs_dir.glob("case3_batch_*/batch_manifest.json")],
        key=lambda p: p.name,
    )
    if include_all:
        return batch_dirs
    return batch_dirs[-1:] if batch_dirs else []


def discover_standalone_run_dirs(root: Path) -> list[Path]:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return []
    run_dirs: list[Path] = []
    for manifest in runs_dir.rglob("run_manifest.json"):
        # Batch child runs are still useful, but this function is mostly for
        # non-batch runs. Keep everything; duplicate handling happens later.
        run_dirs.append(manifest.parent)
    return sorted(run_dirs, key=lambda p: str(p))


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
        if isinstance(value, dict):
            out.update(flatten_dict(value, new_key))
        else:
            out[new_key] = value
    return out


def parse_batch_aggregate_rows(batch_manifest_path: Path) -> list[dict[str, Any]]:
    """Rows from batch_manifest.json aggregate_per_variant.

    These are useful for validating the script's recomputed aggregates.
    """
    manifest = read_json(batch_manifest_path)
    batch_id = manifest.get("batch_id", batch_manifest_path.parent.name)
    rows: list[dict[str, Any]] = []

    variant_metadata = manifest.get("variant_metadata", {}) if isinstance(manifest.get("variant_metadata", {}), dict) else {}
    for variant, stats in manifest.get("aggregate_per_variant", {}).items():
        meta = infer_variant_metadata(variant, variant_metadata.get(variant, {}))
        row: dict[str, Any] = {
            "batch_id": batch_id,
            "source": str(batch_manifest_path),
            "variant": variant,
            "variant_label": VARIANT_LABELS.get(variant, variant),
            "model": manifest.get("model"),
            "n_runs_requested": manifest.get("n_runs_requested"),
            "n_runs_completed": manifest.get("n_runs_completed"),
            **meta,
        }
        row.update(stats)
        rows.append(row)
    return rows


def parse_run_manifest_rows(run_manifest_path: Path) -> list[dict[str, Any]]:
    """Rows from one run_manifest.json; one row per variant in that run."""
    manifest = read_json(run_manifest_path)
    run_id = manifest.get("run_id", run_manifest_path.parent.name)
    batch_id = run_id.split("_run_")[0] if "_run_" in run_id else None
    rows: list[dict[str, Any]] = []

    per_variant = manifest.get("metrics_summary", {}).get("per_variant", {})
    for variant, metrics in per_variant.items():
        if not isinstance(metrics, dict):
            continue
        variant_gt = (manifest.get("gt_by_variant", {}) or {}).get(variant, {})
        meta = infer_variant_metadata(variant, variant_gt)
        turn_meta = extract_turn_metadata(variant_gt, metrics, manifest)
        row = {
            "batch_id": batch_id,
            "run_id": run_id,
            "run_dir": str(run_manifest_path.parent),
            "source": str(run_manifest_path),
            "variant": variant,
            "variant_label": VARIANT_LABELS.get(variant, variant),
            "model": manifest.get("model"),
            "gt_tile_set": variant_gt.get("gt_cell_set", variant_gt.get("gt_tile_set", manifest.get("gt_cell_set", manifest.get("gt_tile_set")))),
            "gt_tile_sequence": variant_gt.get("gt_cell_sequence", variant_gt.get("gt_tile_sequence", manifest.get("gt_cell_sequence", manifest.get("gt_tile_sequence")))),
            "n_path_objects": manifest.get("n_path_objects"),
            "status": manifest.get("predictions", {}).get(variant),
            "cell_width": variant_gt.get("cell_width"),
            "cell_height": variant_gt.get("cell_height"),
            "effective_cols": variant_gt.get("effective_cols"),
            "effective_rows": variant_gt.get("effective_rows"),
            **meta,
            **turn_meta,
        }
        for key, value in metrics.items():
            norm_key = normalize_metric_key(key)

            # Run manifests may store per-run aggregate means as mean_* fields
            # from case_study_3.metrics.aggregate_variant_metrics(). Convert
            # those back to the plain metric names used by this analysis script
            # so across-run aggregation keeps working as before.
            if norm_key.startswith("mean_"):
                base_metric = normalize_metric_key(norm_key.replace("mean_", "", 1))
                if base_metric in ALL_METRICS or base_metric == "exact_match":
                    row[base_metric] = safe_float(value)
                else:
                    row[norm_key] = value

            # Run manifests may also store std_* fields. These are not the
            # across-run standard deviations computed later in this file; they
            # are the within-run standard deviations across paths/predictions.
            # Keep them explicitly named to avoid mixing both variability types.
            elif norm_key.startswith("std_"):
                base_metric = normalize_metric_key(norm_key.replace("std_", "", 1))
                row[f"inner_run_std_{base_metric}"] = safe_float(value)

            elif norm_key in ALL_METRICS or norm_key in {"exact_match"}:
                row[norm_key] = safe_float(value)
            else:
                row[norm_key] = value
        rows.append(row)
    return rows


def parse_prediction_rows(prediction_path: Path) -> list[dict[str, Any]]:
    """Rows from predictions/<variant>.json.

    This is a fallback and also useful for inspecting raw replies/predicted sets.
    """
    payload = read_json(prediction_path)
    if not isinstance(payload, dict):
        return []

    variant = payload.get("variant", prediction_path.stem)
    run_dir = prediction_path.parents[1] if prediction_path.parent.name == "predictions" else prediction_path.parent
    run_id = run_dir.name
    if re.match(r"run_\d{3}$", run_id) and run_dir.parent.name.startswith("case3_batch_"):
        run_id = f"{run_dir.parent.name}_{run_id}"
    batch_id = run_id.split("_run_")[0] if "_run_" in run_id else None

    meta = infer_variant_metadata(variant, payload)
    metrics = payload.get("metrics") or {}
    turn_meta = extract_turn_metadata(payload, metrics)
    row: dict[str, Any] = {
        "batch_id": batch_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source": str(prediction_path),
        "variant": variant,
        "variant_label": VARIANT_LABELS.get(variant, variant),
        "status": payload.get("status"),
        "raw_reply": payload.get("raw_reply"),
        "predicted_tile_sequence": payload.get("predicted_cell_sequence", payload.get("predicted_tile_sequence")),
        "predicted_tile_set": payload.get("predicted_cell_set", payload.get("predicted_tile_set")),
        "gt_tile_sequence": payload.get("gt_cell_sequence"),
        "gt_tile_set": payload.get("gt_cell_set"),
        "parsing_error": payload.get("parsing_error"),
        "image_path": payload.get("image_path"),
        "image_cache_hit": payload.get("image_cache_hit"),
        "cell_width": payload.get("cell_width"),
        "cell_height": payload.get("cell_height"),
        "effective_cols": payload.get("effective_cols"),
        "effective_rows": payload.get("effective_rows"),
        **meta,
        **turn_meta,
    }

    for key, value in metrics.items():
        norm_key = normalize_metric_key(key)
        row[norm_key] = safe_float(value) if norm_key in ALL_METRICS or norm_key == "exact_match" else value
    return [row]


def collect_dataframes(root: Path, batch_id: str | None, include_all: bool) -> dict[str, pd.DataFrame]:
    batch_dirs = discover_batch_dirs(root, batch_id=batch_id, include_all=include_all)
    if not batch_dirs and not include_all:
        # No batch manifests found; fall back to whatever run manifests exist.
        batch_dirs = []

    batch_aggregate_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    # Preferred path: selected batch directories.
    for batch_dir in batch_dirs:
        manifest_path = batch_dir / "batch_manifest.json"
        if manifest_path.exists():
            batch_aggregate_rows.extend(parse_batch_aggregate_rows(manifest_path))

        for run_manifest in sorted(batch_dir.rglob("run_manifest.json")):
            run_rows.extend(parse_run_manifest_rows(run_manifest))
        for pred_path in sorted(batch_dir.rglob("predictions/*.json")):
            prediction_rows.extend(parse_prediction_rows(pred_path))

    # Include non-batch or all run manifests when requested.
    if include_all or not run_rows:
        for run_dir in discover_standalone_run_dirs(root):
            run_manifest = run_dir / "run_manifest.json"
            if run_manifest.exists():
                run_rows.extend(parse_run_manifest_rows(run_manifest))
        for pred_path in sorted((root / "runs").rglob("predictions/*.json")):
            prediction_rows.extend(parse_prediction_rows(pred_path))

    run_df = pd.DataFrame(run_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    batch_aggregate_df = pd.DataFrame(batch_aggregate_rows)

    if not run_df.empty:
        # De-duplicate if the same run was collected through both batch and all-runs paths.
        subset = [c for c in ["run_id", "variant"] if c in run_df.columns]
        if subset:
            run_df = run_df.drop_duplicates(subset=subset, keep="first")
        run_df["variant"] = pd.Categorical(run_df["variant"], categories=VARIANT_ORDER, ordered=True)
        if "turn_difficulty" in run_df.columns:
            run_df["turn_difficulty"] = run_df["turn_difficulty"].map(normalize_turn_difficulty)
            run_df["turn_difficulty"] = pd.Categorical(run_df["turn_difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
        run_df = run_df.sort_values(["variant", "run_id"], na_position="last")

    if not prediction_df.empty:
        subset = [c for c in ["run_id", "variant"] if c in prediction_df.columns]
        if subset:
            prediction_df = prediction_df.drop_duplicates(subset=subset, keep="first")
        prediction_df["variant"] = pd.Categorical(prediction_df["variant"], categories=VARIANT_ORDER, ordered=True)
        if "turn_difficulty" in prediction_df.columns:
            prediction_df["turn_difficulty"] = prediction_df["turn_difficulty"].map(normalize_turn_difficulty)
            prediction_df["turn_difficulty"] = pd.Categorical(prediction_df["turn_difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
        prediction_df = prediction_df.sort_values(["variant", "run_id"], na_position="last")

    return {
        "run_level_metrics": run_df,
        "prediction_details": prediction_df,
        "batch_manifest_aggregates": batch_aggregate_df,
    }


def compute_aggregate_df(run_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute aggregate statistics across runs per variant."""
    if run_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    metric_cols = [m for m in ALL_METRICS if m in run_df.columns]

    for variant, group in run_df.groupby("variant", observed=True, sort=False):
        row: dict[str, Any] = {
            "variant": str(variant),
            "variant_label": VARIANT_LABELS.get(str(variant), str(variant)),
            "grid_level": group["grid_level"].dropna().iloc[0] if "grid_level" in group.columns and group["grid_level"].notna().any() else None,
            "subdivision": group["subdivision"].dropna().iloc[0] if "subdivision" in group.columns and group["subdivision"].notna().any() else None,
            "visual_condition": group["visual_condition"].dropna().iloc[0] if "visual_condition" in group.columns and group["visual_condition"].notna().any() else None,
            "visual_condition_label": group["visual_condition_label"].dropna().iloc[0] if "visual_condition_label" in group.columns and group["visual_condition_label"].notna().any() else None,
            "n_runs": int(group["run_id"].nunique()) if "run_id" in group.columns else int(len(group)),
            "n_rows": int(len(group)),
            "n_parsing_failures": int(group.get("parsing_error", pd.Series(dtype=object)).notna().sum())
            if "parsing_error" in group.columns else None,
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = values.mean() if len(values) else np.nan
            row[f"std_{metric}"] = values.std(ddof=0) if len(values) else np.nan
            row[f"sample_std_{metric}"] = values.std(ddof=1) if len(values) > 1 else np.nan
            row[f"var_{metric}"] = values.var(ddof=0) if len(values) else np.nan
            row[f"sample_var_{metric}"] = values.var(ddof=1) if len(values) > 1 else np.nan
            row[f"min_{metric}"] = values.min() if len(values) else np.nan
            row[f"max_{metric}"] = values.max() if len(values) else np.nan

            inner_col = f"inner_run_std_{metric}"
            if inner_col in group.columns:
                inner_values = pd.to_numeric(group[inner_col], errors="coerce").dropna()
                row[f"mean_inner_run_std_{metric}"] = inner_values.mean() if len(inner_values) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["variant"] = pd.Categorical(out["variant"], categories=VARIANT_ORDER, ordered=True)
        out = out.sort_values("variant")
    return out


def compute_difficulty_aggregate_df(run_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accuracy/quality metrics by path-turn difficulty only."""
    if run_df.empty or "turn_difficulty" not in run_df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    metric_cols = [m for m in ALL_METRICS if m in run_df.columns]
    df = run_df.copy()
    df["turn_difficulty"] = df["turn_difficulty"].map(normalize_turn_difficulty)

    for difficulty, group in df.groupby("turn_difficulty", observed=True, sort=False):
        difficulty_key = normalize_turn_difficulty(difficulty)
        row: dict[str, Any] = {
            "turn_difficulty": difficulty_key,
            "turn_difficulty_label": DIFFICULTY_LABELS.get(difficulty_key, difficulty_key),
            "n_runs": int(group["run_id"].nunique()) if "run_id" in group.columns else int(len(group)),
            "n_rows": int(len(group)),
            "n_variants": int(group["variant"].nunique()) if "variant" in group.columns else None,
            "mean_turn_count": pd.to_numeric(group.get("turn_count", pd.Series(dtype=float)), errors="coerce").mean(),
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = values.mean() if len(values) else np.nan
            row[f"std_{metric}"] = values.std(ddof=0) if len(values) else np.nan
            row[f"sample_std_{metric}"] = values.std(ddof=1) if len(values) > 1 else np.nan
            row[f"min_{metric}"] = values.min() if len(values) else np.nan
            row[f"max_{metric}"] = values.max() if len(values) else np.nan

            inner_col = f"inner_run_std_{metric}"
            if inner_col in group.columns:
                inner_values = pd.to_numeric(group[inner_col], errors="coerce").dropna()
                row[f"mean_inner_run_std_{metric}"] = inner_values.mean() if len(inner_values) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["turn_difficulty"] = pd.Categorical(out["turn_difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
        out = out.sort_values("turn_difficulty")
    return out


def compute_variant_difficulty_aggregate_df(run_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by variant and path-turn difficulty."""
    if run_df.empty or "turn_difficulty" not in run_df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    metric_cols = [m for m in ALL_METRICS if m in run_df.columns]
    df = run_df.copy()
    df["turn_difficulty"] = df["turn_difficulty"].map(normalize_turn_difficulty)

    grouped = df.groupby(["variant", "turn_difficulty"], observed=True, sort=False)
    for (variant, difficulty), group in grouped:
        variant_str = str(variant)
        difficulty_key = normalize_turn_difficulty(difficulty)
        row: dict[str, Any] = {
            "variant": variant_str,
            "variant_label": VARIANT_LABELS.get(variant_str, variant_str),
            "turn_difficulty": difficulty_key,
            "turn_difficulty_label": DIFFICULTY_LABELS.get(difficulty_key, difficulty_key),
            "grid_level": group["grid_level"].dropna().iloc[0] if "grid_level" in group.columns and group["grid_level"].notna().any() else None,
            "subdivision": group["subdivision"].dropna().iloc[0] if "subdivision" in group.columns and group["subdivision"].notna().any() else None,
            "visual_condition": group["visual_condition"].dropna().iloc[0] if "visual_condition" in group.columns and group["visual_condition"].notna().any() else None,
            "visual_condition_label": group["visual_condition_label"].dropna().iloc[0] if "visual_condition_label" in group.columns and group["visual_condition_label"].notna().any() else None,
            "n_runs": int(group["run_id"].nunique()) if "run_id" in group.columns else int(len(group)),
            "n_rows": int(len(group)),
            "mean_turn_count": pd.to_numeric(group.get("turn_count", pd.Series(dtype=float)), errors="coerce").mean(),
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = values.mean() if len(values) else np.nan
            row[f"std_{metric}"] = values.std(ddof=0) if len(values) else np.nan
            row[f"sample_std_{metric}"] = values.std(ddof=1) if len(values) > 1 else np.nan
            row[f"min_{metric}"] = values.min() if len(values) else np.nan
            row[f"max_{metric}"] = values.max() if len(values) else np.nan

            inner_col = f"inner_run_std_{metric}"
            if inner_col in group.columns:
                inner_values = pd.to_numeric(group[inner_col], errors="coerce").dropna()
                row[f"mean_inner_run_std_{metric}"] = inner_values.mean() if len(inner_values) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["variant"] = pd.Categorical(out["variant"], categories=VARIANT_ORDER, ordered=True)
        out["turn_difficulty"] = pd.Categorical(out["turn_difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
        out = out.sort_values(["variant", "turn_difficulty"])
    return out



def add_mean_inner_batch_std_columns(run_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Add mean within-batch std columns to an aggregate dataframe.

    For each batch and for the grouping columns present in target_df, compute
    the population standard deviation across repeated runs inside that batch.
    Then average those batch-level std values over all batches for the same
    output group.

    Examples
    --------
    aggregate_df groups by variant:
        batch_id x variant -> std across runs -> mean over batches

    difficulty_df groups by turn_difficulty:
        batch_id x turn_difficulty -> std across runs/variants -> mean over batches

    variant_difficulty_df groups by variant and turn_difficulty:
        batch_id x variant x turn_difficulty -> std across runs -> mean over batches

    This is different from existing std_* columns, which are population
    standard deviations after pooling all run-level rows together.
    """
    if run_df.empty or target_df.empty:
        return target_df

    metric_cols = [m for m in ALL_METRICS if m in run_df.columns]
    if not metric_cols:
        return target_df

    df = run_df.copy()

    if "batch_id" not in df.columns or df["batch_id"].isna().all():
        if "run_id" not in df.columns:
            return target_df
        df["batch_id"] = df["run_id"].astype(str).str.replace(r"_run_\d+$", "", regex=True)

    if "turn_difficulty" in df.columns:
        df["turn_difficulty"] = df["turn_difficulty"].map(normalize_turn_difficulty)

    # Match the grouping level of target_df. The previous version always
    # assumed target_df had a variant column, which crashes for difficulty_df.
    final_cols: list[str] = []
    for col in ["variant", "turn_difficulty"]:
        if col in df.columns and col in target_df.columns:
            final_cols.append(col)

    if not final_cols:
        return target_df

    group_cols = ["batch_id", *final_cols]

    batch_rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs_in_batch"] = int(group["run_id"].nunique()) if "run_id" in group.columns else int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            # ddof=0: population std inside this batch of repeated runs.
            row[f"inner_batch_std_{metric}"] = values.std(ddof=0) if len(values) else np.nan
        batch_rows.append(row)

    batch_std_df = pd.DataFrame(batch_rows)
    if batch_std_df.empty:
        return target_df

    mean_rows: list[dict[str, Any]] = []
    for keys, group in batch_std_df.groupby(final_cols, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(final_cols, keys))
        row["n_batches_for_inner_batch_std"] = int(group["batch_id"].nunique()) if "batch_id" in group.columns else int(len(group))
        for metric in metric_cols:
            col = f"inner_batch_std_{metric}"
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"mean_inner_batch_std_{metric}"] = values.mean() if len(values) else np.nan
        mean_rows.append(row)

    mean_std_df = pd.DataFrame(mean_rows)
    if mean_std_df.empty:
        return target_df

    out = target_df.copy()
    for col in final_cols:
        out[col] = out[col].astype(str)
        mean_std_df[col] = mean_std_df[col].astype(str)
    out = out.merge(mean_std_df, on=final_cols, how="left")

    if "variant" in out.columns:
        out["variant"] = pd.Categorical(out["variant"], categories=VARIANT_ORDER, ordered=True)
    if "turn_difficulty" in out.columns:
        out["turn_difficulty"] = pd.Categorical(out["turn_difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
    return out

def make_long_metrics_df(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    id_cols = [c for c in ["batch_id", "run_id", "variant", "variant_label", "grid_level", "subdivision", "visual_condition", "visual_condition_label", "turn_count", "turn_difficulty", "turn_difficulty_label", "model", "status"] if c in run_df.columns]
    value_cols = [m for m in ALL_METRICS if m in run_df.columns]
    long_df = run_df.melt(id_vars=id_cols, value_vars=value_cols, var_name="metric", value_name="value")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    return long_df.dropna(subset=["value"])


def save_tables(paths: AnalysisPaths, dfs: dict[str, pd.DataFrame]) -> None:
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        if df is None or df.empty:
            continue
        df.to_csv(paths.tables_dir / f"{name}.csv", index=False)

    # Excel is convenient for thesis/paper work, but keep it optional.
    xlsx_path = paths.tables_dir / "case_study_3_analysis.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, df in dfs.items():
                if df is not None and not df.empty:
                    sheet = name[:31]
                    df.to_excel(writer, sheet_name=sheet, index=False)
    except Exception as exc:
        print(f"[warn] Could not write Excel workbook ({exc}). CSV files were still written.")


def _metric_title(metric: str) -> str:
    return metric.replace("_", " ").title().replace("Tile", "Cell")


def _variant_condition_key(variant: str) -> str:
    variant = str(variant)
    if variant.endswith("_path_only"):
        return "path_only"
    if variant.endswith("_grid_labels"):
        return "grid_labels"
    if variant.endswith("_grid"):
        return "grid"
    return "other"


def _variant_condition_short_label(variant: str) -> str:
    return VISUAL_CONDITION_LABELS.get(_variant_condition_key(variant), str(variant))


def _variant_grid_level_label(variant: str) -> str:
    m = re.match(r"(L\d+|E\d+)_", str(variant))
    return m.group(1) if m else "Other"


def _grouped_variant_label(variant: str) -> str:
    return f"{_variant_grid_level_label(variant)}\n{_variant_condition_short_label(variant)}"


def _ordered_variants_for_plot(variants: Iterable[str]) -> list[str]:
    preferred = [v for v in PLOT_VARIANT_ORDER if v in set(map(str, variants))]
    remaining = [v for v in map(str, variants) if v not in preferred]
    return preferred + sorted(remaining, key=variant_sort_key)


def _primary_ordered_variants_for_plot(variants: Iterable[str]) -> list[str]:
    preferred = [v for v in PRIMARY_PLOT_VARIANT_ORDER if v in set(map(str, variants))]
    remaining = [v for v in map(str, variants) if v not in preferred]
    return preferred + sorted(remaining, key=variant_sort_key)


def _subset_metric_df(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if long_df.empty or "metric" not in long_df.columns:
        return pd.DataFrame()
    df = long_df[long_df["metric"].astype(str) == metric].copy()
    if df.empty:
        return df
    df["variant"] = df["variant"].astype(str)
    return df


def _panel_group_separators(ax: plt.Axes, n_variants: int) -> None:
    # Draw separators between L0/L1/L2 blocks when the grouped order is used.
    for xpos in (2.5, 5.5):
        if xpos < n_variants - 0.5:
            ax.axvline(xpos, color="0.82", linewidth=1.0, zorder=1)


def _annotate_repeat_note(ax: plt.Axes) -> None:
    ax.text(
        0.99,
        0.98,
        "3 repeated runs per setting",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.35",
    )


def plot_primary_metric_panels(aggregate_df: pd.DataFrame, long_df: pd.DataFrame, figures_dir: Path) -> None:
    """Show the primary metrics as point-range panels with raw run points.

    The x-axis is ordered as three blocks of L0/L1/L2:
    path-only, grid, then grid+labels.
    """
    if aggregate_df.empty:
        return

    metrics = [m for m in PRIMARY_METRICS if f"mean_{m}" in aggregate_df.columns]
    if not metrics:
        return

    variants = _primary_ordered_variants_for_plot(aggregate_df["variant"].astype(str).tolist())
    if not variants:
        return

    n_panels = len(metrics)
    ncols = 3
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.2, 4.45 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    agg = aggregate_df.copy()
    agg["variant"] = agg["variant"].astype(str)
    agg = agg.set_index("variant", drop=False)

    condition_order = ["path_only", "grid", "grid_labels"]
    condition_colors = {
        "path_only": "#0072B2",
        "grid": "#009E73",
        "grid_labels": "#D55E00",
    }
    condition_handles = [
        Line2D(
            [0], [0],
            color=condition_colors[cond],
            marker="o",
            linewidth=1.5,
            markersize=4.5,
            label=VISUAL_CONDITION_LABELS[cond],
        )
        for cond in condition_order
    ]

    block_centers = [1, 4, 7]
    block_titles = ["No annotations", "Grid", "Grid + labels"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        mean_col = f"mean_{metric}"
        err_col = f"mean_inner_batch_std_{metric}" if f"mean_inner_batch_std_{metric}" in agg.columns else f"std_{metric}"

        subset = agg.reindex(variants)
        x = np.arange(len(variants))
        means = pd.to_numeric(subset[mean_col], errors="coerce").to_numpy(dtype=float)
        errs = (
            pd.to_numeric(subset[err_col], errors="coerce").fillna(0).to_numpy(dtype=float)
            if err_col in subset.columns
            else np.zeros_like(means)
        )

        for condition in condition_order:
            block_variants = [f"L0_{condition}", f"L1_{condition}", f"L2_{condition}"]
            block_positions = [variants.index(v) for v in block_variants if v in variants]
            if len(block_positions) < 2:
                continue

            block_means = [means[pos] for pos in block_positions]
            block_errs = [errs[pos] for pos in block_positions]
            ax.errorbar(
                block_positions,
                block_means,
                yerr=block_errs,
                fmt="o-",
                color=condition_colors[condition],
                linewidth=1.5,
                markersize=4.5,
                capsize=4,
                zorder=3,
            )

        raw = _subset_metric_df(long_df, metric)
        if not raw.empty:
            raw = raw[raw["variant"].isin(variants)]
            for xi, variant in enumerate(variants):
                values = pd.to_numeric(
                    raw.loc[raw["variant"] == variant, "value"],
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue

                jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
                condition = _variant_condition_key(variant)
                ax.scatter(
                    np.full(len(values), xi) + jitter,
                    values,
                    s=18,
                    alpha=0.55,
                    color=condition_colors.get(condition, "#0072B2"),
                    edgecolors="none",
                    zorder=2,
                )

        ax.set_title(_metric_title(metric), loc="left", fontweight="bold", fontsize=13, pad=6)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels([_variant_grid_level_label(v) for v in variants], rotation=0, ha="center")
        ax.tick_params(axis="x", pad=1)
        _panel_group_separators(ax, len(variants))
        _style_bar_axes(ax)

        for xpos, title in zip(block_centers, block_titles):
            ax.text(
                xpos,
                0.965,
                title,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8.7,
                fontweight="bold",
            )

    legend_ax = axes[n_panels] if n_panels < len(axes) else axes[-1]
    legend_ax.axis("off")

    n_note = None
    if "n_runs" in agg.columns:
        n_runs = pd.to_numeric(agg["n_runs"], errors="coerce")
        if n_runs.notna().any():
            n_note = int(round(n_runs.dropna().max()))

    note_handles = [
        Line2D([], [], linestyle="none", marker=None, label="3 repeated runs per setting"),
    ]
    if n_note is not None:
        note_handles.append(Line2D([], [], linestyle="none", marker=None, label=f"n={n_note} per variant"))

    legend_ax.legend(
        handles=condition_handles + note_handles,
        title="Legend",
        loc="center",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=14,
        title_fontsize=18,
    )

    for j in range(n_panels + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Case Study 3 primary metrics by visualization condition",
        x=0.01,
        y=0.975,
        ha="left",
        fontweight="bold",
        fontsize=24,
    )

    fig.subplots_adjust(top=0.90, hspace=0.52, wspace=0.30)
    fig.tight_layout(rect=[0, 0, 1, 0.915], h_pad=0.55, w_pad=0.8)
    fig.savefig(figures_dir / "primary_metrics_point_range.png", dpi=300)
    plt.close(fig)



def _make_axes_fonts_bold_and_larger(ax: plt.Axes, *, label_size: int = 16, tick_size: int = 13) -> None:
    """Make axis labels and tick labels larger/bold for presentation figures."""
    ax.xaxis.label.set_size(label_size)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_size(label_size)
    ax.yaxis.label.set_weight("bold")

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(tick_size)
        tick_label.set_fontweight("bold")


def plot_f1_accuracy_exact_match_panels(aggregate_df: pd.DataFrame, long_df: pd.DataFrame, figures_dir: Path) -> None:
    """Create an extra primary-metric figure for F1, cell accuracy, and exact match only.

    This intentionally does not replace ``primary_metrics_point_range.png``.
    It uses the same point-range idea as the primary metric figure, but omits
    precision/recall and does not add a main figure title.
    """
    if aggregate_df.empty:
        return

    metrics = [m for m in ["f1", "tile_accuracy", "exact_match"] if f"mean_{m}" in aggregate_df.columns]
    if not metrics:
        return

    variants = _primary_ordered_variants_for_plot(aggregate_df["variant"].astype(str).tolist())
    if not variants:
        return

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(17.2, 7.2),
        sharex=True,
        sharey=True,
    )

    if len(metrics) == 1:
        axes = [axes]
    else:
        axes = np.array(axes).reshape(-1)

    agg = aggregate_df.copy()
    agg["variant"] = agg["variant"].astype(str)
    agg = agg.set_index("variant", drop=False)

    condition_order = ["path_only", "grid", "grid_labels"]
    condition_colors = {
        "path_only": "#0072B2",
        "grid": "#009E73",
        "grid_labels": "#D55E00",
    }

    condition_handles = [
        Line2D(
            [0],
            [0],
            color=condition_colors[cond],
            marker="o",
            linewidth=2.4,
            markersize=8,
            label=VISUAL_CONDITION_LABELS[cond],
        )
        for cond in condition_order
    ]

    block_centers = [1, 4, 7]
    block_titles = ["No annotations", "Grid", "Grid + labels"]
    x = np.arange(len(variants))

    for ax, metric in zip(axes, metrics):
        mean_col = f"mean_{metric}"
        err_col = f"mean_inner_batch_std_{metric}" if f"mean_inner_batch_std_{metric}" in agg.columns else f"std_{metric}"

        subset = agg.reindex(variants)
        means = pd.to_numeric(subset[mean_col], errors="coerce").to_numpy(dtype=float)
        errs = (
            pd.to_numeric(subset[err_col], errors="coerce").fillna(0).to_numpy(dtype=float)
            if err_col in subset.columns
            else np.zeros_like(means)
        )

        for condition in condition_order:
            block_variants = [f"L0_{condition}", f"L1_{condition}", f"L2_{condition}"]
            block_positions = [variants.index(v) for v in block_variants if v in variants]
            if len(block_positions) < 2:
                continue

            block_means = [means[pos] for pos in block_positions]
            block_errs = [errs[pos] for pos in block_positions]

            ax.errorbar(
                block_positions,
                block_means,
                yerr=block_errs,
                fmt="o-",
                color=condition_colors[condition],
                linewidth=2.4,
                markersize=8,
                capsize=5,
                zorder=3,
            )

        raw = _subset_metric_df(long_df, metric)
        if not raw.empty:
            raw = raw[raw["variant"].isin(variants)]

            for xi, variant in enumerate(variants):
                values = pd.to_numeric(
                    raw.loc[raw["variant"] == variant, "value"],
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)

                if len(values) == 0:
                    continue

                jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
                condition = _variant_condition_key(variant)

                ax.scatter(
                    np.full(len(values), xi) + jitter,
                    values,
                    s=36,
                    alpha=0.60,
                    color=condition_colors.get(condition, "#0072B2"),
                    edgecolors="none",
                    zorder=2,
                )

        ax.set_title(
            _metric_title(metric),
            loc="left",
            fontweight="bold",
            fontsize=22,
            pad=8,
        )
        ax.set_ylabel("Score", fontsize=20, fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_variant_grid_level_label(v) for v in variants],
            rotation=0,
            ha="center",
        )
        ax.tick_params(axis="x", pad=2)

        _panel_group_separators(ax, len(variants))
        _style_bar_axes(ax)
        _make_axes_fonts_bold_and_larger(ax)
        subtitle_x_offset = 0.22

        for xpos, title in zip(block_centers, block_titles):
            ax.text(
                xpos + subtitle_x_offset,
                0.965,
                title,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    legend = fig.legend(
        handles=condition_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=22,
        markerscale=1.35,
        handlelength=2.6,
        handletextpad=0.9,
        columnspacing=2.6,
        borderpad=0.8,
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    # Reserve explicit space underneath the plots for the legend.
    # Avoid tight_layout here, because it can clip figure-level legends.
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.91,
        bottom=0.26,
        wspace=0.30,
    )

    fig.savefig(
        figures_dir / "f1_accuracy_exact_match_point_range.png",
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.35,
    )
    plt.close(fig)
    
def plot_error_metric_panels(aggregate_df: pd.DataFrame, long_df: pd.DataFrame, figures_dir: Path) -> None:
    """Show error metrics as point-range panels with raw run values."""
    if aggregate_df.empty:
        return

    metrics = [m for m in ERROR_METRICS if f"mean_{m}" in aggregate_df.columns]
    if not metrics:
        return

    variants = _ordered_variants_for_plot(aggregate_df["variant"].astype(str).tolist())
    if not variants:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(13.8, 5.2), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    agg = aggregate_df.copy()
    agg["variant"] = agg["variant"].astype(str)
    agg = agg.set_index("variant", drop=False)

    for ax, metric in zip(axes, metrics):
        mean_col = f"mean_{metric}"
        err_col = f"mean_inner_batch_std_{metric}" if f"mean_inner_batch_std_{metric}" in agg.columns else f"std_{metric}"
        subset = agg.reindex(variants)
        x = np.arange(len(variants))
        means = pd.to_numeric(subset[mean_col], errors="coerce").to_numpy(dtype=float)
        errs = pd.to_numeric(subset[err_col], errors="coerce").fillna(0).to_numpy(dtype=float) if err_col in subset.columns else np.zeros_like(means)

        ax.errorbar(x, means, yerr=errs, fmt="o-", color="black", linewidth=1.4, markersize=4.5, capsize=4, zorder=3)

        raw = _subset_metric_df(long_df, metric)
        if not raw.empty:
            raw = raw[raw["variant"].isin(variants)]
            for xi, variant in enumerate(variants):
                values = pd.to_numeric(raw.loc[raw["variant"] == variant, "value"], errors="coerce").dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
                ax.scatter(
                    np.full(len(values), xi) + jitter,
                    values,
                    s=18,
                    alpha=0.55,
                    color="#D55E00" if metric == "missed_tile_rate" else "#0072B2",
                    edgecolors="none",
                    zorder=2,
                )

        ax.set_title(_metric_title(metric), loc="left", fontweight="bold")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels([_grouped_variant_label(v) for v in variants], rotation=0, ha="center")
        _panel_group_separators(ax, len(variants))
        _style_bar_axes(ax)
        _annotate_repeat_note(ax)

    fig.suptitle("Case Study 3 error metrics by visualization condition", x=0.01, y=0.995, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(figures_dir / "error_metrics_point_range.png", dpi=300)
    plt.close(fig)


def plot_metric_distribution_by_variant(long_df: pd.DataFrame, figures_dir: Path, metric: str) -> None:
    """Box-and-strip plots to show the run-level distribution of a metric."""
    df = _subset_metric_df(long_df, metric)
    if df.empty:
        return

    variants = _ordered_variants_for_plot(df["variant"].unique().tolist())
    if not variants:
        return

    data = []
    for variant in variants:
        values = pd.to_numeric(df.loc[df["variant"] == variant, "value"], errors="coerce").dropna().to_numpy(dtype=float)
        data.append(values)

    fig, ax = plt.subplots(figsize=(15.0, 5.6))
    bp = ax.boxplot(data, positions=np.arange(len(variants)), widths=0.62, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    for element in ["whiskers", "caps", "medians"]:
        for artist in bp[element]:
            artist.set_color("black")
            artist.set_linewidth(1.0)

    for xi, values in enumerate(data):
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(values), xi) + jitter, values, s=22, alpha=0.6, color="#0072B2", edgecolors="none", zorder=3)

    ax.set_title(f"Run-level distribution of {_metric_title(metric)}", loc="left", fontweight="bold")
    ax.set_ylabel(_metric_title(metric))
    ax.set_xlabel("Visualization condition")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels([_grouped_variant_label(v) for v in variants], rotation=0, ha="center")
    _panel_group_separators(ax, len(variants))
    _style_bar_axes(ax)
    _annotate_repeat_note(ax)
    fig.tight_layout()
    fig.savefig(figures_dir / f"distribution_{metric}.png", dpi=300)
    plt.close(fig)


def plot_condition_comparison_panels(aggregate_df: pd.DataFrame, long_df: pd.DataFrame, figures_dir: Path) -> None:
    """Show the key hierarchy directly: no annotations -> grid -> grid + labels."""
    if aggregate_df.empty:
        return

    metrics = [m for m in ["tile_accuracy", "missed_tile_rate"] if f"mean_{m}" in aggregate_df.columns]
    if not metrics:
        return

    condition_order = ["path_only", "grid", "grid_labels"]
    condition_labels = [VISUAL_CONDITION_LABELS[c] for c in condition_order]
    level_order = ["L0", "L1", "L2"]
    palette = {"L0": "#0072B2", "L1": "#009E73", "L2": "#D55E00"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(13.2, 5.2), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    agg = aggregate_df.copy()
    agg["variant"] = agg["variant"].astype(str)
    agg["grid_level_label"] = agg["variant"].map(_variant_grid_level_label)
    agg["condition_key"] = agg["variant"].map(_variant_condition_key)

    for ax, metric in zip(axes, metrics):
        for level in level_order:
            rows = []
            for cond in condition_order:
                variant = f"{level}_{cond}"
                row = agg.loc[agg["variant"] == variant]
                if row.empty:
                    rows.append((np.nan, np.nan))
                else:
                    mean_val = pd.to_numeric(row[f"mean_{metric}"], errors="coerce").iloc[0]
                    err_col = f"mean_inner_batch_std_{metric}" if f"mean_inner_batch_std_{metric}" in row.columns else f"std_{metric}"
                    err_val = pd.to_numeric(row[err_col], errors="coerce").iloc[0] if err_col in row.columns else np.nan
                    rows.append((mean_val, err_val))

            means = np.array([r[0] for r in rows], dtype=float)
            errs = np.array([0.0 if r[1] is None or not np.isfinite(r[1]) else r[1] for r in rows], dtype=float)
            x = np.arange(len(condition_order))
            ax.errorbar(x, means, yerr=errs, fmt="o-", linewidth=1.5, markersize=5, capsize=4, label=level, color=palette[level], zorder=3)

            raw = _subset_metric_df(long_df, metric)
            if not raw.empty:
                for xi, cond in enumerate(condition_order):
                    variant = f"{level}_{cond}"
                    values = pd.to_numeric(raw.loc[raw["variant"] == variant, "value"], errors="coerce").dropna().to_numpy(dtype=float)
                    if len(values) == 0:
                        continue
                    jitter = np.linspace(-0.05, 0.05, len(values)) if len(values) > 1 else np.array([0.0])
                    ax.scatter(np.full(len(values), xi) + jitter, values, s=18, alpha=0.45, color=palette[level], edgecolors="none", zorder=2)

        ax.set_title(_metric_title(metric), loc="left", fontweight="bold")
        ax.set_ylabel("Score" if metric == "tile_accuracy" else "Rate")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(np.arange(len(condition_order)))
        ax.set_xticklabels(condition_labels, rotation=0, ha="center")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.legend(title="Grid level", frameon=True, framealpha=1.0, facecolor="white", edgecolor="black", loc="upper right")
        if metric == "missed_tile_rate":
            ax.text(0.5, -0.18, "Lower is better for missed cell rate", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
        _annotate_repeat_note(ax)

    fig.suptitle("Main comparison: grid > grid + labels > no annotations", x=0.01, y=0.995, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(figures_dir / "condition_comparison_panels.png", dpi=300)
    plt.close(fig)


def plot_run_variability(long_df: pd.DataFrame, figures_dir: Path) -> None:
    if long_df.empty:
        return

    for metric in [m for m in ALL_METRICS if m in set(long_df["metric"])]:
        df = _subset_metric_df(long_df, metric)
        if df.empty:
            continue

        variants = _ordered_variants_for_plot(df["variant"].unique().tolist())
        x_positions = {v: i for i, v in enumerate(variants)}

        fig, ax = plt.subplots(figsize=(15.0, 5.4))
        for variant in variants:
            values = pd.to_numeric(df[df["variant"] == variant]["value"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(np.full(len(values), x_positions[variant]) + jitter, values, s=52, alpha=0.75)
            ax.errorbar(
                x_positions[variant],
                np.mean(values),
                yerr=np.std(values, ddof=0),
                fmt="o",
                markersize=8,
                capsize=5,
            )

        ax.set_title(f"Run-level variation in {_metric_title(metric)}", loc="left", fontweight="bold")
        ax.set_ylabel(_metric_title(metric))
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([_grouped_variant_label(v) for v in variants], rotation=0, ha="center")
        ax.set_ylim(-0.05, 1.05)
        _panel_group_separators(ax, len(variants))
        ax.grid(axis="y", alpha=0.25)
        _annotate_repeat_note(ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"run_variability_{metric}.png", dpi=300)
        plt.close(fig)


def plot_metric_heatmap(aggregate_df: pd.DataFrame, figures_dir: Path) -> None:
    if aggregate_df.empty:
        return
    mean_cols = [f"mean_{m}" for m in ALL_METRICS if f"mean_{m}" in aggregate_df.columns]
    if not mean_cols:
        return

    matrix = aggregate_df[mean_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    row_labels = [_grouped_variant_label(str(v)) for v in aggregate_df["variant"]]
    col_labels = [_metric_title(c.replace("mean_", "")) for c in mean_cols]

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Mean metric scores by visualization condition", loc="left", fontweight="bold")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")

    ax.grid(False)
    ax.set_axisbelow(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean score")
    fig.tight_layout()
    fig.savefig(figures_dir / "mean_metric_heatmap.png", dpi=300)
    plt.close(fig)





def plot_difficulty_accuracy_bars(difficulty_df: pd.DataFrame, figures_dir: Path) -> None:
    """Exploratory plot: accuracy by path-turn difficulty only."""
    if difficulty_df.empty or "mean_tile_accuracy" not in difficulty_df.columns:
        return

    df = difficulty_df.copy()
    df = df[df["turn_difficulty"].astype(str) != "unknown"]
    if df.empty:
        return

    labels = list(df["turn_difficulty_label"].astype(str))
    x = np.arange(len(df))
    means = pd.to_numeric(df["mean_tile_accuracy"], errors="coerce").to_numpy(dtype=float)
    yerr = (
        pd.to_numeric(df["mean_inner_batch_std_tile_accuracy"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if "mean_inner_batch_std_tile_accuracy" in df.columns
        else pd.to_numeric(df["std_tile_accuracy"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if "std_tile_accuracy" in df.columns
        else None
    )

    fig, ax = plt.subplots(figsize=(8.8, 5))
    ax.bar(x, means, yerr=yerr, capsize=5)
    ax.set_title("Exploratory: cell accuracy by path-turn difficulty", loc="left", fontweight="bold")
    ax.set_ylabel("Cell accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)

    for i, row in enumerate(df.itertuples(index=False)):
        n_rows = getattr(row, "n_rows", None)
        if n_rows is not None:
            ax.text(i, 0.02, f"n={int(n_rows)}", ha="center", va="bottom", fontsize=9)

    ax.text(0.5, -0.18, "No strong conclusion should be drawn from this variable alone.", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
    fig.tight_layout()
    fig.savefig(figures_dir / "difficulty_tile_accuracy_mean_std.png", dpi=300)
    plt.close(fig)


def plot_variant_difficulty_accuracy_bars(variant_difficulty_df: pd.DataFrame, figures_dir: Path) -> None:
    """Exploratory plot: accuracy by variant, split by path-turn difficulty."""
    if variant_difficulty_df.empty or "mean_tile_accuracy" not in variant_difficulty_df.columns:
        return

    df = variant_difficulty_df.copy()
    df = df[df["turn_difficulty"].astype(str) != "unknown"]
    if df.empty:
        return

    variants = [v for v in PLOT_VARIANT_ORDER if v in set(df["variant"].astype(str))]
    difficulties = [d for d in DIFFICULTY_ORDER if d in set(df["turn_difficulty"].astype(str)) and d != "unknown"]
    if not variants or not difficulties:
        return

    x = np.arange(len(variants))
    width = min(0.18, 0.74 / max(1, len(difficulties)))

    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    for idx, difficulty in enumerate(difficulties):
        difficulty_df = df[df["turn_difficulty"].astype(str) == difficulty]
        subset = difficulty_df.set_index(difficulty_df["variant"].astype(str))
        means = []
        yerrs = []
        for variant in variants:
            if variant in subset.index:
                means.append(safe_float(subset.loc[variant, "mean_tile_accuracy"]))
                yerrs.append(safe_float(subset.loc[variant, "mean_inner_batch_std_tile_accuracy"]) if "mean_inner_batch_std_tile_accuracy" in subset.columns else safe_float(subset.loc[variant, "std_tile_accuracy"]) if "std_tile_accuracy" in subset.columns else 0.0)
            else:
                means.append(np.nan)
                yerrs.append(0.0)
        offset = (idx - (len(difficulties) - 1) / 2) * width
        ax.bar(
            x + offset,
            np.array(means, dtype=float),
            width=width,
            yerr=np.array([0.0 if v is None or not np.isfinite(v) else v for v in yerrs], dtype=float),
            capsize=4,
            color=DIFFICULTY_COLORS.get(difficulty, "0.5"),
            edgecolor="white",
            linewidth=0.4,
            label=DIFFICULTY_LABELS.get(difficulty, difficulty),
            zorder=3,
        )

    ax.set_title("Exploratory: cell accuracy by variant and path-turn difficulty", loc="left", fontweight="bold")
    ax.set_ylabel("Cell accuracy (mean ± variability across repeats)")
    ax.set_xlabel("Visualization variant")
    ax.set_xticks(x)
    ax.set_xticklabels([_grouped_variant_label(v) for v in variants], rotation=0, ha="center")
    ax.set_ylim(0, 1.08)
    ax.margins(x=0.02)
    _style_bar_axes(ax)
    _top_center_boxed_legend(ax, title="Path difficulty", ncol=2, bbox_to_anchor=(0.5, 1.20))

    ax.text(0.5, -0.18, "Path difficulty is shown descriptively; avoid overinterpreting this axis.", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(figures_dir / "variant_difficulty_tile_accuracy_mean_std.png", dpi=300)
    plt.close(fig)


def plot_variant_difficulty_accuracy_bars_inner_run_std(variant_difficulty_df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot cell accuracy by variant/difficulty using average within-run std as error bars."""
    if (
        variant_difficulty_df.empty
        or "mean_tile_accuracy" not in variant_difficulty_df.columns
        or "mean_inner_run_std_tile_accuracy" not in variant_difficulty_df.columns
    ):
        return

    df = variant_difficulty_df.copy()
    df = df[df["turn_difficulty"].astype(str) != "unknown"]
    if df.empty:
        return

    variants = [v for v in PLOT_VARIANT_ORDER if v in set(df["variant"].astype(str))]
    difficulties = [d for d in DIFFICULTY_ORDER if d in set(df["turn_difficulty"].astype(str)) and d != "unknown"]
    if not variants or not difficulties:
        return

    x = np.arange(len(variants))
    width = min(0.18, 0.74 / max(1, len(difficulties)))

    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    for idx, difficulty in enumerate(difficulties):
        difficulty_df = df[df["turn_difficulty"].astype(str) == difficulty]
        subset = difficulty_df.set_index(difficulty_df["variant"].astype(str))
        means = []
        yerrs = []
        for variant in variants:
            if variant in subset.index:
                means.append(safe_float(subset.loc[variant, "mean_tile_accuracy"]))
                yerrs.append(safe_float(subset.loc[variant, "mean_inner_run_std_tile_accuracy"]))
            else:
                means.append(np.nan)
                yerrs.append(0.0)
        offset = (idx - (len(difficulties) - 1) / 2) * width
        ax.bar(
            x + offset,
            np.array(means, dtype=float),
            width=width,
            yerr=np.array([0.0 if v is None or not np.isfinite(v) else v for v in yerrs], dtype=float),
            capsize=4,
            color=DIFFICULTY_COLORS.get(difficulty, "0.5"),
            edgecolor="white",
            linewidth=0.4,
            label=DIFFICULTY_LABELS.get(difficulty, difficulty),
            zorder=3,
        )

    ax.set_title("Exploratory: cell accuracy by variant and path-turn difficulty", loc="left", fontweight="bold")
    ax.set_ylabel("Cell accuracy (mean ± mean within-run SD)")
    ax.set_xlabel("Visualization variant")
    ax.set_xticks(x)
    ax.set_xticklabels([_grouped_variant_label(v) for v in variants], rotation=0, ha="center")
    ax.set_ylim(0, 1.08)
    ax.margins(x=0.02)
    _style_bar_axes(ax)
    _top_center_boxed_legend(ax, title="Path difficulty", ncol=2, bbox_to_anchor=(0.5, 1.20))
    ax.text(0.5, -0.18, "Path difficulty is shown descriptively; avoid overinterpreting this axis.", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(figures_dir / "variant_difficulty_tile_accuracy_mean_inner_run_std.png", dpi=300)
    plt.close(fig)


def plot_variant_difficulty_accuracy_bars_inner_batch_std(variant_difficulty_df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot cell accuracy using mean within-batch std across repeated runs as error bars."""
    if variant_difficulty_df.empty or "mean_tile_accuracy" not in variant_difficulty_df.columns:
        return
    if "mean_inner_batch_std_tile_accuracy" not in variant_difficulty_df.columns:
        print("[warn] Skipping inner-batch std plot: mean_inner_batch_std_tile_accuracy is missing.")
        return

    df = variant_difficulty_df.copy()
    df = df[df["turn_difficulty"].astype(str) != "unknown"]
    if df.empty:
        return

    variants = [v for v in PLOT_VARIANT_ORDER if v in set(df["variant"].astype(str))]
    difficulties = [d for d in DIFFICULTY_ORDER if d in set(df["turn_difficulty"].astype(str)) and d != "unknown"]
    if not variants or not difficulties:
        return

    x = np.arange(len(variants))
    width = min(0.18, 0.74 / max(1, len(difficulties)))

    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    for idx, difficulty in enumerate(difficulties):
        difficulty_df = df[df["turn_difficulty"].astype(str) == difficulty]
        subset = difficulty_df.set_index(difficulty_df["variant"].astype(str))
        means = []
        yerrs = []
        for variant in variants:
            if variant in subset.index:
                means.append(safe_float(subset.loc[variant, "mean_tile_accuracy"]))
                yerrs.append(safe_float(subset.loc[variant, "mean_inner_batch_std_tile_accuracy"]))
            else:
                means.append(np.nan)
                yerrs.append(0.0)
        offset = (idx - (len(difficulties) - 1) / 2) * width
        ax.bar(
            x + offset,
            np.array(means, dtype=float),
            width=width,
            yerr=np.array([0.0 if v is None or not np.isfinite(v) else v for v in yerrs], dtype=float),
            capsize=4,
            color=DIFFICULTY_COLORS.get(difficulty, "0.5"),
            edgecolor="white",
            linewidth=0.4,
            label=DIFFICULTY_LABELS.get(difficulty, difficulty),
            zorder=3,
        )

    ax.set_title("Exploratory: cell accuracy by variant and path-turn difficulty", loc="left", fontweight="bold")
    ax.set_ylabel("Cell accuracy (mean ± mean within-batch SD)")
    ax.set_xlabel("Visualization variant")
    ax.set_xticks(x)
    ax.set_xticklabels([_grouped_variant_label(v) for v in variants], rotation=0, ha="center")
    ax.set_ylim(0, 1.08)
    ax.margins(x=0.02)
    _style_bar_axes(ax)
    _top_center_boxed_legend(ax, title="Path difficulty", ncol=2, bbox_to_anchor=(0.5, 1.20))
    ax.text(0.5, -0.18, "Path difficulty is shown descriptively; avoid overinterpreting this axis.", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.35")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(figures_dir / "variant_difficulty_tile_accuracy_mean_inner_batch_std.png", dpi=300)
    plt.close(fig)


def plot_variant_difficulty_accuracy_heatmap(variant_difficulty_df: pd.DataFrame, figures_dir: Path) -> None:
    """Heatmap of mean cell accuracy for each variant x difficulty bucket."""
    if variant_difficulty_df.empty or "mean_tile_accuracy" not in variant_difficulty_df.columns:
        return

    df = variant_difficulty_df.copy()
    df = df[df["turn_difficulty"].astype(str) != "unknown"]
    if df.empty:
        return

    variants = [v for v in PLOT_VARIANT_ORDER if v in set(df["variant"].astype(str))]
    difficulties = [d for d in DIFFICULTY_ORDER if d in set(df["turn_difficulty"].astype(str)) and d != "unknown"]
    if not variants or not difficulties:
        return

    pivot = df.pivot_table(
        index="variant",
        columns="turn_difficulty",
        values="mean_tile_accuracy",
        aggfunc="mean",
        observed=True,
    )
    pivot = pivot.reindex(index=variants, columns=difficulties)
    matrix = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.6, max(4.8, 0.48 * len(variants))))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Mean cell accuracy across variants and path-turn difficulty", loc="left", fontweight="bold")
    ax.set_xlabel("Path-turn difficulty")
    ax.set_ylabel("Variant")
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels([_grouped_variant_label(v) for v in variants])
    ax.set_xticks(np.arange(len(difficulties)))
    ax.set_xticklabels([DIFFICULTY_LABELS.get(d, d) for d in difficulties], rotation=18, ha="right")

    ax.grid(False)
    ax.set_axisbelow(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Cell accuracy (mean)")
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("black")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "variant_difficulty_tile_accuracy_heatmap.png", dpi=300)
    plt.close(fig)


def make_plots(paths: AnalysisPaths, aggregate_df: pd.DataFrame, long_df: pd.DataFrame, difficulty_df: pd.DataFrame | None = None, variant_difficulty_df: pd.DataFrame | None = None) -> None:
    set_publication_style()
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    plot_primary_metric_panels(aggregate_df, long_df, paths.figures_dir)
    plot_f1_accuracy_exact_match_panels(aggregate_df, long_df, paths.figures_dir)
    plot_error_metric_panels(aggregate_df, long_df, paths.figures_dir)
    plot_condition_comparison_panels(aggregate_df, long_df, paths.figures_dir)
    plot_run_variability(long_df, paths.figures_dir)
    plot_metric_heatmap(aggregate_df, paths.figures_dir)

    # Distribution plots that make the run-level spread visible.
    for metric in ["tile_accuracy", "missed_tile_rate", "exact_match"]:
        if f"mean_{metric}" in aggregate_df.columns:
            plot_metric_distribution_by_variant(long_df, paths.figures_dir, metric)

    if difficulty_df is not None and not difficulty_df.empty:
        plot_difficulty_accuracy_bars(difficulty_df, paths.figures_dir)
    if variant_difficulty_df is not None and not variant_difficulty_df.empty:
        plot_variant_difficulty_accuracy_bars(variant_difficulty_df, paths.figures_dir)
        plot_variant_difficulty_accuracy_bars_inner_batch_std(variant_difficulty_df, paths.figures_dir)
        plot_variant_difficulty_accuracy_bars_inner_run_std(variant_difficulty_df, paths.figures_dir)
        plot_variant_difficulty_accuracy_heatmap(variant_difficulty_df, paths.figures_dir)
def print_console_summary(aggregate_df: pd.DataFrame, out_dir: Path) -> None:
    if aggregate_df.empty:
        print("No aggregate metrics found. Check --root/--batch-id and the runs folder structure.")
        return

    cols = [c for c in ["variant_label", "grid_level", "visual_condition_label", "n_runs"] if c in aggregate_df.columns]
    for metric in ["precision", "recall", "f1", "tile_accuracy", "missed_tile_rate", "extra_tile_rate", "exact_match"]:
        if f"mean_{metric}" in aggregate_df.columns:
            cols.extend([f"mean_{metric}", f"std_{metric}"])

    summary = aggregate_df[cols].copy()
    for c in summary.columns:
        if c.startswith(("mean_", "std_")):
            summary[c] = pd.to_numeric(summary[c], errors="coerce").round(4)

    print("\nCase Study 3 aggregate summary")
    print(summary.to_string(index=False))
    print(f"\nSaved tables and figures to: {out_dir}")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Analyze Case Study 3 run/batch outputs.")
    p.add_argument("--root", type=Path, default=default_root, help="Case Study 3 output root. Default: script directory.")
    p.add_argument("--batch-id", type=str, default=None, help="Specific batch id, e.g. case3_batch_20260614_124012.")
    p.add_argument("--all", action="store_true", help="Analyze all batches/runs under root/runs instead of only the latest batch.")
    p.add_argument("--out-dir", type=Path, default=None, help="Analysis output directory. Default: root/analysis_outputs/<batch-or-all>.")
    p.add_argument("--no-plots", action="store_true", help="Only write tables, skip matplotlib figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if args.out_dir is None:
        label = args.batch_id or ("all_runs" if args.all else "latest_batch")
        out_dir = root / "analysis_outputs" / label
    else:
        out_dir = args.out_dir.expanduser().resolve()

    paths = AnalysisPaths(
        root=root,
        out_dir=out_dir,
        tables_dir=out_dir / "tables",
        figures_dir=out_dir / "figures",
    )

    if not (root / "runs").exists():
        raise FileNotFoundError(f"Could not find runs directory under root: {root / 'runs'}")

    dfs = collect_dataframes(root=root, batch_id=args.batch_id, include_all=args.all)
    run_df = dfs["run_level_metrics"]
    aggregate_df = compute_aggregate_df(run_df)
    difficulty_df = compute_difficulty_aggregate_df(run_df)
    variant_difficulty_df = compute_variant_difficulty_aggregate_df(run_df)

    # Add the requested variability statistic:
    # for each batch x variant x difficulty, compute std across repeated runs;
    # then average those batch-level stds over batches.
    aggregate_df = add_mean_inner_batch_std_columns(run_df, aggregate_df)
    difficulty_df = add_mean_inner_batch_std_columns(run_df, difficulty_df)
    variant_difficulty_df = add_mean_inner_batch_std_columns(run_df, variant_difficulty_df)

    long_df = make_long_metrics_df(run_df)

    dfs["recomputed_aggregates"] = aggregate_df
    dfs["difficulty_aggregates"] = difficulty_df
    dfs["variant_difficulty_aggregates"] = variant_difficulty_df
    dfs["long_metrics"] = long_df

    save_tables(paths, dfs)
    if not args.no_plots:
        make_plots(
            paths,
            aggregate_df=aggregate_df,
            long_df=long_df,
            difficulty_df=difficulty_df,
            variant_difficulty_df=variant_difficulty_df,
        )

    summary_payload = {
        "root": str(root),
        "out_dir": str(out_dir),
        "n_run_rows": int(len(run_df)),
        "n_prediction_rows": int(len(dfs["prediction_details"])),
        "n_batch_aggregate_rows": int(len(dfs["batch_manifest_aggregates"])),
        "n_difficulty_rows": int(len(difficulty_df)),
        "n_variant_difficulty_rows": int(len(variant_difficulty_df)),
        "difficulties": sorted([str(v) for v in run_df["turn_difficulty"].dropna().unique()], key=lambda v: difficulty_sort_key(v)) if not run_df.empty and "turn_difficulty" in run_df.columns else [],
        "variants": sorted([str(v) for v in run_df["variant"].dropna().unique()], key=lambda v: variant_sort_key(v)) if not run_df.empty else [],
        "tables": sorted([p.name for p in paths.tables_dir.glob("*.csv")]),
        "figures": sorted([p.name for p in paths.figures_dir.glob("*.png")]) if paths.figures_dir.exists() else [],
    }
    write_json(out_dir / "analysis_summary.json", summary_payload)
    print_console_summary(aggregate_df, out_dir)


if __name__ == "__main__":
    main()
