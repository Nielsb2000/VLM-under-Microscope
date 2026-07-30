from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _flatten_record(record: dict[str, Any], source_path: Path, record_type: str) -> dict[str, Any]:
    random_filters = record.get("random_filters") or {}
    final_filters = record.get("final_filters") or {}
    images = record.get("images") or {}
    actions = record.get("actions") or {}
    histograms = record.get("histograms") or {}

    run_dir = record.get("run_dir")
    if run_dir is None and source_path.name == "run_manifest.json":
        run_dir = str(source_path.parent)

    sample = record.get("sample_id") or record.get("dataset_image") or record.get("image")
    image_name = record.get("image") or (Path(str(sample)).name if sample else None)

    final_score = record.get("final_score", record.get("score"))
    randomized_score = record.get("randomized_score")
    if randomized_score is not None and final_score is not None:
        computed_abs_improvement = float(randomized_score) - float(final_score)
        computed_rel_improvement = 100.0 * computed_abs_improvement / abs(float(randomized_score)) if float(randomized_score) != 0 else None
    else:
        computed_abs_improvement = None
        computed_rel_improvement = None

    out = {
        "source_path": str(source_path),
        "record_type": record_type,
        "run_id": record.get("run_id") or record.get("run_label"),
        "run_index": record.get("run_index"),
        "case_study": record.get("case_study"),
        "status": record.get("status"),
        "started_at": record.get("started_at"),
        "completed_at": record.get("completed_at"),
        "exploratory": bool(record.get("exploratory", False)),
        "method": "exploratory" if bool(record.get("exploratory", False)) else "direct",
        "model_name": record.get("model_name"),
        "seed": record.get("seed"),
        "sample_seed": record.get("sample_seed"),
        "randomization_seed": record.get("randomization_seed"),
        "sample_index": record.get("sample_index"),
        "sample_id": sample,
        "image": image_name,
        "image_category": record.get("image_category"),
        "run_dir": run_dir,
        "final_score": final_score,
        "randomized_score": randomized_score,
        "absolute_improvement": record.get("absolute_improvement", computed_abs_improvement),
        "relative_improvement": record.get("relative_improvement", computed_rel_improvement),
        "filter_adjustments": record.get("filter_adjustments"),
        "vlm_snapshots": record.get("vlm_snapshots"),
        "random_brightness": random_filters.get("brightness"),
        "random_contrast": random_filters.get("contrast"),
        "final_brightness": final_filters.get("brightness"),
        "final_contrast": final_filters.get("contrast"),
        "image_randomized_start": images.get("randomized_start"),
        "image_final_result": images.get("final_result"),
        "image_reference_hidden": images.get("reference_hidden"),
        "hist_randomized": histograms.get("randomized"),
        "hist_result": histograms.get("result"),
        "hist_reference": histograms.get("reference"),
        "hist_comparison": histograms.get("comparison"),
        "action_filter_trajectory": actions.get("filter_trajectory"),
        "action_vlm_snapshots": actions.get("vlm_snapshots"),
        "error": record.get("error"),
        "warnings": "; ".join(record.get("warnings") or []),
    }
    return out


def load_runs(output_root: str | Path, include_manifests: bool = True, include_summaries: bool = True) -> pd.DataFrame:
    """Load run_manifest.json and multi_run_summary_*.json records into one tidy table."""
    root = Path(output_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []

    if include_manifests:
        for path in sorted(root.glob("runs/**/run_manifest.json")):
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            rows.append(_flatten_record(data, path, "manifest"))

    if include_summaries:
        for path in sorted(root.glob("multi_run_summary_*.json")):
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for record in data:
                    rows.append(_flatten_record(record, path, "summary"))
            elif isinstance(data, dict):
                rows.append(_flatten_record(data, path, "summary"))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Prefer manifests over summary rows if the same run is present in both.
    df["record_priority"] = df["record_type"].map({"manifest": 0, "summary": 1}).fillna(2)
    if "run_id" in df.columns:
        df = df.sort_values(["run_id", "record_priority"]).drop_duplicates(subset=["run_id"], keep="first")
    df = df.drop(columns=["record_priority"])

    numeric_cols = [
        "final_score",
        "randomized_score",
        "absolute_improvement",
        "relative_improvement",
        "filter_adjustments",
        "vlm_snapshots",
        "random_brightness",
        "random_contrast",
        "final_brightness",
        "final_contrast",
        "seed",
        "sample_seed",
        "randomization_seed",
        "sample_index",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["started_at", "completed_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    if {"started_at", "completed_at"}.issubset(df.columns):
        df["duration_s"] = (df["completed_at"] - df["started_at"]).dt.total_seconds()

    return df.reset_index(drop=True)


def resolve_run_path(row: pd.Series, relative_path: str | None) -> Path | None:
    if not relative_path or pd.isna(relative_path):
        return None
    p = Path(str(relative_path))
    if p.is_absolute():
        return p
    run_dir = row.get("run_dir")
    if run_dir and not pd.isna(run_dir):
        return Path(str(run_dir)) / p
    return None


def load_filter_trajectory(row: pd.Series) -> pd.DataFrame | None:
    path = resolve_run_path(row, row.get("action_filter_trajectory"))
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    # Normalize expected column names where possible.
    aliases = {
        "brightness": ["brightness", "filter_brightness", "new_brightness"],
        "contrast": ["contrast", "filter_contrast", "new_contrast"],
        "score": ["score", "metric", "final_score", "result_score"],
        "step": ["step", "iteration", "action_index"],
    }
    for canonical, options in aliases.items():
        for option in options:
            if option in df.columns and canonical not in df.columns:
                df[canonical] = df[option]
                break
    if "step" not in df.columns:
        df["step"] = range(len(df))
    return df
