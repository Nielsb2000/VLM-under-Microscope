"""case_study_4/aggregate.py — Multi-run aggregator for Case Study 4.

aggregate_runs(out_dir)
    → Scans outputs/case_study_4/runs/*/predictions/*.json
    → Returns aggregate dict with per-method and per-condition stats

save_aggregate_outputs(agg, out_dir)
    → Saves:
        out_dir/per_sample_method_results.csv
        out_dir/summary_by_method.csv
        out_dir/aggregate_summary.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_PER_SAMPLE_COLS = [
    "run_id", "sample_id", "search_mode", "method_name", "status",
    "gt_target_present", "gt_tile",
    "found", "predicted_tile", "confidence",
    "correct_found_status",
    "correct_tile", "bbox_iou", "localization_error_px", "within_tolerance", "correct_detection",
    "correct_abstention", "false_positive",
    "parsing_error",
]

_SUMMARY_COLS = [
    "method_name", "n_total", "n_parsed", "n_failed",
    "n_present_samples", "n_absent_samples",
    "presence_accuracy", "positive_recall",
    "negative_abstention_accuracy", "false_positive_rate_on_absent",
    "tile_accuracy_on_present",
    "mean_localization_error_px", "mean_bbox_iou", "within_tolerance_rate",
    "parse_failure_rate",
]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_runs(out_dir: Path) -> dict:
    """Scan *out_dir/runs/* for prediction JSONs and aggregate metrics.

    Returns
    -------
    dict with keys:
        n_runs          : int
        all_predictions : list[dict]  — all loaded prediction dicts with run_id injected
        by_method       : dict[str, dict]  — aggregate stats per search_mode
        by_condition    : dict[str, dict]  — aggregate stats per "E1P"/"E1N"/"E2P"/"E2N"
    """
    from outputs.case_study_4.metrics import aggregate_sample_metrics

    runs_dir = out_dir / "runs"
    all_predictions: list[dict] = []

    if not runs_dir.exists():
        return {"n_runs": 0, "all_predictions": [], "by_method": {}, "by_condition": {}}

    manifest_paths = sorted(runs_dir.glob("*/run_manifest.json"))
    loaded_runs = 0

    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        run_id      = manifest.get("run_id", manifest_path.parent.name)
        search_mode = manifest.get("search_mode", "")
        run_dir     = manifest_path.parent

        # Load prediction JSON for this run
        pred_file = run_dir / "predictions" / f"{search_mode}.json"
        if not pred_file.exists():
            # Try any prediction JSON
            pred_files = list((run_dir / "predictions").glob("*.json"))
            if not pred_files:
                continue
            pred_file = pred_files[0]

        try:
            pred = json.loads(pred_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        pred["run_id"] = run_id
        all_predictions.append(pred)
        loaded_runs += 1

    # Group by method_name (= search_mode)
    by_method: dict[str, list[dict]] = {}
    for pred in all_predictions:
        key = pred.get("method_name") or pred.get("search_mode") or "unknown"
        by_method.setdefault(key, []).append(pred)

    # Group by condition (E1P / E1N / E2P / E2N)
    _mode_to_code = {
        "atlas_global_search": "E1",
        "grid_scan_search":    "E2",
    }
    by_condition: dict[str, list[dict]] = {}
    for pred in all_predictions:
        mode    = pred.get("method_name") or pred.get("search_mode") or ""
        present = pred.get("gt_target_present")
        code    = _mode_to_code.get(mode, mode)
        suffix  = "P" if present else "N"
        key     = f"{code}{suffix}"
        by_condition.setdefault(key, []).append(pred)

    return {
        "n_runs":          loaded_runs,
        "all_predictions": all_predictions,
        "by_method":       {m: aggregate_sample_metrics(preds) for m, preds in by_method.items()},
        "by_condition":    {c: aggregate_sample_metrics(preds) for c, preds in by_condition.items()},
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def save_aggregate_outputs(agg: dict, out_dir: Path) -> None:
    """Save per_sample_method_results.csv, summary_by_method.csv, aggregate_summary.json."""
    out_dir = Path(out_dir)

    # --- per_sample_method_results.csv ---
    _write_csv(
        out_dir / "per_sample_method_results.csv",
        agg.get("all_predictions", []),
        _PER_SAMPLE_COLS,
    )

    # --- summary_by_method.csv ---
    summary_rows = []
    for method, stats in agg.get("by_method", {}).items():
        row = {"method_name": method, **stats}
        summary_rows.append(row)
    _write_csv(out_dir / "summary_by_method.csv", summary_rows, _SUMMARY_COLS)

    # --- aggregate_summary.json ---
    summary = {
        "n_runs":        agg.get("n_runs", 0),
        "by_method":     agg.get("by_method", {}),
        "by_condition":  agg.get("by_condition", {}),
    }
    (out_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
