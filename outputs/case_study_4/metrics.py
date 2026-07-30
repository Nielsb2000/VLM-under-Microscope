"""case_study_4/metrics.py — Per-prediction and aggregate metrics for Case Study 4.

compute_prediction_metrics(parsed, gt_entry)
    → full prediction result dict (correct_found_status, tile match, IoU, …)

aggregate_sample_metrics(predictions)
    → aggregate summary across all samples for a method/condition
"""
from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def compute_bbox_iou(pred_bbox: list, gt_bbox: list) -> float:
    """Compute intersection-over-union for two [x1, y1, x2, y2] bboxes."""
    ix1 = max(pred_bbox[0], gt_bbox[0])
    iy1 = max(pred_bbox[1], gt_bbox[1])
    ix2 = min(pred_bbox[2], gt_bbox[2])
    iy2 = min(pred_bbox[3], gt_bbox[3])

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    pred_area    = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area      = (gt_bbox[2]   - gt_bbox[0])   * (gt_bbox[3]   - gt_bbox[1])
    union        = pred_area + gt_area - intersection
    return intersection / union if union > 0 else 0.0


def compute_localization_error(pred_center: list, gt_center: list) -> float:
    """Euclidean distance in pixels between two [cx, cy] centres."""
    dx = pred_center[0] - gt_center[0]
    dy = pred_center[1] - gt_center[1]
    return math.sqrt(dx * dx + dy * dy)


def _bbox_to_center(bbox: list) -> list[float]:
    return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]


# ---------------------------------------------------------------------------
# Per-prediction metrics
# ---------------------------------------------------------------------------

def compute_prediction_metrics(
    parsed:   dict,
    gt_entry: dict,
    *,
    method_name: str = "",
    raw_response: str | None = None,
) -> dict:
    """Compute the full prediction result dict for one sample.

    Parameters
    ----------
    parsed      : output of response_parser.parse_vlm_response()
    gt_entry    : one GT entry dict (from ground_truth.load_gt_store)
    method_name : search mode / experiment ID string
    raw_response: the raw text returned by the VLM (for traceability)

    Returns
    -------
    A prediction dict matching the Case Study 4 schema.
    """
    from outputs.case_study_4.response_parser import tiles_match  # avoid circular at module level

    sample_id        = gt_entry["sample_id"]
    gt_present       = gt_entry["target_present"]
    gt_tile          = gt_entry.get("gt_tile")
    gt_bbox          = gt_entry.get("gt_bbox")
    gt_center        = gt_entry.get("gt_center")
    tolerance_px     = gt_entry.get("acceptable_tolerance_px")

    # Derive gt_center from gt_bbox when absent
    if gt_center is None and gt_bbox is not None:
        gt_center = _bbox_to_center(gt_bbox)

    pred_found  = parsed.get("found", False)
    pred_tile   = parsed.get("tile")
    pred_bbox   = parsed.get("bbox")
    pred_center = _bbox_to_center(pred_bbox) if pred_bbox is not None else None
    confidence  = parsed.get("confidence", 0.0)
    reason      = parsed.get("reason", "")
    parsing_error = parsed.get("parsing_error")
    status      = "parsing_error" if parsing_error else "completed"

    result: dict[str, Any] = {
        "sample_id":            sample_id,
        "method_name":          method_name,
        "target_pattern_image": gt_entry.get("target_pattern_image"),
        "search_image":         gt_entry.get("search_image"),
        "raw_response":         raw_response,
        "found":                pred_found,
        "predicted_tile":       pred_tile,
        "predicted_bbox":       pred_bbox,
        "predicted_center":     pred_center,
        "confidence":           confidence,
        "gt_target_present":    gt_present,
        "gt_tile":              gt_tile,
        "gt_bbox":              gt_bbox,
        "gt_center":            gt_center,
        "status":               status,
        "parsing_error":        parsing_error,
        "error":                None,
    }

    correct_found_status = (pred_found == gt_present)
    result["correct_found_status"] = correct_found_status

    if gt_present:
        # ---- Positive (present) case ----
        correct_tile = tiles_match(pred_tile, gt_tile) if pred_found else False
        result["correct_tile"] = correct_tile

        # BBox IoU
        if pred_bbox is not None and gt_bbox is not None and pred_found:
            result["bbox_iou"] = round(compute_bbox_iou(pred_bbox, gt_bbox), 4)
        else:
            result["bbox_iou"] = None

        # Localisation error
        if pred_center is not None and gt_center is not None and pred_found:
            err = compute_localization_error(pred_center, gt_center)
            result["localization_error_px"] = round(err, 2)
            if tolerance_px is not None:
                result["within_tolerance"] = err <= tolerance_px
            else:
                result["within_tolerance"] = None
        else:
            result["localization_error_px"] = None
            result["within_tolerance"]      = None

        # Correct detection = found AND correct tile (AND within_tolerance if available)
        if pred_found and correct_tile:
            wt = result["within_tolerance"]
            result["correct_detection"] = True if wt is None else bool(wt)
        else:
            result["correct_detection"] = False

    else:
        # ---- Negative (absent) case ----
        result["correct_abstention"] = not pred_found
        result["false_positive"]     = pred_found

    return result


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_sample_metrics(predictions: list[dict]) -> dict:
    """Aggregate per-sample prediction dicts into method-level summary metrics.

    Parameters
    ----------
    predictions : list of dicts from compute_prediction_metrics()

    Returns
    -------
    Aggregate summary dict.
    """
    n_total   = len(predictions)
    n_parsed  = sum(1 for p in predictions if not p.get("parsing_error"))
    n_failed  = n_total - n_parsed

    present_preds = [p for p in predictions if p.get("gt_target_present") is True]
    absent_preds  = [p for p in predictions if p.get("gt_target_present") is False]

    # Presence accuracy: correct_found_status over all samples
    n_correct_found = sum(1 for p in predictions if p.get("correct_found_status"))
    presence_accuracy = n_correct_found / n_total if n_total > 0 else None

    # Positive recall (sensitivity): among present cases, how many were found
    n_found_when_present = sum(1 for p in present_preds if p.get("found") is True)
    positive_recall = (
        n_found_when_present / len(present_preds)
        if present_preds else None
    )

    # Negative abstention accuracy: among absent cases, how many returned found=false
    n_abstained_when_absent = sum(1 for p in absent_preds if p.get("correct_abstention") is True)
    negative_abstention_accuracy = (
        n_abstained_when_absent / len(absent_preds)
        if absent_preds else None
    )

    # False positive rate on absent cases
    n_fp = sum(1 for p in absent_preds if p.get("false_positive") is True)
    false_positive_rate = (
        n_fp / len(absent_preds)
        if absent_preds else None
    )

    # Tile accuracy on present cases (found AND correct_tile)
    n_correct_tile = sum(1 for p in present_preds if p.get("correct_tile") is True)
    tile_accuracy_on_present = (
        n_correct_tile / len(present_preds)
        if present_preds else None
    )

    # Mean localisation error on present cases (where available)
    loc_errors = [
        p["localization_error_px"]
        for p in present_preds
        if p.get("localization_error_px") is not None
    ]
    mean_localization_error_px = (
        round(sum(loc_errors) / len(loc_errors), 2)
        if loc_errors else None
    )

    # Mean bbox IoU on present cases (where available)
    ious = [p["bbox_iou"] for p in present_preds if p.get("bbox_iou") is not None]
    mean_bbox_iou = round(sum(ious) / len(ious), 4) if ious else None

    # Within-tolerance rate
    tolerance_vals = [
        p["within_tolerance"]
        for p in present_preds
        if p.get("within_tolerance") is not None
    ]
    within_tolerance_rate = (
        sum(tolerance_vals) / len(tolerance_vals)
        if tolerance_vals else None
    )

    # Parse failure rate
    parse_failure_rate = n_failed / n_total if n_total > 0 else 0.0

    def _r(v: float | None, digits: int = 4) -> float | None:
        return round(v, digits) if v is not None else None

    return {
        "n_total":                       n_total,
        "n_parsed":                      n_parsed,
        "n_failed":                      n_failed,
        "n_present_samples":             len(present_preds),
        "n_absent_samples":              len(absent_preds),
        "presence_accuracy":             _r(presence_accuracy),
        "positive_recall":               _r(positive_recall),
        "negative_abstention_accuracy":  _r(negative_abstention_accuracy),
        "false_positive_rate_on_absent": _r(false_positive_rate),
        "tile_accuracy_on_present":      _r(tile_accuracy_on_present),
        "mean_localization_error_px":    _r(mean_localization_error_px, 2),
        "mean_bbox_iou":                 _r(mean_bbox_iou),
        "within_tolerance_rate":         _r(within_tolerance_rate),
        "parse_failure_rate":            _r(parse_failure_rate),
    }
