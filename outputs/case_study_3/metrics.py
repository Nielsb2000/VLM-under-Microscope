"""case_study_3/metrics.py - Per-prediction and per-variant metrics for Case Study 3.

compute_prediction_metrics(predicted_tile_set, gt_tile_set, **extra)
    -> full per-prediction result dict (precision, recall, F1, exact_match, ...)

aggregate_variant_metrics(results)
    -> summary dict aggregated over all predictions for one variant
"""
from __future__ import annotations

import statistics


def compute_prediction_metrics(
    predicted_tile_sequence: list[str],
    predicted_tile_set:      list[str],
    gt_tile_sequence:        list[str],
    gt_tile_set:             list[str],
    parsing_error:           str | None = None,
    **extra,
) -> dict:
    """Compute tile-level metrics for a single prediction.

    Parameters
    ----------
    predicted_tile_sequence : ordered predicted tiles
    predicted_tile_set      : unique predicted tiles (sorted)
    gt_tile_sequence        : ground-truth tile order
    gt_tile_set             : unique GT tiles (sorted)
    parsing_error           : non-None if the model response could not be parsed
    **extra                 : any extra fields to merge into the result

    Returns
    -------
    dict with all fields defined in the Case Study 3 spec.
    """
    pred_set = set(predicted_tile_set)
    gt_set   = set(gt_tile_set)

    tp_set = pred_set & gt_set
    fp_set = pred_set - gt_set
    fn_set = gt_set   - pred_set

    tp  = len(tp_set)
    fp  = len(fp_set)
    fn  = len(fn_set)
    n_p = len(pred_set)
    n_g = len(gt_set)

    precision  = tp / n_p if n_p > 0 else 0.0
    recall     = tp / n_g if n_g > 0 else 0.0
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
    exact_match = pred_set == gt_set

    tile_accuracy_pct  = recall                    # TP / GT  (= recall)
    missed_tile_rate   = fn / n_g if n_g > 0 else 0.0
    extra_tile_rate    = fp / n_p if n_p > 0 else 0.0

    result = {
        "predicted_tile_sequence": predicted_tile_sequence,
        "predicted_tile_set":      sorted(pred_set),
        "gt_tile_sequence":        gt_tile_sequence,
        "gt_tile_set":             sorted(gt_set),
        "true_positive_tiles":     sorted(tp_set),
        "false_positive_tiles":    sorted(fp_set),
        "false_negative_tiles":    sorted(fn_set),
        "true_positive_count":     tp,
        "false_positive_count":    fp,
        "false_negative_count":    fn,
        "predicted_count":         n_p,
        "gt_count":                n_g,
        "precision":               round(precision,         4),
        "recall":                  round(recall,            4),
        "f1":                      round(f1,                4),
        "tile_accuracy_percent":   round(tile_accuracy_pct, 4),
        "missed_tile_rate":        round(missed_tile_rate,  4),
        "extra_tile_rate":         round(extra_tile_rate,   4),
        "exact_match":             exact_match,
        "parsing_error":           parsing_error,
        "status":                  "parsing_error" if parsing_error else "completed",
    }
    result.update(extra)
    return result


def aggregate_variant_metrics(results: list[dict]) -> dict:
    """Aggregate per-prediction results for one variant into summary metrics.

    Parameters
    ----------
    results : list of dicts returned by compute_prediction_metrics

    Returns
    -------
    Summary dict with counts, means, rates, and within-run variability.

    Notes
    -----
    The std_* fields are population standard deviations across predictions/paths
    inside this one run+variant. The downstream analysis script renames these to
    inner_run_std_* so they do not get confused with across-run std values.
    """
    n_total  = len(results)
    n_parsed = sum(1 for r in results if not r.get("parsing_error"))
    n_failed = n_total - n_parsed

    def _values(field: str) -> list[float]:
        return [
            r[field]
            for r in results
            if not r.get("parsing_error") and field in r and r[field] is not None
        ]

    def _mean(field: str) -> float | None:
        vals = _values(field)
        return round(sum(vals) / len(vals), 4) if vals else None

    def _pstdev(field: str) -> float | None:
        vals = _values(field)
        return round(statistics.pstdev(vals), 4) if vals else None

    return {
        "n_paths":               n_total,
        "n_parsed":              n_parsed,
        "n_parsing_failures":    n_failed,

        "mean_precision":        _mean("precision"),
        "mean_recall":           _mean("recall"),
        "mean_f1":               _mean("f1"),
        "exact_match_rate":      _mean("exact_match"),
        "mean_tile_accuracy":    _mean("tile_accuracy_percent"),
        "mean_missed_tile_rate": _mean("missed_tile_rate"),
        "mean_extra_tile_rate":  _mean("extra_tile_rate"),

        # Inner-run variability: std across predictions/paths inside this one run+variant.
        "std_precision":         _pstdev("precision"),
        "std_recall":            _pstdev("recall"),
        "std_f1":                _pstdev("f1"),
        "std_exact_match":       _pstdev("exact_match"),
        "std_tile_accuracy":     _pstdev("tile_accuracy_percent"),
        "std_missed_tile_rate":  _pstdev("missed_tile_rate"),
        "std_extra_tile_rate":   _pstdev("extra_tile_rate"),
    }
