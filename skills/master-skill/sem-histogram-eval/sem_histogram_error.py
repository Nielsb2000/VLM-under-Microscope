# =============================================================================
# ⚠️  DUAL-COPY FILE — KEEP BOTH COPIES IDENTICAL
#
# This file exists in TWO locations:
#   1. sem_histogram_error.py                                (project root — human/CI runs)
#   2. skills/master-skill/sem-histogram-eval/sem_histogram_error.py  (sandbox — agent runs)
#
# The sandbox copy is the ONLY one the agent can execute.
# The root copy is the canonical version for human and CI use.
#
# ANY change to either file MUST be applied to the other immediately.
# The only intentional difference is the _HIST_RESULT path resolution
# block (sandbox vs host fallback) — everything else must be character-
# for-character identical.
# =============================================================================

"""sem_histogram_error.py — SEM image quality evaluation via histogram comparison.

Computes the SEM Histogram Error between a rendered canvas image and a reference
brightness histogram:

    SEM histogram error = W(image_hist, ref_hist) + lambda * clipping_fraction

where:
  - W                  is the Wasserstein (Earth Mover's) distance between the two
                       normalised brightness histograms.
  - lambda             is the clipping penalty weight (default 5.0).
  - clipping_fraction  is the normalised histogram mass in the first and last
                       ``edge_bins`` bins (default 5), measuring how much the image
                       is clipped to pure black or pure white.

Lower scores are better.  A score of 0 means identical brightness distribution
and no clipping.

Usage (standalone script):
    python sem_histogram_error.py [--paint-url http://localhost:3000]
                                  [--edge-bins 5] [--clipping-weight 5.0]

The script fetches:
  GET /api/histogram/current    — histogram of the canvas with current filters applied
  GET /api/histogram/reference  — histogram captured before randomisation

Saves to sem-service/histograms/result/:
  result_hist.json     — current histogram + metric scores
  result_hist.png      — standalone result brightness histogram
  ref_hist.png         — standalone reference histogram (copy for side-by-side viewing)
  comparison_hist.png  — result bars with reference overlaid in red

This script is intentionally separate from the agent loop — the AGENT is not
allowed to call this and must rely solely on its visual (VLM) assessment to
decide when the image quality is good.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR   = Path(__file__).parent
# When running inside the AIO Sandbox, sem-service/histograms is mounted at
# /workspace/histograms.  Running on the host the script falls back to the
# sem-service/histograms path two levels above the skills tree.
_SANDBOX_HIST = Path("/workspace/histograms")
_HOST_HIST    = Path(__file__).resolve().parents[3] / "sem-service" / "histograms"
_HIST_RESULT  = (_SANDBOX_HIST if _SANDBOX_HIST.exists() else _HOST_HIST) / "result"


# ---------------------------------------------------------------------------
# Histogram PNG helper
# ---------------------------------------------------------------------------

def _save_histogram_png(
    bins: list,
    out_path: Path,
    title: str = "Brightness Histogram",
    ref_bins: list | None = None,
    score_label: str | None = None,
) -> None:
    """Save a brightness histogram bar chart as a PNG.

    If ``ref_bins`` is supplied, the reference is overlaid as a step line so
    the two distributions can be compared visually in the result plot.
    If ``score_label`` is supplied, it is shown as a legend entry on the plot.
    """
    n = len(bins)
    total = sum(bins) or 1
    p = [b / total for b in bins]
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Colour-gradient bars (dark → light)
    colours = [f"#{v:02x}{v:02x}{v:02x}" for v in [int(35 + 220 * i / (n - 1)) for i in range(n)]]
    ax.bar(x, p, width=1.0, color=colours, linewidth=0)

    legend_handles = []
    if score_label is not None:
        legend_handles.append(
            matplotlib.patches.Patch(color="none", label=score_label)
        )
    if ref_bins is not None:
        ref_total = sum(ref_bins) or 1
        p_ref = [b / ref_total for b in ref_bins]
        ax.step(x, p_ref, where="mid", color="#e53935", linewidth=1.5, label="Reference", alpha=0.9)
        legend_handles.append(
            matplotlib.lines.Line2D([0], [0], color="#e53935", linewidth=1.5, label="Reference")
        )
    if legend_handles:
        ax.legend(handles=legend_handles, facecolor="#2a2a4e", edgecolor="#555", labelcolor="white", fontsize=9)

    ax.set_xlim(0, n - 1)
    ax.set_xlabel("Brightness", color="#cccccc", fontsize=11)
    ax.set_ylabel("Probability", color="#cccccc", fontsize=11)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.set_title(title, color="#e0e0e0", fontsize=11, fontweight="bold", pad=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def sem_histogram_error(
    image_bins: list[int] | list[float],
    ref_bins:   list[int] | list[float],
    *,
    edge_bins:       int   = 5,
    clipping_weight: float = 5.0,
) -> dict:
    """Compute the SEM Histogram Error between an image and a reference histogram.

    Parameters
    ----------
    image_bins:
        Raw histogram bin counts for the image being evaluated.
    ref_bins:
        Raw histogram bin counts for the reference (un-modified) image.
    edge_bins:
        Number of bins at each extreme (black / white) counted as clipping.
    clipping_weight (lambda):
        Penalty multiplier for the clipping fraction.

    Returns
    -------
    dict with keys:
        ``score``               — the SEM histogram error (lower is better)
        ``wasserstein``         — W distance component
        ``clipping_fraction``   — mass in clipped bins
        ``clipping_penalty``    — lambda * clipping_fraction
        ``n_bins``              — number of bins (should be 256 for 8-bit)
    """
    n = len(image_bins)
    if n == 0:
        return {"error": "image_bins is empty"}
    if len(ref_bins) != n:
        return {"error": f"Histogram lengths differ: {n} vs {len(ref_bins)}"}

    total_img = sum(image_bins)
    total_ref = sum(ref_bins)
    if total_img == 0:
        return {"error": "image_bins sums to zero — cannot normalise"}
    if total_ref == 0:
        return {"error": "ref_bins sums to zero — cannot normalise"}

    # Normalise to probability distributions
    p_img = [b / total_img for b in image_bins]
    p_ref = [b / total_ref for b in ref_bins]

    # Wasserstein distance (1D) via CDF difference
    # W1 = sum |CDF_img(i) - CDF_ref(i)| / (n - 1)   (divided by bin range)
    # We follow the standard unnormalised 1D discrete formula:
    # W1 = sum_i |CDF_img(i) - CDF_ref(i)|
    # but normalised by (n - 1) so it is expressed in [0, 1] regardless of n.
    cdf_img = cdf_ref = 0.0
    wasserstein = 0.0
    for i in range(n):
        cdf_img += p_img[i]
        cdf_ref += p_ref[i]
        wasserstein += abs(cdf_img - cdf_ref)
    wasserstein /= (n - 1)

    # Clipping fraction — mass in the first and last edge_bins bins of the *image*
    clipping_fraction = sum(p_img[:edge_bins]) + sum(p_img[max(0, n - edge_bins):])

    clipping_penalty = clipping_weight * clipping_fraction
    score = wasserstein + clipping_penalty

    return {
        "score":             round(score, 6),
        "wasserstein":       round(wasserstein, 6),
        "clipping_fraction": round(clipping_fraction, 6),
        "clipping_penalty":  round(clipping_penalty, 6),
        "n_bins":            n,
    }


# ---------------------------------------------------------------------------
# CLI / standalone runner
# ---------------------------------------------------------------------------

def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SEM image quality via brightness histogram comparison."
    )
    parser.add_argument(
        "--paint-url",
        default="http://localhost:3000",
        help="Base URL of the sem-service (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--edge-bins",
        type=int,
        default=5,
        help="Number of edge bins to consider as clipping (default: 5)",
    )
    parser.add_argument(
        "--clipping-weight",
        type=float,
        default=5.0,
        help="Clipping penalty weight lambda (default: 5.0)",
    )
    args = parser.parse_args(argv)

    base = args.paint_url.rstrip("/")

    print(f"Fetching current histogram from {base}/api/histogram/current …")
    try:
        current = _fetch_json(f"{base}/api/histogram/current")
    except URLError as e:
        print(f"ERROR: Could not reach sem-service: {e}", file=sys.stderr)
        sys.exit(1)

    if "error" in current:
        print(f"ERROR from /api/histogram/current: {current['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching reference histogram from {base}/api/histogram/reference …")
    try:
        reference = _fetch_json(f"{base}/api/histogram/reference")
    except URLError as e:
        print(f"ERROR: Could not reach sem-service: {e}", file=sys.stderr)
        sys.exit(1)

    if "error" in reference:
        print(f"ERROR from /api/histogram/reference: {reference['error']}", file=sys.stderr)
        sys.exit(1)

    # Fetch the randomized histogram (optional — saved at randomize time)
    randomized: dict | None = None
    try:
        randomized = _fetch_json(f"{base}/api/histogram/randomized")
        if "error" in randomized:
            randomized = None
    except Exception:
        randomized = None

    result = sem_histogram_error(
        current["bins"],
        reference["bins"],
        edge_bins=args.edge_bins,
        clipping_weight=args.clipping_weight,
    )

    # Score the randomized state too (how hard was the challenge?)
    result_rand: dict = {}
    if randomized:
        result_rand = sem_histogram_error(
            randomized["bins"],
            reference["bins"],
            edge_bins=args.edge_bins,
            clipping_weight=args.clipping_weight,
        )

    if "error" in result:
        print(f"ERROR computing metric: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print("\n── SEM Histogram Error ──────────────────────────────────────")
    print("  Evaluation complete. Score saved to result files (not shown here).")
    print(f"  Histogram bins          : {result['n_bins']}")
    print(f"  Reference image         : {reference.get('filename', '?')}")
    print(f"  Reference captured at   : {reference.get('capturedAt', '?')}")
    if reference.get('randomFilters'):
        print(f"  Random filters applied  : [withheld — see result JSON]")
    print("─────────────────────────────────────────────────────────────\n")

    # Fetch final filter values and session iteration counts
    try:
        session_stats  = _fetch_json(f"{base}/api/session/stats")
        final_filters  = session_stats.get('currentFilters', {})
        filter_adjustments = session_stats.get('filterAdjustments', 'n/a')
        vlm_snapshots      = session_stats.get('vlmSnapshots', 'n/a')
    except Exception as e:
        print(f"  Warning: could not fetch session stats: {e}", file=sys.stderr)
        final_filters = {}; filter_adjustments = 'n/a'; vlm_snapshots = 'n/a'

    print(f"  Final filters           : brightness={final_filters.get('brightness','?')}  contrast={final_filters.get('contrast','?')}")
    print(f"  Agent VLM snapshots     : {vlm_snapshots}")
    print(f"  Agent filter adjustments: {filter_adjustments}")

    # --- Save result histogram JSON + PNGs ---
    from datetime import datetime as _dt
    try:
        from zoneinfo import ZoneInfo as _ZI
        ts = _dt.now(_ZI("Europe/Amsterdam")).strftime("%Y%m%d_%H%M")
    except Exception:
        ts = _dt.now().strftime("%Y%m%d_%H%M")

    _HIST_RESULT.mkdir(parents=True, exist_ok=True)
    json_path      = _HIST_RESULT / f"result_hist_{ts}.json"
    png_result     = _HIST_RESULT / f"result_hist_{ts}.png"
    png_reference  = _HIST_RESULT / f"ref_hist_{ts}.png"
    png_comparison = _HIST_RESULT / f"comparison_hist_{ts}.png"

    result_payload = {
        **result,
        "bins":               current["bins"],
        "capturedAt":         current.get("capturedAt", ""),
        "edge_bins":          args.edge_bins,
        "clipping_weight":    args.clipping_weight,
        "finalFilters":       final_filters,
        "randomFilters":      reference.get("randomFilters", {}),
        "filterAdjustments":  filter_adjustments,
        "vlmSnapshots":       vlm_snapshots,
        "referenceImage":     reference.get("filename", ""),
        "referenceCapturedAt": reference.get("capturedAt", ""),
        "randomizedScore":    result_rand.get("score") if result_rand else None,
        "randomizedBins":     randomized["bins"] if randomized else None,
    }
    json_path.write_text(json.dumps(result_payload, indent=2))
    print(f"  Result JSON saved    → {json_path}")

    # Standalone: result histogram only
    _quality = (
        "excellent" if result['score'] < 0.05 else
        "good"      if result['score'] < 0.15 else
        "fair"      if result['score'] < 0.40 else
        "poor"
    )
    _save_histogram_png(
        current["bins"],
        png_result,
        title="Result Histogram",
        score_label=f"score = {result['score']:.4f}  ({_quality})  |  0–0.05 excellent  0.05–0.15 good  0.15–0.40 fair  >0.40 poor",
    )
    print(f"  Result PNG saved     → {png_result}")

    # Standalone: reference histogram copy in result folder for easy side-by-side
    _save_histogram_png(
        reference["bins"],
        png_reference,
        title=f"Reference Histogram  ({reference.get('filename', '?')})",
    )
    print(f"  Reference PNG saved  → {png_reference}")

    # Standalone: randomized histogram (the distorted starting point)
    png_randomized = _HIST_RESULT / f"rand_hist_{ts}.png"
    if randomized:
        _rand_quality = (
            "excellent" if result_rand.get('score', 1) < 0.05 else
            "good"      if result_rand.get('score', 1) < 0.15 else
            "fair"      if result_rand.get('score', 1) < 0.40 else
            "poor"
        )
        _save_histogram_png(
            randomized["bins"],
            png_randomized,
            title="Randomized Histogram  (agent starting point)",
            score_label=f"challenge score = {result_rand.get('score', 0):.4f}  ({_rand_quality})",
        )
        print(f"  Randomized PNG saved → {png_randomized}")

    # Comparison: result bars + reference overlay
    _save_histogram_png(
        current["bins"],
        png_comparison,
        title="Comparison: result vs reference",
        ref_bins=reference["bins"],
        score_label=f"score = {result['score']:.4f}  ({_quality})  |  0–0.05 excellent  0.05–0.15 good  0.15–0.40 fair  >0.40 poor",
    )
    print(f"  Comparison PNG saved → {png_comparison}")

    # --- Export actual rendered images (not histogram charts) ---
    from urllib.request import urlopen as _urlopen
    img_result_path    = _HIST_RESULT / f"result_img_{ts}.png"
    img_reference_path = _HIST_RESULT / f"ref_img_{ts}.png"
    img_randomized_path = _HIST_RESULT / f"rand_img_{ts}.png"

    # Result image: current canvas render with agent's final filters (no objects/segmentation)
    try:
        img_result_bytes = _urlopen(f"{base}/api/histogram/result-image", timeout=30).read()
        img_result_path.write_bytes(img_result_bytes)
        result_payload["resultImage"] = str(img_result_path)
        print(f"  Result image saved   → {img_result_path}")
    except Exception as _e:
        print(f"  Warning: could not fetch result image: {_e}", file=sys.stderr)

    # Reference image: background rendered at neutral filters (ground truth)
    try:
        img_ref_bytes = _urlopen(f"{base}/api/histogram/reference-image", timeout=15).read()
        img_reference_path.write_bytes(img_ref_bytes)
        result_payload["referenceImageFile"] = str(img_reference_path)
        print(f"  Reference image saved→ {img_reference_path}")
    except Exception as _e:
        print(f"  Warning: could not fetch reference image: {_e}", file=sys.stderr)

    # Randomized image: canvas render captured at the moment of randomization
    try:
        img_rand_bytes = _urlopen(f"{base}/api/histogram/randomized-image", timeout=15).read()
        img_randomized_path.write_bytes(img_rand_bytes)
        result_payload["randomizedImageFile"] = str(img_randomized_path)
        print(f"  Randomized img saved → {img_randomized_path}")
    except Exception as _e:
        print(f"  Warning: could not fetch randomized image: {_e}", file=sys.stderr)

    # Update JSON with image paths now that they are resolved
    json_path.write_text(json.dumps(result_payload, indent=2))

    # Overwrite plain-name "latest" copies so the most recent run is always
    # at the predictable path — the timestamped files are the permanent archive.
    import shutil as _shutil
    latest_copies = [
        (json_path,      _HIST_RESULT / "result_hist.json"),
        (png_result,     _HIST_RESULT / "result_hist.png"),
        (png_reference,  _HIST_RESULT / "ref_hist.png"),
        (png_comparison, _HIST_RESULT / "comparison_hist.png"),
    ]
    if randomized:
        latest_copies.append((png_randomized, _HIST_RESULT / "rand_hist.png"))
    if img_result_path.exists():
        latest_copies.append((img_result_path,     _HIST_RESULT / "result_img.png"))
    if img_reference_path.exists():
        latest_copies.append((img_reference_path,  _HIST_RESULT / "ref_img.png"))
    if img_randomized_path.exists():
        latest_copies.append((img_randomized_path, _HIST_RESULT / "rand_img.png"))
    for src, dst in latest_copies:
        _shutil.copy2(src, dst)
    print(f"  Latest copies updated in {_HIST_RESULT}\n")


if __name__ == "__main__":
    main()
