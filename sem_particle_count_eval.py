"""sem_particle_count_eval.py — Case Study 2: Particle Counting via Region Navigation

Evaluates the agent's ability to:
  1. Follow a path drawn on the atlas by navigating through grid tiles.
  2. Count particles in each tile the path crosses, using VLM and/or SAM2 segmentation.
  3. Compare results across all segmentation variants (baseline, centroids, bboxes, mask,
     and all pairwise/full combinations).

Metrics collected per run:
  - path_accuracy      — fraction of expected tiles correctly visited by the agent.
  - per_tile_errors    — per-tile absolute / relative error vs ground-truth count.
  - total_error        — absolute / relative error on total particle count along the path.

Ground truth counts are left as ``None`` until filled in manually — the script handles
that gracefully and reports "not yet counted" where applicable.

Usage:
    # List available scenarios and variants
    python sem_particle_count_eval.py --list

    # Dry-run: print prompt + expected tiles, don't call the agent
    python sem_particle_count_eval.py --scenario path_horizontal --variant baseline --dry-run

    # Run one variant of one scenario and save results
    python sem_particle_count_eval.py --scenario path_horizontal --variant seg_all

    # Run all variants of a scenario
    python sem_particle_count_eval.py --scenario path_horizontal --all-variants

    # Run all scenarios x all variants
    python sem_particle_count_eval.py --all

Service URLs (override via env or flags):
    SEM_SERVICE_URL  — default http://localhost:3000
    AGENT_API_URL    — default http://localhost:3001
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request as _Req, urlopen

# ---------------------------------------------------------------------------
# Service endpoints (overridable via env)
# ---------------------------------------------------------------------------

SEM_URL   = os.environ.get("SEM_SERVICE_URL", "http://localhost:3000")
AGENT_URL = os.environ.get("AGENT_API_URL",   "http://localhost:3001")

# ---------------------------------------------------------------------------
# Segmentation variant definitions
# All 2^3 combinations of (centroids, bboxes, mask), plus the baseline (all off).
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict[str, bool]] = {
    "baseline":             {"centroids": False, "bboxes": False, "mask": False},
    "seg_centroids":        {"centroids": True,  "bboxes": False, "mask": False},
    "seg_bboxes":           {"centroids": False, "bboxes": True,  "mask": False},
    "seg_mask":             {"centroids": False, "bboxes": False, "mask": True},
    "seg_bboxes_centroids": {"centroids": True,  "bboxes": True,  "mask": False},
    "seg_bboxes_mask":      {"centroids": False, "bboxes": True,  "mask": True},
    "seg_mask_centroids":   {"centroids": True,  "bboxes": False, "mask": True},
    "seg_all":              {"centroids": True,  "bboxes": True,  "mask": True},
}

# ---------------------------------------------------------------------------
# Ground truth data structures
# ---------------------------------------------------------------------------

@dataclass
class TileGroundTruth:
    """Ground truth particle count for a single tile."""
    tile_x: int
    tile_y: int
    count: Optional[int] = None   # None = not yet counted by hand


@dataclass
class PathScenario:
    """A pre-defined path drawn on the atlas for one evaluation scenario.

    Attributes:
        name:            Unique identifier used as CLI argument and in output filenames.
        description:     Human-readable description of the path.
        region:          SEM tile dataset region number.
        fw:              Field-width value (µm) for the region.
        tile_width:      Pixel width of one tile (typically 1920).
        tile_height:     Pixel height of one tile (typically 1200).
        waypoints:       Polyline waypoints in atlas pixel coordinates.  These are drawn
                         on the atlas as a visible red line before handing off to the agent.
                         Atlas pixel coords: tile (tx, ty) spans
                           x ∈ [tx*tile_width, (tx+1)*tile_width)
                           y ∈ [ty*tile_height, (ty+1)*tile_height)
        expected_tiles:  Ordered list of (tile_x, tile_y) the path crosses.  Used both
                         to draw the path correctly and to evaluate navigation accuracy.
        tile_counts:     Per-tile ground truth counts.  Keys are (tile_x, tile_y) tuples.
                         Set to None to defer counting.
    """
    name:           str
    description:    str
    region:         int
    fw:             int
    tile_width:     int
    tile_height:    int
    waypoints:      list[tuple[float, float]]
    expected_tiles: list[tuple[int, int]]
    tile_counts:    dict[tuple[int, int], Optional[int]] = field(default_factory=dict)

    @property
    def ground_truth_total(self) -> Optional[int]:
        """Sum of all tile counts.  Returns None if any tile is uncounted."""
        counts = list(self.tile_counts.values())
        if not counts or any(c is None for c in counts):
            return None
        return sum(counts)  # type: ignore[arg-type]


def _tile_center(tx: int, ty: int, tw: int, th: int) -> tuple[float, float]:
    """Return atlas-pixel center of tile (tx, ty)."""
    return (tx * tw + tw / 2, ty * th + th / 2)


def waypoints_from_tiles(tiles: list[tuple[int, int]], tw: int, th: int) -> list[tuple[float, float]]:
    """Build a polyline through the centers of the given tiles."""
    return [_tile_center(tx, ty, tw, th) for tx, ty in tiles]


# ---------------------------------------------------------------------------
# PATH_SCENARIOS — pre-defined evaluation scenarios
#
# Add your own paths here.  Set tile_counts to the manually counted values
# once you have them; leave as None in the meantime.
#
# Tile coordinates are for Region 11, fw=120 µm (Combined_New_Scans_Andrea).
# Adjust region/fw/tile coordinates for your actual dataset.
# ---------------------------------------------------------------------------

_TW, _TH = 1920, 1200    # standard tile dimensions for this dataset

PATH_SCENARIOS: dict[str, PathScenario] = {}

def _add_scenario(s: PathScenario) -> None:
    PATH_SCENARIOS[s.name] = s


# ── Scenario 1: horizontal path through 3 tiles ────────────────────────────
_tiles_h = [(3, 14), (4, 14), (5, 14)]
_add_scenario(PathScenario(
    name="path_horizontal",
    description="Horizontal path through 3 tiles at row y=14 (region 11, fw=120µm)",
    region=11, fw=120,
    tile_width=_TW, tile_height=_TH,
    waypoints=waypoints_from_tiles(_tiles_h, _TW, _TH),
    expected_tiles=_tiles_h,
    tile_counts={
        (3, 14): None,   # TODO: count by hand
        (4, 14): None,
        (5, 14): None,
    },
))

# ── Scenario 2: diagonal path through 4 tiles ──────────────────────────────
_tiles_d = [(3, 14), (4, 15), (5, 16), (6, 17)]
_add_scenario(PathScenario(
    name="path_diagonal",
    description="Diagonal path through 4 tiles (region 11, fw=120µm)",
    region=11, fw=120,
    tile_width=_TW, tile_height=_TH,
    waypoints=waypoints_from_tiles(_tiles_d, _TW, _TH),
    expected_tiles=_tiles_d,
    tile_counts={
        (3, 14): None,
        (4, 15): None,
        (5, 16): None,
        (6, 17): None,
    },
))

# ── Scenario 3: L-shaped path through 5 tiles ──────────────────────────────
_tiles_l = [(3, 14), (4, 14), (5, 14), (5, 15), (5, 16)]
_add_scenario(PathScenario(
    name="path_lshape",
    description="L-shaped path: 3 horizontal then 2 vertical (region 11, fw=120µm)",
    region=11, fw=120,
    tile_width=_TW, tile_height=_TH,
    waypoints=waypoints_from_tiles(_tiles_l, _TW, _TH),
    expected_tiles=_tiles_l,
    tile_counts={t: None for t in _tiles_l},
))

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http(method: str, url: str, body: dict | None = None, timeout: int = 300) -> dict:
    """Fire a JSON HTTP request; return parsed response dict."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = _Req(url, data=data, headers=headers, method=method.upper())
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _sem(method: str, path: str, body: dict | None = None) -> dict:
    return _http(method, f"{SEM_URL}{path}", body)


def _agent(method: str, path: str, body: dict | None = None, timeout: int = 900) -> dict:
    return _http(method, f"{AGENT_URL}{path}", body, timeout=timeout)

# ---------------------------------------------------------------------------
# Tile intersection — compute which tiles a polyline passes through
# ---------------------------------------------------------------------------

def _tiles_for_segment(
    x1: float, y1: float,
    x2: float, y2: float,
    tw: int, th: int,
) -> list[tuple[int, int]]:
    """Return all (tile_x, tile_y) the line segment from (x1,y1)→(x2,y2) crosses.

    Uses a grid-traversal (DDA) algorithm, returning tiles in traversal order.
    """
    tiles: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    tx, ty = int(x1 // tw), int(y1 // th)
    tx1, ty1 = int(x2 // tw), int(y2 // th)

    def _add(t: tuple[int, int]) -> None:
        if t not in seen:
            seen.add(t)
            tiles.append(t)

    _add((tx, ty))
    if tx == tx1 and ty == ty1:
        return tiles

    dx = x2 - x1
    dy = y2 - y1
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1

    if dx != 0:
        t_max_x = ((tx + (1 if dx > 0 else 0)) * tw - x1) / dx
        t_delta_x = abs(tw / dx)
    else:
        t_max_x = float("inf")
        t_delta_x = float("inf")

    if dy != 0:
        t_max_y = ((ty + (1 if dy > 0 else 0)) * th - y1) / dy
        t_delta_y = abs(th / dy)
    else:
        t_max_y = float("inf")
        t_delta_y = float("inf")

    for _ in range(200):          # safety cap
        if t_max_x < t_max_y:
            tx += step_x
            t_max_x += t_delta_x
        else:
            ty += step_y
            t_max_y += t_delta_y
        _add((tx, ty))
        if (tx, ty) == (tx1, ty1):
            break

    return tiles


def tiles_for_polyline(
    waypoints: list[tuple[float, float]],
    tw: int, th: int,
) -> list[tuple[int, int]]:
    """Compute deduplicated ordered tile list for a multi-segment polyline."""
    tiles: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for (x1, y1), (x2, y2) in zip(waypoints, waypoints[1:]):
        for t in _tiles_for_segment(x1, y1, x2, y2, tw, th):
            if t not in seen:
                seen.add(t)
                tiles.append(t)
    return tiles

# ---------------------------------------------------------------------------
# Prompt generators
# ---------------------------------------------------------------------------

def _seg_description(cfg: dict[str, bool]) -> str:
    """Human-readable label for the segmentation features requested."""
    parts = []
    if cfg["mask"]:        parts.append("mask")
    if cfg["bboxes"]:      parts.append("bounding boxes")
    if cfg["centroids"]:   parts.append("centroids")
    if not parts:
        return "nothing (baseline)"
    return ", ".join(parts)


def build_prompt(scenario: PathScenario, variant_name: str) -> str:
    """Return the agent prompt for a given scenario × variant combination."""
    cfg = VARIANTS[variant_name]
    is_baseline = not any(cfg.values())

    if is_baseline:
        return f"""\
I've drawn a path in the atlas (region {scenario.region}, fw={scenario.fw}µm). \
Follow this path by navigating using the grid mode. For each tile this path enters, \
count the number of particles using your VLM. Finally return the number of \
particles along the path and for each tile.

1. Enter grid mode for region {scenario.region}, fw={scenario.fw}µm, then enter atlas mode \
   to see the drawn path.
2. Trace which tiles the drawn path enters and note their (tile_x, tile_y) coordinates.
3. Exit atlas mode and navigate to each of those tiles one by one using camera_goto.
4. For each tile: call get_canvas_image, then analyze_sandbox_image to count the \
   number of particles visible in the image.
5. Continue until all tiles along the path have been visited.
6. Return a structured summary with:
   - Each tile's coordinates and its particle count, e.g. "tile (x=3, y=14): 7 particles"
   - The total particle count across all tiles, e.g. "total: 23 particles"
"""
    else:
        seg_desc = _seg_description(cfg)
        seg_args = ", ".join(
            f"{k}=True" for k, v in cfg.items() if v
        )
        return f"""\
I've drawn a path in the atlas (region {scenario.region}, fw={scenario.fw}µm). \
Follow this path by navigating using the grid mode. For each tile this path enters, \
use your segmentation tool and VLM to count the number of particles. Finally return \
the particle count along the path and for each tile.

1. Enter grid mode for region {scenario.region}, fw={scenario.fw}µm, then enter atlas mode \
   to see the drawn path.
2. Trace which tiles the drawn path enters and note their (tile_x, tile_y) coordinates.
3. Exit atlas mode and navigate to each of those tiles one by one using camera_goto.
4. For each tile:
   a. Call segment_viewport({seg_args}) to segment the image and obtain {seg_desc}.
   b. Call get_canvas_image, then analyze_sandbox_image to count the number of \
      particles visible in the annotated image.
5. Continue until all tiles along the path have been visited.
6. Return a structured summary with:
   - Each tile's coordinates and its particle count, e.g. "tile (x=3, y=14): 7 particles"
   - The total particle count across all tiles, e.g. "total: 23 particles"
"""

# ---------------------------------------------------------------------------
# Scenario setup — draw the path on the atlas canvas
# ---------------------------------------------------------------------------

def setup_scenario(scenario: PathScenario) -> None:
    """Initialise the sem-service canvas for a scenario run.

    Steps:
      1. Reset canvas state.
      2. Init tile grid for scenario.region / scenario.fw.
      3. Enter atlas mode.
      4. Draw the path as a series of red line segments through waypoint centers.
    """
    print(f"[setup] Resetting canvas…")
    _sem("POST", "/api/canvas/new", {"width": scenario.tile_width, "height": scenario.tile_height})

    print(f"[setup] Loading tile grid — region={scenario.region}, fw={scenario.fw}…")
    _sem("POST", "/api/camera/init", {"region": scenario.region, "fw": scenario.fw})

    print(f"[setup] Entering atlas mode…")
    _sem("POST", "/api/atlas/enter", {"region": scenario.region, "fw": scenario.fw})

    print(f"[setup] Drawing path ({len(scenario.waypoints)} waypoints)…")
    wpts = scenario.waypoints
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(wpts, wpts[1:])):
        _sem("POST", "/api/draw/line", {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "stroke": "#ff3333",
            "strokeWidth": 4,
            "label": f"path_segment_{i}",
            "createdBy": "eval",
        })

    # Draw dots at each waypoint for clarity
    for i, (x, y) in enumerate(wpts):
        _sem("POST", "/api/draw/dot", {
            "cx": x, "cy": y,
            "radius": 8,
            "fill": "#ff3333",
            "stroke": "#ffffff",
            "strokeWidth": 2,
            "label": f"waypoint_{i}",
            "createdBy": "eval",
        })

    print(f"[setup] Path drawn. Expected tiles: {scenario.expected_tiles}")

# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

def reset_agent() -> None:
    """Clear agent memory so each run starts from a clean state."""
    _agent("POST", "/reset")


def run_agent(prompt: str) -> dict:
    """Send a prompt to the agent and return {"reply": str, "trace": list}."""
    return _agent("POST", "/chat", {"message": prompt})

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_visited_tiles(trace: list[dict]) -> list[tuple[int, int]]:
    """Parse agent trace to find tile positions visited via camera_goto.

    Returns a deduplicated ordered list of (tile_x, tile_y) pairs.
    Only explicit camera_goto calls are captured; relative moves are not tracked
    (the agent is instructed to use camera_goto for reproducibility).
    """
    visited: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for step in trace:
        for call in step.get("calls", []):
            if call.get("tool") != "paint_canvas":
                continue
            if call.get("action") != "camera_goto":
                continue
            summary = call.get("input_summary", "")
            # "x=3, y=14" or "x=3,y=14"
            m = re.search(r"x[=:\s]+(\d+)[,\s]+y[=:\s]+(\d+)", summary, re.IGNORECASE)
            if m:
                t = (int(m.group(1)), int(m.group(2)))
                if t not in seen:
                    seen.add(t)
                    visited.append(t)

    return visited


def extract_tile_counts_from_reply(reply: str) -> dict[tuple[int, int], int]:
    """Parse agent reply text to extract per-tile particle counts.

    Recognises patterns like:
      tile (3, 14): 7 particles
      tile (x=3, y=14): 7
      (3,14) - 7 particles
      Region (3, 14): 7
    Returns a dict mapping (tile_x, tile_y) → count.
    """
    counts: dict[tuple[int, int], int] = {}

    patterns = [
        # "tile (3, 14): 7"  or  "region (3, 14): 7"
        r"(?:tile|region)[^(]*\(\s*(\d+)\s*[,\s]+\s*(\d+)\s*\)[^\d]*(\d+)",
        # "(3, 14): 7"  or  "(3, 14) - 7"
        r"\(\s*(\d+)\s*[,\s]+\s*(\d+)\s*\)\s*[:\-]\s*(\d+)",
        # "x=3, y=14: 7"  or  "x=3, y=14 - 7"
        r"x[=\s]+(\d+)[,\s]+y[=\s]+(\d+)[^\d]*(\d+)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, reply, re.IGNORECASE):
            x, y, cnt = int(m.group(1)), int(m.group(2)), int(m.group(3))
            counts[(x, y)] = cnt

    return counts


def extract_total_count_from_reply(reply: str) -> Optional[int]:
    """Parse agent reply text for a stated total particle count."""
    patterns = [
        r"total[^\d]*(\d+)\s*particles?",
        r"(\d+)\s*particles?\s*(?:in\s+)?total",
        r"total\s*(?:count|particles?)?[:\s]+(\d+)",
        r"grand\s+total[^\d]*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, reply, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None

# ---------------------------------------------------------------------------
# Count comparison — text-based
# ---------------------------------------------------------------------------

def compare_count_text(model_text: str, ground_truth: Optional[int]) -> dict:
    """Compare the agent's stated count (extracted from text) to a ground truth value.

    This is intentionally a simple text-parsing comparison, not a semantic one.
    The first integer found in ``model_text`` is used as the model's count.

    Returns a dict with:
        model_count      — parsed int (or None if not parseable)
        ground_truth     — the supplied ground truth (or None)
        error            — |model_count - ground_truth| (or None)
        relative_error   — error / max(ground_truth, 1) (or None)
        comparable       — True if both counts were available
        note             — explanatory string when not comparable
    """
    if ground_truth is None:
        return {
            "model_count": None,
            "ground_truth": None,
            "error": None,
            "relative_error": None,
            "comparable": False,
            "note": "Ground truth not yet counted — fill in tile_counts manually.",
        }

    m = re.search(r"\d+", model_text)
    model_count = int(m.group()) if m else None

    if model_count is None:
        return {
            "model_count": None,
            "ground_truth": ground_truth,
            "error": None,
            "relative_error": None,
            "comparable": False,
            "note": "Could not parse a count from model output.",
        }

    error = abs(model_count - ground_truth)
    relative_error = error / max(ground_truth, 1)
    return {
        "model_count": model_count,
        "ground_truth": ground_truth,
        "error": error,
        "relative_error": round(relative_error, 4),
        "comparable": True,
    }

# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    scenario: PathScenario,
    visited_tiles: list[tuple[int, int]],
    tile_counts_from_agent: dict[tuple[int, int], int],
    total_count_from_agent: Optional[int],
    reply: str,
) -> dict:
    """Compute all evaluation metrics for one scenario × variant run.

    Returns a dict with:
        path_accuracy          — fraction of expected tiles correctly visited
        correct_tiles          — list of tiles correctly visited
        missed_tiles           — list of expected tiles not visited
        extra_tiles            — tiles visited but not expected
        per_tile_comparisons   — list of compare_count_text results keyed by tile
        total_comparison       — compare_count_text result for overall total
        raw_reply              — the agent's full reply text
    """
    expected = set(scenario.expected_tiles)
    visited_set = set(visited_tiles)

    correct_tiles  = sorted(expected & visited_set)
    missed_tiles   = sorted(expected - visited_set)
    extra_tiles    = sorted(visited_set - expected)
    path_accuracy  = len(correct_tiles) / max(len(expected), 1)

    # Per-tile comparisons
    per_tile = []
    for tx, ty in scenario.expected_tiles:
        gt = scenario.tile_counts.get((tx, ty))
        agent_count = tile_counts_from_agent.get((tx, ty))
        agent_text = str(agent_count) if agent_count is not None else ""
        comparison = compare_count_text(agent_text, gt)
        comparison["tile"] = (tx, ty)
        per_tile.append(comparison)

    # Total comparison — prefer agent's stated total, fall back to summing tile counts
    if total_count_from_agent is not None:
        total_agent_text = str(total_count_from_agent)
    elif tile_counts_from_agent:
        total_agent_text = str(sum(tile_counts_from_agent.values()))
    else:
        total_agent_text = ""

    total_comparison = compare_count_text(total_agent_text, scenario.ground_truth_total)

    return {
        "path_accuracy":        round(path_accuracy, 4),
        "correct_tiles":        correct_tiles,
        "missed_tiles":         missed_tiles,
        "extra_tiles":          extra_tiles,
        "per_tile_comparisons": per_tile,
        "total_comparison":     total_comparison,
        "raw_reply":            reply,
    }

# ---------------------------------------------------------------------------
# Single run orchestrator
# ---------------------------------------------------------------------------

def run_one(
    scenario: PathScenario,
    variant_name: str,
    *,
    dry_run: bool = False,
    skip_reset: bool = False,
) -> dict:
    """Set up, run, and evaluate one scenario × variant combination.

    Returns a result dict that can be JSON-serialised.
    """
    prompt = build_prompt(scenario, variant_name)

    print(f"\n{'='*70}")
    print(f"  Scenario : {scenario.name}")
    print(f"  Variant  : {variant_name}")
    print(f"  Expected : {scenario.expected_tiles}")
    print(f"{'='*70}")
    print("\n[prompt]\n" + prompt)

    if dry_run:
        print("\n[dry-run] Skipping agent call.")
        return {
            "scenario":   scenario.name,
            "variant":    variant_name,
            "dry_run":    True,
            "prompt":     prompt,
            "expected_tiles": scenario.expected_tiles,
        }

    # 1. Set up canvas / atlas
    setup_scenario(scenario)

    # 2. Reset agent memory
    if not skip_reset:
        print("[agent] Resetting agent…")
        reset_agent()

    # 3. Call agent
    print("[agent] Sending prompt…")
    response = run_agent(prompt)
    reply = response.get("reply", "")
    trace = response.get("trace", [])

    # 4. Parse
    visited      = extract_visited_tiles(trace)
    tile_counts  = extract_tile_counts_from_reply(reply)
    total_count  = extract_total_count_from_agent(reply, tile_counts)

    # 5. Metrics
    metrics = compute_metrics(scenario, visited, tile_counts, total_count, reply)

    result = {
        "scenario":         scenario.name,
        "variant":          variant_name,
        "timestamp":        datetime.now().isoformat(),
        "expected_tiles":   scenario.expected_tiles,
        "visited_tiles":    visited,
        "tile_counts_agent":  {str(k): v for k, v in tile_counts.items()},
        "total_count_agent":  total_count,
        "metrics":          metrics,
        "trace_summary":    _summarise_trace(trace),
    }

    _print_result_summary(result)
    return result


def extract_total_count_from_agent(
    reply: str,
    tile_counts: dict[tuple[int, int], int],
) -> Optional[int]:
    """Return agent's total count: prefer explicit statement, else sum of tile counts."""
    stated = extract_total_count_from_reply(reply)
    if stated is not None:
        return stated
    if tile_counts:
        return sum(tile_counts.values())
    return None


def _summarise_trace(trace: list[dict]) -> dict:
    """Count tool call categories from the trace."""
    summary: dict[str, int] = {}
    for step in trace:
        for call in step.get("calls", []):
            cat = call.get("category", "other")
            summary[cat] = summary.get(cat, 0) + 1
    return summary

# ---------------------------------------------------------------------------
# Output / reporting
# ---------------------------------------------------------------------------

def _print_result_summary(result: dict) -> None:
    m = result["metrics"]
    print(f"\n[result] path_accuracy = {m['path_accuracy']:.1%}")
    print(f"         visited        = {result['visited_tiles']}")
    print(f"         missed         = {m['missed_tiles']}")
    print(f"         extra          = {m['extra_tiles']}")
    tc = m["total_comparison"]
    if tc["comparable"]:
        print(f"         total_error    = {tc['error']} (rel: {tc['relative_error']:.1%})")
        print(f"         model_total    = {tc['model_count']}  |  gt_total = {tc['ground_truth']}")
    else:
        print(f"         total_error    = N/A  ({tc.get('note', '')})")
    for pt in m["per_tile_comparisons"]:
        tile = pt["tile"]
        if pt["comparable"]:
            print(f"         tile {tile}: model={pt['model_count']}, gt={pt['ground_truth']}, "
                  f"err={pt['error']} ({pt['relative_error']:.1%})")
        else:
            print(f"         tile {tile}: model={pt['model_count']}  ({pt.get('note', 'N/A')})")


def save_result(result: dict, out_dir: Path) -> Path:
    """Save a single run result as JSON; return the written path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = result.get("timestamp", datetime.now().isoformat()).replace(":", "-")
    fname = f"{result['scenario']}__{result['variant']}__{ts}.json"
    path = out_dir / fname

    # Serialise: convert tuple keys to strings
    def _prep(obj):
        if isinstance(obj, dict):
            return {str(k): _prep(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_prep(i) for i in obj]
        return obj

    path.write_text(json.dumps(_prep(result), indent=2))
    print(f"[saved] {path}")
    return path


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate multiple run results into a comparison table.

    Returns a dict with per-variant summary stats across all scenarios.
    """
    agg: dict[str, dict] = {}
    for r in results:
        v = r["variant"]
        if v not in agg:
            agg[v] = {
                "path_accuracy_scores": [],
                "total_relative_errors": [],
                "per_tile_relative_errors": [],
            }
        m = r["metrics"]
        agg[v]["path_accuracy_scores"].append(m["path_accuracy"])
        tc = m["total_comparison"]
        if tc["comparable"]:
            agg[v]["total_relative_errors"].append(tc["relative_error"])
        for pt in m["per_tile_comparisons"]:
            if pt["comparable"]:
                agg[v]["per_tile_relative_errors"].append(pt["relative_error"])

    summary: dict[str, dict] = {}
    for v, data in agg.items():
        def _mean(lst: list) -> Optional[float]:
            return round(sum(lst) / len(lst), 4) if lst else None

        summary[v] = {
            "mean_path_accuracy":        _mean(data["path_accuracy_scores"]),
            "mean_total_relative_error": _mean(data["total_relative_errors"]),
            "mean_tile_relative_error":  _mean(data["per_tile_relative_errors"]),
            "n_runs":                    len(data["path_accuracy_scores"]),
        }
    return summary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Case Study 2 — Particle Counting Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--scenario",      help="Name of scenario to run (see --list)")
    p.add_argument("--variant",       default="baseline", help="Variant name (see --list)")
    p.add_argument("--all-variants",  action="store_true", help="Run all variants for the chosen scenario")
    p.add_argument("--all",           action="store_true", help="Run all scenarios × all variants")
    p.add_argument("--dry-run",       action="store_true", help="Print prompt without calling the agent")
    p.add_argument("--list",          action="store_true", help="List available scenarios and variants, then exit")
    p.add_argument("--out-dir",       default="results/case_study_2", help="Directory to save result JSONs")
    p.add_argument("--sem-url",       default=SEM_URL,   help="sem-service base URL")
    p.add_argument("--agent-url",     default=AGENT_URL, help="Agent API base URL")
    p.add_argument("--skip-reset",    action="store_true",
                   help="Don't reset agent memory between runs (faster for debugging)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Override globals from flags
    global SEM_URL, AGENT_URL
    SEM_URL   = args.sem_url
    AGENT_URL = args.agent_url

    if args.list:
        print("\nAvailable scenarios:")
        for name, s in PATH_SCENARIOS.items():
            gt_filled = sum(1 for c in s.tile_counts.values() if c is not None)
            gt_total  = len(s.tile_counts)
            print(f"  {name:<25} {s.description}")
            print(f"    tiles={s.expected_tiles}  gt={gt_filled}/{gt_total} filled")
        print("\nAvailable variants:")
        for name, cfg in VARIANTS.items():
            flags = [k for k, v in cfg.items() if v] or ["none (baseline)"]
            print(f"  {name:<28} seg_features=[{', '.join(flags)}]")
        return

    out_dir = Path(args.out_dir)
    results: list[dict] = []

    if args.all:
        # All scenarios × all variants
        pairs = list(product(PATH_SCENARIOS.keys(), VARIANTS.keys()))
        print(f"Running {len(pairs)} combinations…")
        for scenario_name, variant_name in pairs:
            r = run_one(PATH_SCENARIOS[scenario_name], variant_name,
                        dry_run=args.dry_run, skip_reset=args.skip_reset)
            results.append(r)
            if not args.dry_run:
                save_result(r, out_dir)

    elif args.all_variants:
        # One scenario, all variants
        if not args.scenario:
            print("ERROR: --all-variants requires --scenario")
            sys.exit(1)
        if args.scenario not in PATH_SCENARIOS:
            print(f"ERROR: unknown scenario '{args.scenario}'. Use --list to see options.")
            sys.exit(1)
        scenario = PATH_SCENARIOS[args.scenario]
        for variant_name in VARIANTS:
            r = run_one(scenario, variant_name,
                        dry_run=args.dry_run, skip_reset=args.skip_reset)
            results.append(r)
            if not args.dry_run:
                save_result(r, out_dir)

    else:
        # Single run
        if not args.scenario:
            print("ERROR: provide --scenario NAME  (or use --all / --all-variants)")
            print("       Run with --list to see available scenarios.")
            sys.exit(1)
        if args.scenario not in PATH_SCENARIOS:
            print(f"ERROR: unknown scenario '{args.scenario}'. Use --list to see options.")
            sys.exit(1)
        if args.variant not in VARIANTS:
            print(f"ERROR: unknown variant '{args.variant}'. Use --list to see options.")
            sys.exit(1)
        r = run_one(PATH_SCENARIOS[args.scenario], args.variant,
                    dry_run=args.dry_run, skip_reset=args.skip_reset)
        results.append(r)
        if not args.dry_run:
            save_result(r, out_dir)

    # Aggregate summary (only for multi-run invocations with ground truth data)
    if len(results) > 1 and not args.dry_run:
        summary = aggregate_results(results)
        print("\n" + "="*70)
        print("AGGREGATE SUMMARY")
        print("="*70)
        for variant_name, stats in summary.items():
            print(f"  {variant_name:<28}  "
                  f"path_acc={stats['mean_path_accuracy']}  "
                  f"tile_err={stats['mean_tile_relative_error']}  "
                  f"total_err={stats['mean_total_relative_error']}  "
                  f"(n={stats['n_runs']})")
        # Save aggregate
        agg_path = out_dir / f"aggregate__{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        agg_path.write_text(json.dumps(summary, indent=2))
        print(f"\n[saved aggregate] {agg_path}")


if __name__ == "__main__":
    main()
