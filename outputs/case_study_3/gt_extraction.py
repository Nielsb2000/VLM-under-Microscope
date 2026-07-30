"""case_study_3/gt_extraction.py - Path-to-grid-cell ground-truth extraction.

Standalone pure-Python module; no agent_tools/langchain dependency.

Uses a DDA (digital differential analyser) grid-traversal algorithm to find
every atlas grid cell entered by a freehand path drawn on the SEM atlas canvas.

Grid levels
-----------
L0 / grid_level=0:
    The analysis grid equals the original SEM acquisition-tile grid.
L1 / grid_level=1:
    Each original SEM tile is subdivided into 2 x 2 virtual cells.
L2 / grid_level=2:
    Each original SEM tile is subdivided into 4 x 4 virtual cells.

In general::

    subdivision = 2 ** grid_level
    cell_width  = atlas.tileWidth  / subdivision
    cell_height = atlas.tileHeight / subdivision

The returned IDs are always global atlas-grid coordinates in the format
``(x,y)``, with x increasing left-to-right and y increasing top-to-bottom.
For compatibility with the original runner, legacy ``tile`` field names are
kept, but the preferred interpretation is ``cell`` for grid_level > 0.
"""
from __future__ import annotations

import math
import re


# ---------------------------------------------------------------------------
# Coordinate-ID conversion
# ---------------------------------------------------------------------------

def tile_id(tile_x: int, tile_y: int) -> str:
    """Convert 0-based (x, y) coordinates to an ID like '(0,0)'."""
    return f"({tile_x},{tile_y})"


def grid_metadata(atlas: dict, grid_level: int = 0) -> dict:
    """Return derived analysis-grid metadata for an atlas and grid level."""
    level = max(0, int(grid_level or 0))
    subdivision = 2 ** level

    base_tw = float(atlas.get("tileWidth", 1920))
    base_th = float(atlas.get("tileHeight", 1200))
    cols = int(atlas.get("cols", 0) or 0)
    rows = int(atlas.get("rows", 0) or 0)

    cell_w = base_tw / subdivision
    cell_h = base_th / subdivision
    effective_cols = cols * subdivision
    effective_rows = rows * subdivision

    return {
        "grid_level": level,
        "subdivision": subdivision,
        "base_tile_width": base_tw,
        "base_tile_height": base_th,
        "cell_width": cell_w,
        "cell_height": cell_h,
        "effective_cols": effective_cols,
        "effective_rows": effective_rows,
        "coordinate_system": "(x,y) 0-based global analysis-grid coordinates: x=column left-to-right, y=row top-to-bottom",
    }


# ---------------------------------------------------------------------------
# DDA grid traversal
# ---------------------------------------------------------------------------

def _clamp_cell(tx: int, ty: int, max_cols: int, max_rows: int) -> tuple[int, int]:
    if max_cols > 0:
        tx = max(0, min(tx, max_cols - 1))
    if max_rows > 0:
        ty = max(0, min(ty, max_rows - 1))
    return tx, ty


def _cell_for_point(
    x: float,
    y: float,
    cell_w: float,
    cell_h: float,
    max_cols: int,
    max_rows: int,
) -> tuple[int, int]:
    tx = int(math.floor(x / cell_w)) if cell_w else 0
    ty = int(math.floor(y / cell_h)) if cell_h else 0
    return _clamp_cell(tx, ty, max_cols, max_rows)


def _tiles_for_segment_dda(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cell_w: float,
    cell_h: float,
    max_cols: int,
    max_rows: int,
) -> list[tuple[int, int]]:
    """Return every (cell_x, cell_y) the segment (x1,y1)->(x2,y2) enters."""
    tiles: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def _add(t: tuple[int, int]) -> None:
        t = _clamp_cell(t[0], t[1], max_cols, max_rows)
        if t not in seen:
            seen.add(t)
            tiles.append(t)

    tx, ty = _cell_for_point(x1, y1, cell_w, cell_h, max_cols, max_rows)
    tx1, ty1 = _cell_for_point(x2, y2, cell_w, cell_h, max_cols, max_rows)
    _add((tx, ty))
    if tx == tx1 and ty == ty1:
        return tiles

    dx, dy = x2 - x1, y2 - y1
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1

    t_mx = ((tx + (1 if dx > 0 else 0)) * cell_w - x1) / dx if dx else float("inf")
    t_my = ((ty + (1 if dy > 0 else 0)) * cell_h - y1) / dy if dy else float("inf")
    td_x = abs(cell_w / dx) if dx else float("inf")
    td_y = abs(cell_h / dy) if dy else float("inf")

    # Bound iteration count by grid size, with margin for clamping/edge cases.
    max_steps = max(500, (max_cols + max_rows + 4) * 4)
    for _ in range(max_steps):
        if t_mx < t_my:
            tx += sx
            t_mx += td_x
        else:
            ty += sy
            t_my += td_y
        _add((tx, ty))
        if (tx, ty) == (tx1, ty1):
            break

    return tiles


# ---------------------------------------------------------------------------
# SVG/Fabric path parsing
# ---------------------------------------------------------------------------

def _parse_svg_path_waypoints(path_str: str) -> list[tuple[float, float]]:
    """Extract ordered (x, y) waypoints from a Fabric.js SVG path string.

    Handles M, L, Q, C, Z and their relative lowercase variants. Curves are
    sampled at t=0.25/0.5/0.75/1.0, matching the original CS3 implementation.
    """
    pts: list[tuple[float, float]] = []
    tokens = re.findall(
        r"[MLQCZmlqcz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        path_str or "",
    )
    i = 0
    cx = cy = sx = sy = 0.0

    while i < len(tokens):
        cmd = tokens[i]
        i += 1
        if cmd in ("M", "m"):
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if cmd == "m":
                x += cx
                y += cy
            cx = sx = x
            cy = sy = y
            pts.append((cx, cy))
        elif cmd in ("L", "l"):
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if cmd == "l":
                x += cx
                y += cy
            cx, cy = x, y
            pts.append((cx, cy))
        elif cmd in ("Q", "q"):
            x1, y1 = float(tokens[i]), float(tokens[i + 1])
            i += 2
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if cmd == "q":
                x1 += cx
                y1 += cy
                x += cx
                y += cy
            for t in (0.25, 0.5, 0.75, 1.0):
                pts.append((
                    (1 - t) ** 2 * cx + 2 * (1 - t) * t * x1 + t ** 2 * x,
                    (1 - t) ** 2 * cy + 2 * (1 - t) * t * y1 + t ** 2 * y,
                ))
            cx, cy = x, y
        elif cmd in ("C", "c"):
            x1, y1 = float(tokens[i]), float(tokens[i + 1])
            i += 2
            x2, y2 = float(tokens[i]), float(tokens[i + 1])
            i += 2
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if cmd == "c":
                x1 += cx
                y1 += cy
                x2 += cx
                y2 += cy
                x += cx
                y += cy
            for t in (0.25, 0.5, 0.75, 1.0):
                pts.append((
                    (1 - t) ** 3 * cx
                    + 3 * (1 - t) ** 2 * t * x1
                    + 3 * (1 - t) * t ** 2 * x2
                    + t ** 3 * x,
                    (1 - t) ** 3 * cy
                    + 3 * (1 - t) ** 2 * t * y1
                    + 3 * (1 - t) * t ** 2 * y2
                    + t ** 3 * y,
                ))
            cx, cy = x, y
        elif cmd in ("Z", "z"):
            pts.append((sx, sy))
        else:
            # Unknown command; stop rather than silently producing nonsense.
            break

    return pts


def _freehand_canvas_points(obj: dict) -> list[tuple[float, float]]:
    """Return freehand waypoints in atlas/export coordinates.

    The renderer draws historical freehand paths directly in export-ready
    coordinates and only applies left/top when ``coordMode == 'local'``. Ground
    truth follows the same rule to avoid a renderer/GT mismatch.
    """
    pts = _parse_svg_path_waypoints(obj.get("path", ""))
    if not pts:
        return []

    if obj.get("coordMode") == "local":
        left = float(obj.get("left") or 0)
        top = float(obj.get("top") or 0)
        min_x = min(p[0] for p in pts)
        min_y = min(p[1] for p in pts)
        return [(left + px - min_x, top + py - min_y) for px, py in pts]

    return pts


# ---------------------------------------------------------------------------
# Object traversal
# ---------------------------------------------------------------------------

def _traversed_tiles_for_obj(
    obj: dict,
    cell_w: float,
    cell_h: float,
    max_cols: int,
    max_rows: int,
) -> list[tuple[int, int]]:
    """Return every (cell_x, cell_y) entered by one canvas object."""
    otype = obj.get("type", "")
    tiles: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def _merge(new_tiles: list[tuple[int, int]]) -> None:
        for t in new_tiles:
            t = _clamp_cell(t[0], t[1], max_cols, max_rows)
            if t not in seen:
                seen.add(t)
                tiles.append(t)

    def _seg(x1: float, y1: float, x2: float, y2: float) -> None:
        _merge(_tiles_for_segment_dda(x1, y1, x2, y2, cell_w, cell_h, max_cols, max_rows))

    if otype in ("line", "arrow"):
        _seg(float(obj.get("x1", 0)), float(obj.get("y1", 0)), float(obj.get("x2", 0)), float(obj.get("y2", 0)))
    elif otype in ("dot", "text"):
        cx = float(obj.get("cx", obj.get("x", 0)))
        cy = float(obj.get("cy", obj.get("y", 0)))
        _merge([_cell_for_point(cx, cy, cell_w, cell_h, max_cols, max_rows)])
    elif otype == "rect":
        x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
        w, h = float(obj.get("width", 0)), float(obj.get("height", 0))
        for x1, y1, x2, y2 in [
            (x, y, x + w, y),
            (x + w, y, x + w, y + h),
            (x + w, y + h, x, y + h),
            (x, y + h, x, y),
        ]:
            _seg(x1, y1, x2, y2)
    elif otype == "ellipse":
        ecx = float(obj.get("cx", 0))
        ecy = float(obj.get("cy", 0))
        rx = float(obj.get("rx", 0))
        ry = float(obj.get("ry", 0))
        n = max(32, int(2 * math.pi * max(rx, ry) / max(1.0, min(cell_w, cell_h)) * 8))
        pts = [
            (ecx + rx * math.cos(2 * math.pi * k / n), ecy + ry * math.sin(2 * math.pi * k / n))
            for k in range(n)
        ]
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + [pts[0]]):
            _seg(ax, ay, bx, by)
    elif otype == "freehand":
        pts = _freehand_canvas_points(obj)
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            _seg(ax, ay, bx, by)

    return tiles


# ---------------------------------------------------------------------------
# Public GT extraction
# ---------------------------------------------------------------------------

def gt_from_objects(objects: list[dict], atlas: dict, grid_level: int = 0) -> dict:
    """Compute deterministic GT grid cells from canvas objects and atlas geometry.

    Parameters
    ----------
    objects:
        Canvas objects from ``GET /api/objects``.
    atlas:
        From ``GET /api/camera/state`` -> ``.atlas``. Must contain
        ``tileWidth``, ``tileHeight``, ``cols`` and ``rows``.
    grid_level:
        0 = original SEM tile grid, 1 = 2x2 subdivision per tile,
        2 = 4x4 subdivision per tile, etc.

    Returns
    -------
    dict with both preferred cell names and legacy tile names:
        gt_cell_sequence, gt_cell_set, gt_tile_sequence, gt_tile_set, path_points.
    """
    meta = grid_metadata(atlas, grid_level=grid_level)
    cell_w = float(meta["cell_width"])
    cell_h = float(meta["cell_height"])
    effective_cols = int(meta["effective_cols"])
    effective_rows = int(meta["effective_rows"])

    sequence: list[str] = []
    raw_traversal: list[list[int]] = []

    for obj in objects:
        turn_info = l0_turn_analysis_from_objects(objects, atlas)
        traversed = _traversed_tiles_for_obj(obj, cell_w, cell_h, effective_cols, effective_rows)
        for tx, ty in traversed:
            raw_traversal.append([tx, ty])
            tid = tile_id(tx, ty)
            if not sequence or sequence[-1] != tid:
                sequence.append(tid)

    cell_set = sorted(set(sequence), key=_sort_cell_id)
    return {
        **meta,
        **turn_info,
        "gt_cell_sequence": sequence,
        "gt_cell_set": cell_set,
        # Backward-compatible names used by existing metrics code.
        "gt_tile_sequence": sequence,
        "gt_tile_set": cell_set,
        "path_points": raw_traversal,
    }


def _sort_cell_id(cid: str) -> tuple[int, int]:
    m = re.match(r"\((\d+),(\d+)\)", cid)
    if not m:
        return (0, 0)
    return (int(m.group(2)), int(m.group(1)))


# ---------------------------------------------------------------------------
# L0 tile-turn difficulty
# ---------------------------------------------------------------------------

def _dedupe_consecutive(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove only consecutive duplicate coordinates."""
    out: list[tuple[int, int]] = []
    for c in coords:
        if not out or out[-1] != c:
            out.append(c)
    return out


def _movement_axis(a: tuple[int, int], b: tuple[int, int]) -> str | None:
    """Return movement axis between two original-tile coordinates.

    A 90-degree turn is counted when the movement changes between horizontal
    and vertical in the original SEM tile grid, not the virtual cell grid.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    if dx == 0 and dy == 0:
        return None

    if dx != 0 and dy == 0:
        return "horizontal"

    if dx == 0 and dy != 0:
        return "vertical"

    # Defensive fallback. The DDA traversal should usually produce adjacent
    # horizontal/vertical tile steps, but this prevents crashes on odd input.
    return "horizontal" if abs(dx) >= abs(dy) else "vertical"


def count_l0_tile_turns(tile_coords: list[tuple[int, int]]) -> int:
    """Count 90-degree turns in an original SEM-tile traversal sequence."""
    coords = _dedupe_consecutive(tile_coords)

    axes: list[str] = []
    for a, b in zip(coords, coords[1:]):
        axis = _movement_axis(a, b)
        if axis is not None:
            axes.append(axis)

    if not axes:
        return 0

    # Compress repeated movement along the same axis.
    compressed_axes: list[str] = []
    for axis in axes:
        if not compressed_axes or compressed_axes[-1] != axis:
            compressed_axes.append(axis)

    return max(0, len(compressed_axes) - 1)


def classify_turn_difficulty(turn_count: int) -> str:
    """Map number of 90-degree turns to the case-study difficulty category."""
    if turn_count == 0:
        return "straight"
    if turn_count == 1:
        return "easy"
    if turn_count == 2:
        return "medium"
    if turn_count == 3:
        return "hard"
    return "very_hard"


def l0_turn_analysis_from_objects(objects: list[dict], atlas: dict) -> dict:
    """Compute turn count/difficulty using the original SEM tile grid only.

    This intentionally ignores L1/L2 virtual cells. The grid is the larger
    SEM acquisition-tile grid, e.g. 1920x1080 px per tile when that is what
    the atlas metadata reports.
    """
    meta = grid_metadata(atlas, grid_level=0)

    tile_w = float(meta["base_tile_width"])
    tile_h = float(meta["base_tile_height"])
    cols = int(atlas.get("cols", 0) or 0)
    rows = int(atlas.get("rows", 0) or 0)

    sequence_coords: list[tuple[int, int]] = []

    for obj in objects:
        traversed = _traversed_tiles_for_obj(
            obj,
            tile_w,
            tile_h,
            cols,
            rows,
        )

        for coord in traversed:
            if not sequence_coords or sequence_coords[-1] != coord:
                sequence_coords.append(coord)

    turn_count = count_l0_tile_turns(sequence_coords)

    return {
        "turn_count": turn_count,
        "turn_difficulty": classify_turn_difficulty(turn_count),
        "turn_tile_sequence_l0": [tile_id(x, y) for x, y in sequence_coords],
        "turn_coordinate_system": (
            "original SEM acquisition-tile grid; turns counted as horizontal/vertical "
            "axis changes between consecutive L0 tiles"
        ),
        "turn_base_tile_width": tile_w,
        "turn_base_tile_height": tile_h,
    }