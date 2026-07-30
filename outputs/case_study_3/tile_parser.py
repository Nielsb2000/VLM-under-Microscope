"""case_study_3/tile_parser.py - Parse a VLM text response into grid-cell IDs.

The model is asked to return JSON like::

    {"cells_entered": ["(0,0)", "(1,0)", "(1,1)"]}

For backward compatibility with the original Case Study 3, the parser also
accepts::

    {"tiles_entered": ["(0,0)", "(1,0)", "(1,1)"]}

Supported input formats
-----------------------
1. Valid JSON object with a ``cells_entered`` or ``tiles_entered`` key.
2. Markdown code block (```json ... ```) containing the above JSON.
3. Comma-separated coordinate IDs, e.g. ``(0,0), (1,0), (1,1)``.
4. Bullet or numbered list with one coordinate ID per line.
5. Natural language containing coordinate ID patterns.

Normalisation
-------------
* Spaces inside the parentheses are stripped.
* Duplicates are removed while preserving first-occurrence order.
* Only patterns matching ``(\d+,\d+)`` are accepted as coordinate IDs.

Returns
-------
parse_tile_ids(text) -> dict:
    {
        "ok": bool,
        "predicted_tile_sequence": list[str],  # legacy name, ordered, deduped
        "predicted_tile_set": list[str],       # legacy name, sorted unique
        "predicted_cell_sequence": list[str],  # preferred name
        "predicted_cell_set": list[str],       # preferred name
        "parsing_error": str | None,
    }
"""
from __future__ import annotations

import json
import re
from typing import Any


# Regex for a single coordinate ID: (x,y), with optional whitespace around digits.
_TILE_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_COORD_KEYS = ("cells_entered", "tiles_entered")


def _normalise_id(raw: str) -> str | None:
    """Normalise a raw coordinate ID string to canonical '(x,y)' form."""
    m = _TILE_RE.fullmatch(raw.strip())
    if m:
        return f"({m.group(1)},{m.group(2)})"

    # Also accept plain 'x,y' without parentheses.
    m2 = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", raw.strip())
    if m2:
        return f"({m2.group(1)},{m2.group(2)})"
    return None


def _normalise_ids(raw: list[Any]) -> list[str]:
    """Normalise and deduplicate a list of raw coordinate ID values."""
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        tid = _normalise_id(str(item))
        if tid is not None and tid not in seen:
            seen.add(tid)
            result.append(tid)
    return result


def _coordinate_list_from_json_obj(obj: Any) -> list[Any] | None:
    """Return the first supported coordinate-list field from a JSON object."""
    if not isinstance(obj, dict):
        return None
    for key in _COORD_KEYS:
        value = obj.get(key)
        if isinstance(value, list):
            return value
    return None


def _extract_tile_ids_from_text(text: str) -> list[str]:
    """Extract all coordinate-ID patterns from arbitrary text, preserving order."""
    found = _TILE_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for x, y in found:
        tid = f"({x},{y})"
        if tid not in seen:
            seen.add(tid)
            result.append(tid)
    return result


def parse_tile_ids(text: str) -> dict:
    """Parse a VLM response and return structured coordinate-ID results."""
    if not text or not text.strip():
        return _failure("Empty response")

    stripped = text.strip()

    # Strategy 1: direct JSON parse.
    try:
        obj = json.loads(stripped)
        raw = _coordinate_list_from_json_obj(obj)
        if raw is not None:
            return _success(_normalise_ids(raw))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Strategy 2: JSON inside a markdown code fence.
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            raw = _coordinate_list_from_json_obj(obj)
            if raw is not None:
                return _success(_normalise_ids(raw))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 3: any shallow JSON object containing either supported key.
    json_match = re.search(
        r"\{[^{}]*(?:\"cells_entered\"|\"tiles_entered\")[^{}]*\}",
        text,
        re.DOTALL,
    )
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            raw = _coordinate_list_from_json_obj(obj)
            if raw is not None:
                return _success(_normalise_ids(raw))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 4: extract coordinate IDs by pattern from arbitrary text.
    ids = _extract_tile_ids_from_text(text)
    if ids:
        return _success(ids)

    return _failure("No coordinate IDs found in response")


def _sort_key(tid: str) -> tuple[int, int]:
    """Sort key for '(x,y)' IDs: row first, then column."""
    m = re.match(r"\((\d+),(\d+)\)", tid)
    if m:
        return (int(m.group(2)), int(m.group(1)))
    return (0, 0)


def _success(ids: list[str]) -> dict:
    sorted_ids = sorted(set(ids), key=_sort_key)
    return {
        "ok": True,
        # Legacy names used by the existing metrics/runner code.
        "predicted_tile_sequence": ids,
        "predicted_tile_set": sorted_ids,
        # Preferred names for the grid-level version of Case Study 3.
        "predicted_cell_sequence": ids,
        "predicted_cell_set": sorted_ids,
        "parsing_error": None,
    }


def _failure(message: str) -> dict:
    return {
        "ok": False,
        "predicted_tile_sequence": [],
        "predicted_tile_set": [],
        "predicted_cell_sequence": [],
        "predicted_cell_set": [],
        "parsing_error": message,
    }
