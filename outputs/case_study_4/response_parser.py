"""case_study_4/response_parser.py — Prompt builder and VLM response parser.

Prompt design
-------------
The model receives two images (reference pattern + search region) and is
instructed to output strict JSON.  The prompt explicitly:
  * allows "found: false" (NONE is a valid answer)
  * strongly discourages guessing
  * does NOT expose GT tile or bbox to the model

Response parser
---------------
Tries strategies in order:
  1. Direct json.loads of the full response
  2. Fenced markdown code block (```json … ```)
  3. Regex-extracted JSON object {…} from free text
  4. Natural-language extraction (found/tile/confidence keywords)
  5. Returns parsing_error if all fail

Tile normalisation
------------------
  "B3"         → "B3"         (letter-number kept as-is)
  "(2,3)"      → "(2,3)"      (grid coordinate kept as-is)
  "row 2 col 3"→ "(2,3)"      (text converted to grid coordinate)
  "tile_2_3"   → "(2,3)"      (underscore format converted)
  Comparisons are case-insensitive and ignore surrounding whitespace.
"""
from __future__ import annotations

import json
import re


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise visual pattern-matching assistant for SEM "
    "(scanning electron microscope) images.\n"
    "Your task is to determine whether a reference pattern appears in a "
    "search region and, if present, localise it."
)

_SHARED_INSTRUCTION = """\
You will be shown two images:
  1. REFERENCE PATTERN — the specific object or pattern you are looking for.
  2. SEARCH REGION — a larger SEM image in which you must look for the reference.

RULES (read carefully):
- You MUST return a JSON object in the exact schema shown below.
- You are ALLOWED and ENCOURAGED to return "found": false if the pattern is absent.
- Do NOT guess. If you cannot clearly identify the same pattern, return "found": false.
- If your confidence is below 0.5, you MUST return "found": false.
- Hallucinating a tile or location when the pattern is not present is a CRITICAL ERROR.

Return ONLY valid JSON — no explanation text before or after the JSON:
{{
  "found": true,
  "tile": "<tile ID or region label where the pattern is found>",
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.0,
  "reason": "<brief explanation>"
}}

Or, if the pattern is absent:
{{
  "found": false,
  "tile": null,
  "bbox": null,
  "confidence": 0.0,
  "reason": "<why you do not see it>"
}}

Field definitions:
  found      — true if the reference pattern is clearly visible in the search region
  tile       — tile or region label where the pattern is found (null if not found)
  bbox       — approximate bounding box [x1, y1, x2, y2] in the search image in pixels
               (null if not found or cannot be estimated)
  confidence — your confidence that "found" is correct (0.0 = not confident, 1.0 = certain)
  reason     — one sentence explaining your decision\
"""

_SUFFIX_BY_MODE: dict[str, str] = {
    "atlas_global_search": (
        "\n\nContext: The search region is a full SEM atlas image composed of "
        "multiple tiles arranged in a grid. If the pattern is found, identify "
        "the tile using whatever labeling is visible (e.g. 'A1', '(0,2)'), or "
        "describe its position if no labels are present."
    ),
    "grid_scan_search": (
        "\n\nContext: The search region is a grid of SEM image tiles. "
        "If the pattern is found, identify the grid cell by its row-column "
        "coordinate label (e.g. '(0,0)' for top-left, '(1,0)' for the cell to its right)."
    ),
}


def build_prompt(search_mode: str) -> str:
    """Return the full instruction prompt for *search_mode*."""
    base   = _SHARED_INSTRUCTION
    suffix = _SUFFIX_BY_MODE.get(search_mode, "")
    return base + suffix


def get_system_prompt() -> str:
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tile ID normalisation
# ---------------------------------------------------------------------------

# (row,col) or (x,y) format
_GRID_COORD_RE = re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)')

# "row N col M" / "row N, column M" / "R N C M" variants
_ROW_COL_RE = re.compile(
    r'(?:row|r)\s*(\d+)[,\s]+(?:col(?:umn)?|c)\s*(\d+)',
    re.IGNORECASE,
)

# "tile_N_M" or "tile-N-M"
_TILE_UNDERSCORE_RE = re.compile(r'tile[_\-](\d+)[_\-](\d+)', re.IGNORECASE)


def normalise_tile_id(raw: str | None) -> str | None:
    """Normalise a raw tile ID string to a consistent representation.

    Conversions applied (in order):
      "(x,y)"        → "(x,y)"      (grid coordinate — canonical)
      "row N col M"  → "(N,M)"      (text → coordinate)
      "tile_N_M"     → "(N,M)"      (underscore → coordinate)
      "B3", "A1"     → "B3", "A1"   (letter-number — kept as-is, upper-cased)
    Returns None if *raw* is None or empty.
    """
    if not raw:
        return None
    raw = str(raw).strip()

    m = _GRID_COORD_RE.fullmatch(raw)
    if m:
        return f"({m.group(1)},{m.group(2)})"

    m = _ROW_COL_RE.fullmatch(raw)
    if m:
        return f"({m.group(1)},{m.group(2)})"

    m = _TILE_UNDERSCORE_RE.fullmatch(raw)
    if m:
        return f"({m.group(1)},{m.group(2)})"

    # letter-number like "B3" → upper-case, keep as-is
    if re.fullmatch(r'[A-Za-z]+\d+', raw):
        return raw.upper()

    # partial grid coordinate anywhere in the string
    m = _GRID_COORD_RE.search(raw)
    if m:
        return f"({m.group(1)},{m.group(2)})"

    return raw.strip()


def tiles_match(pred: str | None, gt: str | None) -> bool:
    """Return True if normalised tile IDs are equal (case-insensitive)."""
    if pred is None or gt is None:
        return False
    return (normalise_tile_id(pred) or "").upper() == (normalise_tile_id(gt) or "").upper()


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_REQUIRED_RESPONSE_KEYS = {"found"}

_JSON_BLOCK_RE = re.compile(
    r'```(?:json)?\s*(\{.*?\})\s*```',
    re.DOTALL | re.IGNORECASE,
)

# Grab any top-level {...} object (greedy-last if multiple)
_JSON_OBJECT_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)


def _try_parse_json(text: str) -> dict | None:
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "found" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _try_parse_markdown_block(text: str) -> dict | None:
    for m in _JSON_BLOCK_RE.finditer(text):
        obj = _try_parse_json(m.group(1))
        if obj is not None:
            return obj
    return None


def _try_parse_inline_json(text: str) -> dict | None:
    """Try to extract the last {...} object from free text."""
    candidates = list(_JSON_OBJECT_RE.finditer(text))
    for match in reversed(candidates):
        obj = _try_parse_json(match.group())
        if obj is not None:
            return obj
    return None


def _try_parse_natural_language(text: str) -> dict | None:
    """Last-resort: extract found/tile/confidence from natural language."""
    lower = text.lower()

    # found?
    if re.search(r'\bnot\s+found\b|\bnone\b|found.*?false|absent', lower):
        found = False
    elif re.search(r'\bfound\b|located|detected|present', lower):
        found = True
    else:
        return None

    # tile
    tile: str | None = None
    m = _GRID_COORD_RE.search(text)
    if m:
        tile = f"({m.group(1)},{m.group(2)})"
    else:
        m2 = re.search(r'\b([A-Z]\d+)\b', text)
        if m2:
            tile = m2.group(1)

    # confidence
    confidence: float = 0.5
    m3 = re.search(r'confidence[:\s]+([0-9.]+)', lower)
    if m3:
        try:
            confidence = float(m3.group(1))
        except ValueError:
            pass

    # bbox
    bbox = None
    m4 = re.search(r'\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]', text)
    if m4:
        bbox = [int(m4.group(i)) for i in range(1, 5)]

    return {
        "found":      found,
        "tile":       tile if found else None,
        "bbox":       bbox if found else None,
        "confidence": confidence,
        "reason":     "(extracted from natural language — no structured JSON found)",
    }


def _coerce_response(raw: dict) -> dict:
    """Coerce and normalise the parsed JSON into the canonical response shape."""
    found = bool(raw.get("found", False))
    tile  = raw.get("tile")
    bbox  = raw.get("bbox")
    conf  = raw.get("confidence", 0.0)
    reason = raw.get("reason", "")

    # If found==False, force tile/bbox to None
    if not found:
        tile = None
        bbox = None

    # Normalise tile
    tile = normalise_tile_id(tile) if tile else None

    # Coerce bbox to list of 4 numbers
    if bbox is not None:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                bbox = [float(v) for v in bbox]
            except (TypeError, ValueError):
                bbox = None
        else:
            bbox = None

    # Clamp confidence
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.0

    return {
        "found":      found,
        "tile":       tile,
        "bbox":       bbox,
        "confidence": conf,
        "reason":     str(reason),
    }


def parse_vlm_response(text: str) -> dict:
    """Parse a raw VLM text response into a structured prediction dict.

    Tries strategies in order:
      1. Direct json.loads
      2. Fenced markdown ```json … ``` block
      3. Inline {…} JSON object extracted from free text
      4. Natural-language keyword extraction

    Returns
    -------
    dict with keys:
        ok            : bool
        found         : bool
        tile          : str | None
        bbox          : list[float] | None
        confidence    : float
        reason        : str
        parsing_error : str | None
    """
    if not text or not text.strip():
        return {
            "ok": False, "found": False, "tile": None, "bbox": None,
            "confidence": 0.0, "reason": "", "parsing_error": "Empty response",
        }

    raw: dict | None = None

    raw = _try_parse_json(text)
    if raw is None:
        raw = _try_parse_markdown_block(text)
    if raw is None:
        raw = _try_parse_inline_json(text)
    if raw is None:
        raw = _try_parse_natural_language(text)

    if raw is None:
        return {
            "ok": False, "found": False, "tile": None, "bbox": None,
            "confidence": 0.0, "reason": text[:200],
            "parsing_error": "Could not parse any structured response",
        }

    coerced = _coerce_response(raw)
    return {
        "ok":           True,
        "parsing_error": None,
        **coerced,
    }
