"""case_study_4/ground_truth.py — Ground Truth store utilities for Case Study 4.

Canonical GT store path: data/case_study_4/ground_truth/pattern_search_gt.json

GT store structure
------------------
{
    "dataset_name": "case_study_4_pattern_search_gt",
    "updated_at": "2026-06-01T12:00:00+02:00",
    "entries": [...]
}

Each entry
----------
{
    "sample_id":                str,          # primary key (required)
    "target_pattern_image":     str,          # path (required)
    "search_image":             str,          # path (required)
    "search_mode":              str,          # "atlas_global_search" | "grid_scan_search" (required)
    "target_present":           bool,         # required
    "gt_tile":                  str | null,   # required if target_present == True
    "gt_bbox":                  list | null,  # [x1, y1, x2, y2] – recommended if present
    "gt_center":                list | null,  # [cx, cy] – derived from bbox when missing
    "acceptable_tolerance_px":  int | null,
    "notes":                    str | null,
}
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SEARCH_MODES = {"atlas_global_search", "grid_scan_search"}

_REQUIRED_FIELDS = {"sample_id", "target_pattern_image", "target_present"}

# search_image may be null when entry was created from the UI before pairing with an atlas
# search_mode may be absent when it will be supplied at run-time via --search-mode CLI flag
_OPTIONAL_FIELDS = {"search_image", "search_mode", "gt_tile", "gt_tiles", "gt_bbox", "gt_center",
                    "acceptable_tolerance_px", "source_region", "notes"}

_CSV_COLUMNS = [
    "sample_id", "target_pattern_image", "search_image", "search_mode",
    "target_present", "gt_tiles", "gt_tile", "gt_bbox", "acceptable_tolerance_px", "source_region", "notes",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_gt_entry(entry: dict) -> None:
    """Validate a single GT entry.  Raises ValueError with a descriptive message on failure."""
    missing = _REQUIRED_FIELDS - set(entry)
    if missing:
        raise ValueError(f"GT entry missing required fields: {missing!r}  (entry={entry.get('sample_id', '?')!r})")

    if not isinstance(entry["sample_id"], str) or not entry["sample_id"].strip():
        raise ValueError("sample_id must be a non-empty string")

    if entry.get("search_mode") is not None and entry["search_mode"] not in VALID_SEARCH_MODES:
        raise ValueError(
            f"search_mode {entry['search_mode']!r} is not valid. "
            f"Expected one of {VALID_SEARCH_MODES}"
        )

    if not isinstance(entry["target_present"], bool):
        raise ValueError(
            f"target_present must be bool, got {type(entry['target_present']).__name__!r} "
            f"for sample_id={entry['sample_id']!r}"
        )

    if entry["target_present"]:
        # Accept gt_tiles (list, new) or gt_tile (string, legacy) — at least one required
        # unless search_image is also null (entry was saved from UI without pairing yet)
        has_tiles = bool(entry.get("gt_tiles")) or bool(entry.get("gt_tile"))
        if not has_tiles and entry.get("search_image") is not None:
            raise ValueError(
                f"gt_tiles (or gt_tile) is required when target_present=True and search_image is set "
                f"(sample_id={entry['sample_id']!r})"
            )
        if entry.get("gt_tiles") is not None:
            tiles = entry["gt_tiles"]
            if not isinstance(tiles, list) or not all(isinstance(t, str) for t in tiles):
                raise ValueError(
                    f"gt_tiles must be a list of strings (sample_id={entry['sample_id']!r})"
                )
    else:
        for field in ("gt_tile", "gt_tiles", "gt_bbox", "gt_center"):
            if entry.get(field) is not None:
                raise ValueError(
                    f"{field} must be null when target_present=False "
                    f"(sample_id={entry['sample_id']!r})"
                )

    if entry.get("gt_bbox") is not None:
        bbox = entry["gt_bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            raise ValueError(
                f"gt_bbox must be a list of 4 numbers [x1, y1, x2, y2] "
                f"(sample_id={entry['sample_id']!r})"
            )

    if entry.get("gt_center") is not None:
        c = entry["gt_center"]
        if not (isinstance(c, list) and len(c) == 2 and all(isinstance(v, (int, float)) for v in c)):
            raise ValueError(
                f"gt_center must be a list of 2 numbers [cx, cy] "
                f"(sample_id={entry['sample_id']!r})"
            )


def validate_gt_store(store: dict) -> None:
    """Validate the top-level GT store dict.  Raises ValueError on failure."""
    if not isinstance(store, dict):
        raise ValueError("GT store must be a JSON object")
    if "entries" not in store:
        raise ValueError("GT store must have an 'entries' key")
    if not isinstance(store["entries"], list):
        raise ValueError("GT store 'entries' must be a list")

    seen_ids: set[str] = set()
    for entry in store["entries"]:
        validate_gt_entry(entry)
        sid = entry["sample_id"]
        if sid in seen_ids:
            raise ValueError(f"Duplicate sample_id {sid!r} in GT store")
        seen_ids.add(sid)


# ---------------------------------------------------------------------------
# GT center derivation
# ---------------------------------------------------------------------------

def _derive_center(entry: dict) -> list[float] | None:
    """Return gt_center for entry, deriving from gt_bbox when gt_center is absent."""
    c = entry.get("gt_center")
    if c is not None:
        return c
    bbox = entry.get("gt_bbox")
    if bbox is not None:
        return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    return None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_gt_store(gt_store_path: str | Path) -> dict:
    """Load and validate the GT store from *gt_store_path*.

    Returns the validated store dict (entries list may be empty).
    Raises FileNotFoundError or ValueError on failure.
    """
    p = Path(gt_store_path)
    if not p.exists():
        raise FileNotFoundError(f"GT store not found: {p}")
    store = json.loads(p.read_text(encoding="utf-8"))
    validate_gt_store(store)
    return store


def save_gt_store(store: dict, gt_store_path: str | Path) -> None:
    """Persist *store* to disk, updating 'updated_at'."""
    validate_gt_store(store)
    store["updated_at"] = datetime.now(timezone.utc).isoformat()
    p = Path(gt_store_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(store, indent=2), encoding="utf-8")


def get_gt_entry(gt_store_path: str | Path, sample_id: str) -> dict | None:
    """Return the GT entry with *sample_id*, or None if not found."""
    store = load_gt_store(gt_store_path)
    for entry in store["entries"]:
        if entry["sample_id"] == sample_id:
            return entry
    return None


def save_or_update_gt_entry(entry: dict, gt_store_path: str | Path) -> None:
    """Upsert *entry* by sample_id into the GT store at *gt_store_path*.

    Creates the store file if it does not exist.
    """
    validate_gt_entry(entry)
    p = Path(gt_store_path)
    if p.exists():
        store = load_gt_store(p)
    else:
        store = _empty_store_dict()

    entries = store["entries"]
    for i, e in enumerate(entries):
        if e["sample_id"] == entry["sample_id"]:
            entries[i] = entry
            save_gt_store(store, p)
            return
    entries.append(entry)
    save_gt_store(store, p)


def filter_gt_entries(
    gt_store_path: str | Path,
    *,
    sample_ids:     list[str] | None = None,
    search_modes:   list[str] | None = None,
    target_present: bool | None = None,
    limit:          int | None = None,
) -> list[dict]:
    """Return GT entries matching the given filters.

    All filters are ANDed; None means 'no filter on this dimension'.
    """
    store   = load_gt_store(gt_store_path)
    entries = store["entries"]

    if sample_ids is not None:
        entries = [e for e in entries if e["sample_id"] in sample_ids]
    if search_modes is not None:
        entries = [e for e in entries if e["search_mode"] in search_modes]
    if target_present is not None:
        entries = [e for e in entries if e["target_present"] == target_present]
    if limit is not None:
        entries = entries[:limit]

    return entries


# ---------------------------------------------------------------------------
# CSV summary export
# ---------------------------------------------------------------------------

def export_gt_summary_csv(gt_store_path: str | Path, csv_path: str | Path) -> None:
    """Export the GT store to a human-readable CSV file.

    Columns: sample_id, target_pattern_image, search_image, search_mode,
             target_present, gt_tile, gt_bbox, acceptable_tolerance_px, notes
    """
    store  = load_gt_store(gt_store_path)
    out    = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for entry in store["entries"]:
            row = {k: entry.get(k, "") for k in _CSV_COLUMNS}
            if row["gt_bbox"] is not None and isinstance(row["gt_bbox"], list):
                row["gt_bbox"] = json.dumps(row["gt_bbox"])
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Store creation
# ---------------------------------------------------------------------------

def _empty_store_dict() -> dict:
    return {
        "dataset_name": "case_study_4_pattern_search_gt",
        "updated_at":   datetime.now(timezone.utc).isoformat(),
        "entries":      [],
    }


def create_empty_gt_store(gt_store_path: str | Path) -> None:
    """Write an empty GT store to *gt_store_path*.

    Raises FileExistsError if the file already exists.
    """
    p = Path(gt_store_path)
    if p.exists():
        raise FileExistsError(f"GT store already exists at {p}. Delete it first or use save_or_update_gt_entry().")
    save_gt_store(_empty_store_dict(), p)


def create_fake_gt_store(
    gt_store_path: str | Path,
    target_pattern_image: str | None = None,
    search_image:         str | None = None,
) -> None:
    """Write a 2-entry fake GT store for smoke testing.

    If *target_pattern_image* and *search_image* are given, those paths are used.
    Otherwise placeholder paths are used (image existence is NOT validated here).

    Entry 001 — positive present (atlas_global_search)
    Entry 002 — negative absent  (grid_scan_search)
    """
    pat_img    = target_pattern_image or "data/case_study_4/patterns/fake_pattern.png"
    search_img = search_image          or "data/case_study_4/search_regions/fake_atlas.png"

    store = _empty_store_dict()
    store["dataset_name"] = "case_study_4_fake_gt"
    store["entries"] = [
        {
            "sample_id":               "fake_001",
            "target_pattern_image":    pat_img,
            "search_image":            search_img,
            "search_mode":             "atlas_global_search",
            "target_present":          True,
            "gt_tile":                 "A1",
            "gt_bbox":                 [10, 10, 30, 30],
            "gt_center":               [20, 20],
            "acceptable_tolerance_px": 20,
            "notes":                   "Fake positive entry for smoke testing",
        },
        {
            "sample_id":               "fake_002",
            "target_pattern_image":    pat_img,
            "search_image":            search_img,
            "search_mode":             "grid_scan_search",
            "target_present":          False,
            "gt_tile":                 None,
            "gt_bbox":                 None,
            "gt_center":               None,
            "acceptable_tolerance_px": None,
            "notes":                   "Fake negative entry for smoke testing",
        },
    ]
    p = Path(gt_store_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(store, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Image path validation (non-fatal helper)
# ---------------------------------------------------------------------------

def check_image_paths(gt_store_path: str | Path) -> list[str]:
    """Return a list of warning strings for GT entries whose image paths do not exist."""
    store    = load_gt_store(gt_store_path)
    warnings: list[str] = []
    for entry in store["entries"]:
        sid = entry["sample_id"]
        for field in ("target_pattern_image", "search_image"):
            val = entry.get(field)
            if not val:
                continue
            path = Path(val)
            if not path.exists():
                warnings.append(f"[{sid}] {field} not found: {path}")
    return warnings
