"""run_case_study_4.py — Case Study 4: Visual Object/Pattern Retrieval in SEM Atlas Regions

Tests whether VLMs can find a reference object/pattern in a larger SEM atlas or grid
region, and whether the model can correctly abstain (return found=false) when the
pattern is absent.

Four experimental conditions (determined by GT entry fields)
------------------------------------------------------------
  E1P  atlas_global_search + target present  — atlas overview, pattern exists
  E1N  atlas_global_search + target absent   — atlas overview, pattern does NOT exist
  E2P  grid_scan_search    + target present  — tile grid, pattern exists
  E2N  grid_scan_search    + target absent   — tile grid, pattern does NOT exist

Scientific questions
--------------------
Can a VLM locate a reference pattern/object in a larger SEM search region?
Can it correctly abstain (return found=false) when the target is absent?

Prerequisites
-------------
1. GT store: data/case_study_4/ground_truth/pattern_search_gt.json
2. Pattern images and search images listed in GT entries must exist on disk.
3. .env with OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, MODEL_REASONING_EFFORT

Usage
-----
    # List GT entries and check image paths
    python run_case_study_4.py --list

    # Dry-run: build prompts without calling the VLM API
    python run_case_study_4.py --dry-run

    # Run all GT entries
    python run_case_study_4.py --run

    # Run a single sample
    python run_case_study_4.py --run --sample pattern_search_001

    # Smoke test: fake GT + mock VLM responses (no API key required)
    python run_case_study_4.py --smoke

    # Re-aggregate existing runs
    python run_case_study_4.py --aggregate

    # Custom GT store path
    python run_case_study_4.py --run --gt-path /path/to/my_gt.json
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT / "outputs"))

from case_study_4.ground_truth    import (
    load_gt_store, filter_gt_entries, create_fake_gt_store,
    check_image_paths, validate_gt_entry,
)
from case_study_4.response_parser import (
    build_prompt, get_system_prompt, parse_vlm_response,
)
from case_study_4.metrics         import compute_prediction_metrics, aggregate_sample_metrics
from case_study_4.aggregate       import aggregate_runs, save_aggregate_outputs


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_GT_PATH = _PROJECT_ROOT / "data" / "case_study_4" / "ground_truth" / "pattern_search_gt.json"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "outputs" / "case_study_4"
_SEM_SERVICE_URL = "http://localhost:3000"   # sem-service base URL for tile fetching

# CLI search-mode → internal search_mode string
_SEARCH_MODE_MAP = {
    "atlas": "atlas_global_search",
    "grid":  "grid_scan_search",
}


def _parse_region_label(label: str | None) -> int | None:
    """Parse a region label like 'Region000' or 'Region001' → int, or None on failure."""
    if not label:
        return None
    import re as _re
    m = _re.search(r'(\d+)', label)
    return int(m.group(1)) if m else None


def _rel(p: Path) -> str:
    """Return *p* relative to the project root when possible, else absolute."""
    try:
        return str(p.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# OpenAI config (same pattern as Case Study 3)
# ---------------------------------------------------------------------------

def _load_openai_config() -> tuple[str, str, str, str]:
    """Return (api_key, base_url, model_name, reasoning_effort) from .env / environment."""
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key          = os.environ.get("OPENAI_API_KEY", "")
    base_url         = os.environ.get("OPENAI_BASE_URL", "")
    model            = os.environ.get("MODEL_NAME", "gpt-4o")
    reasoning_effort = os.environ.get("MODEL_REASONING_EFFORT", "medium")
    return api_key, base_url, model, reasoning_effort


# ---------------------------------------------------------------------------
# VLM call — two-image variant (reference pattern + search region)
# ---------------------------------------------------------------------------

def _call_vlm(
    pattern_path:     Path,
    search_path:      Path,
    system_prompt:    str,
    user_prompt:      str,
    api_key:          str,
    base_url:         str,
    model:            str,
    reasoning_effort: str = "medium",
) -> dict:
    """Call the OpenAI vision API with two images and return the raw response.

    The user message contains:
      label text → pattern image → label text → search image → instruction

    Returns dict with keys: ok, reply, model, usage, messages, error
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {"ok": False, "reply": None, "messages": [], "error": "openai package not installed"}

    try:
        pattern_b64 = base64.b64encode(pattern_path.read_bytes()).decode()
        search_b64  = base64.b64encode(search_path.read_bytes()).decode()
    except Exception as exc:
        return {"ok": False, "reply": None, "messages": [], "error": f"Could not read image: {exc}"}

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text",      "text": "Reference pattern (the object/pattern to find):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pattern_b64}", "detail": "high"}},
                {"type": "text",      "text": "Search region (look for the reference pattern in this image):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{search_b64}",  "detail": "high"}},
                {"type": "text",      "text": user_prompt},
            ],
        },
    ]

    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(api_key=api_key, **kwargs)

    create_kwargs: dict = {"model": model, "messages": messages}
    if reasoning_effort and reasoning_effort != "none" and "gpt-4" not in model:
        create_kwargs["reasoning_effort"] = reasoning_effort

    try:
        resp = client.chat.completions.create(**create_kwargs)
    except Exception as exc:
        return {"ok": False, "reply": None, "messages": messages, "error": f"OpenAI call failed: {exc}"}

    reply = resp.choices[0].message.content or ""
    usage = {}
    if resp.usage:
        usage = {
            "prompt_tokens":     resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens":      resp.usage.total_tokens,
        }
    return {
        "ok":       True,
        "reply":    reply,
        "model":    model,
        "usage":    usage,
        "messages": messages,
        "error":    None,
    }


# ---------------------------------------------------------------------------
# Mock VLM (for --smoke / --mock-vlm)
# ---------------------------------------------------------------------------

def _mock_vlm_response(gt_entry: dict) -> dict:
    """Return a plausible mock VLM response without calling the API.

    For positive entries the mock claims found=false (conservative — forces
    a false-negative rather than guessing).  For negative entries it correctly
    abstains.
    """
    if gt_entry.get("target_present"):
        resp_text = json.dumps({
            "found":      False,
            "tile":       None,
            "bbox":       None,
            "confidence": 0.3,
            "reason":     "Mock response: pattern not confidently located.",
        })
    else:
        resp_text = json.dumps({
            "found":      False,
            "tile":       None,
            "bbox":       None,
            "confidence": 0.85,
            "reason":     "Mock response: pattern absent.",
        })
    return {
        "ok":       True,
        "reply":    resp_text,
        "model":    "mock",
        "usage":    {},
        "messages": [],
        "error":    None,
    }


# ---------------------------------------------------------------------------
# Grid-scan runner — iterates every tile in a region, calls VLM per tile
# ---------------------------------------------------------------------------

def _fetch_tile_image_bytes(region: int, fw: int, x: int, y: int) -> bytes | None:
    """Download a tile image from sem-service. Returns None on failure."""
    url = f"{_SEM_SERVICE_URL}/api/camera/tile-image?region={region}&fw={fw}&x={x}&y={y}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read()
    except Exception as exc:
        print(f"  [warn] Could not fetch tile ({x},{y}): {exc}", file=sys.stderr)
        return None


def _fetch_tile_list(region: int, fw: int) -> list[dict]:
    """Return [{x, y}, …] for the given region+fw from sem-service."""
    url = f"{_SEM_SERVICE_URL}/api/camera/tiles?region={region}&fw={fw}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("tiles", [])
    except Exception as exc:
        raise RuntimeError(f"Could not fetch tile list from sem-service: {exc}") from exc


def run_sample_grid(
    gt_entry:         dict,
    *,
    region:           int | None = None,
    fw:               int | None = None,
    out_dir:          Path,
    api_key:          str,
    base_url:         str,
    model:            str,
    reasoning_effort: str  = "medium",
    dry_run:          bool = False,
    mock_vlm:         bool = False,
) -> dict:
    """Grid-scan variant: fetch every tile from sem-service and call VLM for each.

    Region resolution priority (first wins):
      1. CLI --region / --fw override
      2. gt_entry['search_region_id'] + gt_entry['search_region_fw']  (for negative entries)
      3. gt_entry['source_region'] label parsed to int  (for positive entries)

    Positive entry (target_present=True):
      Searches the source region. Correct if found in any tile in gt_tiles.
    Negative entry (target_present=False):
      Searches a DIFFERENT region (search_region_id). Correct if NOT found anywhere.
    """
    ts_start  = datetime.now(timezone.utc)
    ts_str    = ts_start.strftime("%Y%m%d_%H%M%S")
    sample_id = gt_entry["sample_id"]
    search_mode = "grid_scan_search"
    gt_present  = gt_entry["target_present"]
    gt_tiles    = gt_entry.get("gt_tiles") or ([gt_entry["gt_tile"]] if gt_entry.get("gt_tile") else [])

    # --- Resolve region ---
    # source_region is always the region to scan (positive: where pattern lives;
    # negative: a different region where pattern is absent).
    # CLI --region overrides.
    if region is None:
        region = _parse_region_label(gt_entry.get("source_region"))
        if region is None:
            raise ValueError(
                f"Cannot determine region for sample '{sample_id}'. "
                f"Set --region on the CLI, or add 'source_region' to the GT entry (e.g. 'Region000')."
            )

    if fw is None:
        fw = gt_entry.get("source_region_fw")

    # Auto-resolve fw from sem-service if still unknown
    if fw is None:
        try:
            url = f"{_SEM_SERVICE_URL}/api/camera/regions"
            with urllib.request.urlopen(url, timeout=10) as resp:
                regions_data = json.loads(resp.read())
            matches = sorted(r["fw"] for r in regions_data.get("regions", []) if r["region"] == region)
            fw = matches[0] if matches else 0
        except Exception:
            fw = 0

    run_id  = f"case4_{sample_id}_grid_r{region}fw{fw}_{ts_str}"
    run_dir = out_dir / "runs" / run_id

    print(f"\n{'='*70}")
    print(f"  Case Study 4 — Grid Scan")
    print(f"  Run ID      : {run_id}")
    print(f"  Sample      : {sample_id}")
    print(f"  Region      : {region}  fw={fw}")
    print(f"  GT present  : {gt_present}   gt_tiles={gt_tiles}")
    print(f"  Model       : {model}")
    print(f"{'='*70}")

    if dry_run:
        tiles = _fetch_tile_list(region, fw)
        print(f"  [dry-run] {len(tiles)} tiles to scan in Region{region} fw={fw}")
        return {"run_id": run_id, "case_study": "case_study_4", "sample_id": sample_id, "status": "dry_run"}

    # Setup dirs
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    (run_dir / "prompts").mkdir(exist_ok=True)
    (run_dir / "full_trace").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "summary").mkdir(exist_ok=True)
    (run_dir / "inputs" / "tiles").mkdir(exist_ok=True)

    # Copy pattern
    pattern_src = Path(gt_entry["target_pattern_image"])
    if not pattern_src.is_absolute():
        pattern_src = _PROJECT_ROOT / pattern_src
    _safe_copy(pattern_src, run_dir / "inputs" / "target_pattern.png")
    (run_dir / "inputs" / "ground_truth_entry.json").write_text(
        json.dumps(gt_entry, indent=2), encoding="utf-8"
    )

    # Fetch tile list
    tiles = _fetch_tile_list(region, fw)
    print(f"  → {len(tiles)} tiles to scan")

    user_prompt   = build_prompt(search_mode)
    system_prompt = get_system_prompt()
    (run_dir / "prompts" / "grid_scan_prompt.md").write_text(
        f"# Prompt — {search_mode}\n\n## System\n\n{system_prompt}\n\n## User\n\n{user_prompt}\n",
        encoding="utf-8",
    )

    per_tile_results: list[dict] = []
    trace_messages:   list[dict] = []
    found_tiles:      list[str]  = []

    for tile_info in tiles:
        tx, ty = tile_info["x"], tile_info["y"]
        tile_label = f"({tx},{ty})"
        print(f"  → tile {tile_label} ", end="", flush=True)

        tile_bytes = _fetch_tile_image_bytes(region, fw, tx, ty)
        if tile_bytes is None:
            print("SKIP (fetch failed)")
            per_tile_results.append({"tile": tile_label, "status": "fetch_failed"})
            continue

        tile_path = run_dir / "inputs" / "tiles" / f"tile_{ty}_{tx}.png"
        tile_path.write_bytes(tile_bytes)

        if mock_vlm:
            # Mock: only claim found if this tile is in gt_tiles
            is_gt_tile = tile_label in gt_tiles
            vlm_result = {
                "ok": True, "model": "mock", "usage": {}, "messages": [],
                "reply": json.dumps({
                    "found": is_gt_tile, "tile": tile_label if is_gt_tile else None,
                    "bbox": None, "confidence": 0.9 if is_gt_tile else 0.1,
                    "reason": "mock",
                }),
            }
        else:
            vlm_result = _call_vlm(
                pattern_path     = run_dir / "inputs" / "target_pattern.png",
                search_path      = tile_path,
                system_prompt    = system_prompt,
                user_prompt      = user_prompt,
                api_key          = api_key,
                base_url         = base_url,
                model            = model,
                reasoning_effort = reasoning_effort,
            )

        parsed = parse_vlm_response(vlm_result.get("reply") or "") if vlm_result["ok"] else {
            "ok": False, "found": False, "tile": None, "bbox": None,
            "confidence": 0.0, "reason": "", "parsing_error": vlm_result.get("error"),
        }

        if parsed.get("found"):
            found_tiles.append(tile_label)
            print(f"FOUND (conf={parsed.get('confidence',0):.2f})")
        else:
            print(f"not found")

        trace_messages.extend(vlm_result.get("messages", []))
        per_tile_results.append({
            "tile": tile_label, "found": parsed.get("found"), "confidence": parsed.get("confidence"),
            "reason": parsed.get("reason"), "parsing_error": parsed.get("parsing_error"),
        })

    _write_jsonl(run_dir / "full_trace" / "model_messages.jsonl", trace_messages)
    _write_jsonl(run_dir / "full_trace" / "tool_calls.jsonl", [])

    # Aggregate result
    overall_found = len(found_tiles) > 0
    correct_found_status = (overall_found == gt_present)
    # Normalise tile labels for comparison: strip spaces, lowercase
    def _norm(t): return t.strip().lower().replace(' ', '')
    correct_tile = None
    if gt_present and overall_found and gt_tiles:
        gt_normalised = {_norm(t) for t in gt_tiles}
        correct_tile = any(_norm(ft) in gt_normalised for ft in found_tiles)

    print(f"  → overall found={overall_found}  found_tiles={found_tiles}")
    print(f"  → correct_found_status={correct_found_status}  correct_tile={correct_tile}")

    pred_result = {
        "sample_id":            sample_id,
        "method_name":          search_mode,
        "status":               "completed",
        "gt_target_present":    gt_present,
        "gt_tiles":             gt_tiles,
        "found":                overall_found,
        "found_tiles":          found_tiles,
        "tiles_scanned":        len(tiles),
        "confidence":           max((r.get("confidence") or 0) for r in per_tile_results) if per_tile_results else 0,
        "correct_found_status": correct_found_status,
        "correct_tile":         correct_tile,
        "per_tile":             per_tile_results,
    }

    (run_dir / "predictions" / "grid_scan.json").write_text(json.dumps(pred_result, indent=2), encoding="utf-8")
    metrics_path_json = run_dir / "metrics" / "per_method_results.json"
    metrics_path_csv  = run_dir / "metrics" / "per_method_results.csv"
    metrics_path_json.write_text(json.dumps({"run_id": run_id, "sample_id": sample_id, "results": [pred_result]}, indent=2), encoding="utf-8")
    _write_metrics_csv(metrics_path_csv, [pred_result])

    manifest = {
        "run_id":               run_id,
        "case_study":           "case_study_4",
        "sample_id":            sample_id,
        "started_at":           ts_start.isoformat(),
        "completed_at":         datetime.now(timezone.utc).isoformat(),
        "status":               "completed",
        "model":                vlm_result.get("model", model) if per_tile_results else model,
        "search_mode":          search_mode,
        "region":               region,
        "fw":                   fw,
        "tiles_scanned":        len(tiles),
        "gt_target_present":    gt_present,
        "gt_tiles":             gt_tiles,
        "method_list":          [search_mode],
        "prediction_status":    "completed",
        "metrics_summary": {
            "found":                overall_found,
            "found_tiles":          found_tiles,
            "correct_found_status": correct_found_status,
            "correct_tile":         correct_tile,
        },
        "warnings": [],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  Run saved → {run_dir}")
    return manifest




def _safe_copy(src: Path, dst: Path) -> None:
    """Copy *src* to *dst*, creating parent dirs.  Non-fatal on failure."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as exc:
        print(f"  [warn] Could not copy {src} → {dst}: {exc}", file=sys.stderr)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _write_run_summary(run_dir: Path, manifest: dict) -> None:
    """Write a human-readable Markdown summary of the run."""
    ms     = manifest.get("metrics_summary", {})
    pred   = manifest.get("prediction_status", "?")
    method = manifest.get("search_mode", "?")
    cfs    = ms.get("correct_found_status", "?")

    lines = [
        f"# Run Summary — {manifest['run_id']}",
        "",
        f"**Case Study**: {manifest['case_study']}  ",
        f"**Sample ID**: `{manifest['sample_id']}`  ",
        f"**Search mode**: `{method}`  ",
        f"**Model**: `{manifest.get('model', '?')}`  ",
        f"**Status**: `{manifest.get('status', '?')}`  ",
        f"**Started**: {manifest.get('started_at', '?')}  ",
        f"**Completed**: {manifest.get('completed_at', '?')}  ",
        "",
        "## Ground Truth",
        "",
        f"- Target present: `{manifest.get('gt_target_present', '?')}`  ",
        f"- GT tile: `{manifest.get('gt_tile', 'null')}`  ",
        f"- GT bbox: `{manifest.get('gt_bbox', 'null')}`  ",
        "",
        "## Prediction",
        "",
        f"- Prediction status: `{pred}`  ",
        f"- Found: `{ms.get('found', '?')}`  ",
        f"- Predicted tile: `{ms.get('predicted_tile', 'null')}`  ",
        f"- Confidence: `{ms.get('confidence', '?')}`  ",
        "",
        "## Metrics",
        "",
        f"- correct_found_status: `{cfs}`  ",
    ]

    gt_present = manifest.get("gt_target_present")
    if gt_present:
        lines += [
            f"- correct_tile: `{ms.get('correct_tile', '?')}`  ",
            f"- bbox_iou: `{ms.get('bbox_iou', 'n/a')}`  ",
            f"- localization_error_px: `{ms.get('localization_error_px', 'n/a')}`  ",
            f"- within_tolerance: `{ms.get('within_tolerance', 'n/a')}`  ",
        ]
    else:
        lines += [
            f"- correct_abstention: `{ms.get('correct_abstention', '?')}`  ",
            f"- false_positive: `{ms.get('false_positive', '?')}`  ",
        ]

    path = run_dir / "summary" / "run_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Single-sample runner
# ---------------------------------------------------------------------------

def run_sample(
    gt_entry:         dict,
    *,
    out_dir:          Path,
    api_key:          str,
    base_url:         str,
    model:            str,
    reasoning_effort: str  = "medium",
    dry_run:          bool = False,
    mock_vlm:         bool = False,
) -> dict:
    """Run the pattern-retrieval task for one GT entry.  Returns the run manifest.

    Saves all artifacts under out_dir/runs/<run_id>/
    """
    ts_start  = datetime.now(timezone.utc)
    ts_str    = ts_start.strftime("%Y%m%d_%H%M%S")
    sample_id = gt_entry["sample_id"]
    run_id    = f"case4_{sample_id}_{ts_str}"
    run_dir   = out_dir / "runs" / run_id

    search_mode = gt_entry["search_mode"]
    gt_present  = gt_entry["target_present"]

    print(f"\n{'='*70}")
    print(f"  Case Study 4 — Visual Pattern Retrieval")
    print(f"  Run ID      : {run_id}")
    print(f"  Sample      : {sample_id}")
    print(f"  Search mode : {search_mode}")
    print(f"  GT present  : {gt_present}   gt_tile={gt_entry.get('gt_tile')!r}")
    print(f"  Model       : {model}")
    print(f"{'='*70}")

    user_prompt   = build_prompt(search_mode)
    system_prompt = get_system_prompt()

    if dry_run:
        print(f"\n  [dry-run] prompt preview:")
        print("    " + user_prompt[:300].replace("\n", "\n    ") + "…")
        return {
            "run_id":     run_id,
            "case_study": "case_study_4",
            "sample_id":  sample_id,
            "status":     "dry_run",
        }

    # --- Create run directories ---
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    (run_dir / "prompts").mkdir(exist_ok=True)
    (run_dir / "full_trace").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "summary").mkdir(exist_ok=True)

    # --- Copy inputs ---
    pattern_src = Path(gt_entry["target_pattern_image"])
    search_src  = Path(gt_entry["search_image"])

    if not pattern_src.is_absolute():
        pattern_src = _PROJECT_ROOT / pattern_src
    if not search_src.is_absolute():
        search_src = _PROJECT_ROOT / search_src

    _safe_copy(pattern_src, run_dir / "inputs" / "target_pattern.png")
    _safe_copy(search_src,  run_dir / "inputs" / "search_image.png")

    # Save GT entry (GT location fields are included — they are for record-keeping,
    # the model does NOT receive them in the prompt)
    (run_dir / "inputs" / "ground_truth_entry.json").write_text(
        json.dumps(gt_entry, indent=2), encoding="utf-8"
    )

    # --- Save prompt ---
    prompt_path = run_dir / "prompts" / f"{search_mode}_prompt.md"
    prompt_path.write_text(
        f"# Prompt — {search_mode}\n\n## System\n\n{system_prompt}\n\n## User\n\n{user_prompt}\n",
        encoding="utf-8",
    )

    # --- Call VLM ---
    warnings: list[str] = []

    if mock_vlm:
        vlm_result = _mock_vlm_response(gt_entry)
        print(f"  → [mock-vlm] using hardcoded response")
    elif not pattern_src.exists() or not search_src.exists():
        missing = []
        if not pattern_src.exists():
            missing.append(str(pattern_src))
        if not search_src.exists():
            missing.append(str(search_src))
        warn = f"Image file(s) not found: {missing}"
        warnings.append(warn)
        print(f"  → [warn] {warn}", file=sys.stderr)
        vlm_result = {
            "ok": False, "reply": None, "messages": [], "model": model,
            "usage": {}, "error": warn,
        }
    else:
        vlm_result = _call_vlm(
            pattern_path     = run_dir / "inputs" / "target_pattern.png",
            search_path      = run_dir / "inputs" / "search_image.png",
            system_prompt    = system_prompt,
            user_prompt      = user_prompt,
            api_key          = api_key,
            base_url         = base_url,
            model            = model,
            reasoning_effort = reasoning_effort,
        )
        print(f"  → VLM call ok={vlm_result['ok']}")

    # --- Save full_trace ---
    trace_messages = vlm_result.get("messages", [])
    if vlm_result.get("reply"):
        trace_messages = trace_messages + [
            {"role": "assistant", "content": vlm_result["reply"]}
        ]
    _write_jsonl(run_dir / "full_trace" / "model_messages.jsonl", trace_messages)
    _write_jsonl(run_dir / "full_trace" / "tool_calls.jsonl", [])  # no agent tools

    # --- Parse response ---
    raw_reply = vlm_result.get("reply") or ""
    if not vlm_result["ok"]:
        parsed = {
            "ok": False, "found": False, "tile": None, "bbox": None,
            "confidence": 0.0, "reason": "",
            "parsing_error": vlm_result.get("error", "VLM call failed"),
        }
    else:
        parsed = parse_vlm_response(raw_reply)

    if not parsed["ok"]:
        print(f"  → [warn] parsing failed: {parsed['parsing_error']}")
        if raw_reply:
            print(f"  → [warn] raw reply (last 300 chars): …{raw_reply[-300:]}")

    # --- Compute metrics ---
    pred_result = compute_prediction_metrics(
        parsed       = parsed,
        gt_entry     = gt_entry,
        method_name  = search_mode,
        raw_response = raw_reply,
    )

    status = pred_result.get("status", "completed")
    print(
        f"  → status={status}"
        f"  found={pred_result.get('found')}"
        f"  correct_found_status={pred_result.get('correct_found_status')}"
    )
    if gt_present:
        print(
            f"     correct_tile={pred_result.get('correct_tile')}"
            f"  loc_err={pred_result.get('localization_error_px')}"
        )
    else:
        print(
            f"     correct_abstention={pred_result.get('correct_abstention')}"
            f"  false_positive={pred_result.get('false_positive')}"
        )

    # --- Save prediction JSON ---
    pred_path = run_dir / "predictions" / f"{search_mode}.json"
    pred_path.write_text(json.dumps(pred_result, indent=2), encoding="utf-8")

    # --- Save metrics ---
    metrics_path_json = run_dir / "metrics" / "per_method_results.json"
    metrics_path_csv  = run_dir / "metrics" / "per_method_results.csv"

    metrics_json = {"run_id": run_id, "sample_id": sample_id, "results": [pred_result]}
    metrics_path_json.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    _write_metrics_csv(metrics_path_csv, [pred_result])

    # --- Build manifest ---
    # Compact metrics summary (subset of pred_result, for quick overview)
    ms_keys_present = ["found", "predicted_tile", "confidence", "correct_found_status",
                       "correct_tile", "bbox_iou", "localization_error_px", "within_tolerance"]
    ms_keys_absent  = ["found", "predicted_tile", "confidence", "correct_found_status",
                       "correct_abstention", "false_positive"]
    ms_keys         = ms_keys_present if gt_present else ms_keys_absent

    metrics_summary = {k: pred_result.get(k) for k in ms_keys}

    manifest = {
        "run_id":               run_id,
        "case_study":           "case_study_4",
        "sample_id":            sample_id,
        "started_at":           ts_start.isoformat(),
        "completed_at":         datetime.now(timezone.utc).isoformat(),
        "status":               status,
        "model":                vlm_result.get("model", model),
        "search_mode":          search_mode,
        "gt_target_present":    gt_present,
        "gt_tile":              gt_entry.get("gt_tile"),
        "gt_bbox":              gt_entry.get("gt_bbox"),
        "target_pattern_image": gt_entry["target_pattern_image"],
        "search_image":         gt_entry["search_image"],
        "method_list":          [search_mode],
        "prediction_status":    status,
        "prediction_paths":     {search_mode: _rel(pred_path)},
        "metric_paths": {
            "json": _rel(metrics_path_json),
            "csv":  _rel(metrics_path_csv),
        },
        "prompt_paths":         {search_mode: _rel(prompt_path)},
        "log_paths": {
            "model_messages": _rel(run_dir / "full_trace" / "model_messages.jsonl"),
        },
        "warnings":             warnings,
        "metrics_summary":      metrics_summary,
    }

    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  Run saved → {run_dir}")

    _write_run_summary(run_dir, manifest)
    return manifest


# ---------------------------------------------------------------------------
# CSV writer for metrics
# ---------------------------------------------------------------------------

_METRICS_CSV_FIELDS = [
    "run_id", "sample_id", "method_name", "status", "gt_target_present",
    "gt_tile", "found", "predicted_tile", "confidence",
    "correct_found_status",
    # present-only
    "correct_tile", "bbox_iou", "localization_error_px", "within_tolerance", "correct_detection",
    # absent-only
    "correct_abstention", "false_positive",
    "parsing_error",
]


def _write_metrics_csv(path: Path, results: list[dict], run_id: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_METRICS_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in _METRICS_CSV_FIELDS}
            if run_id and not row.get("run_id"):
                row["run_id"] = run_id
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Aggregate across completed runs
# ---------------------------------------------------------------------------

def _aggregate_and_print(out_dir: Path) -> None:
    runs_dir = out_dir / "runs"
    if not runs_dir.exists() or not list(runs_dir.glob("*/run_manifest.json")):
        print("  No completed runs found to aggregate.", file=sys.stderr)
        return

    agg = aggregate_runs(out_dir)
    save_aggregate_outputs(agg, out_dir)

    print(f"\n  Aggregate summary ({agg['n_runs']} runs):")
    for method, stats in agg.get("by_method", {}).items():
        print(
            f"    {method:30s}"
            f"  n={stats['n_total']}"
            f"  presence_acc={stats.get('presence_accuracy', 'n/a')}"
            f"  pos_recall={stats.get('positive_recall', 'n/a')}"
            f"  neg_abstention={stats.get('negative_abstention_accuracy', 'n/a')}"
        )
    print(f"  Saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _run_smoke_test(out_dir: Path) -> None:
    """Create a fake GT store, run 2 samples with mock VLM, assert artifacts exist."""
    print("\n" + "="*70)
    print("  Case Study 4 — SMOKE TEST")
    print("="*70)

    passed = 0
    failed = 0

    def _assert(condition: bool, label: str) -> None:
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {label}")
            passed += 1
        else:
            print(f"  FAIL  {label}", file=sys.stderr)
            failed += 1

    with tempfile.TemporaryDirectory(prefix="cs4_smoke_") as tmp:
        tmp_dir   = Path(tmp)
        gt_path   = tmp_dir / "gt.json"
        smoke_out = tmp_dir / "outputs"

        # Create tiny 32×32 synthetic PNGs
        pat_img    = tmp_dir / "pattern.png"
        search_img = tmp_dir / "search.png"
        _create_tiny_png(pat_img,    color=(200, 100, 50))
        _create_tiny_png(search_img, color=(50,  100, 200))

        # --- Stage 1: empty store ---
        from case_study_4.ground_truth import create_empty_gt_store, load_gt_store
        empty_path = tmp_dir / "empty.json"
        create_empty_gt_store(empty_path)
        store = load_gt_store(empty_path)
        _assert(store["entries"] == [], "Empty GT store has zero entries")

        # --- Stage 2: fake store ---
        create_fake_gt_store(gt_path, target_pattern_image=str(pat_img),
                             search_image=str(search_img))
        store = load_gt_store(gt_path)
        _assert(len(store["entries"]) == 2, "Fake GT store has 2 entries")
        _assert(store["entries"][0]["target_present"] is True,  "Entry 0 is positive")
        _assert(store["entries"][1]["target_present"] is False, "Entry 1 is negative")

        # --- Stage 3: run both samples ---
        entries = filter_gt_entries(gt_path)
        manifests = []
        for entry in entries:
            m = run_sample(
                gt_entry  = entry,
                out_dir   = smoke_out,
                api_key   = "smoke-test",
                base_url  = "",
                model     = "mock",
                mock_vlm  = True,
            )
            manifests.append(m)

        _assert(len(manifests) == 2, "Both samples produced manifests")

        # --- Stage 4: check run folder structure ---
        for m in manifests:
            run_id  = m["run_id"]
            run_dir = smoke_out / "runs" / run_id
            _assert(run_dir.exists(),                          f"[{run_id}] run_dir exists")
            _assert((run_dir / "run_manifest.json").exists(),  f"[{run_id}] run_manifest.json")
            _assert((run_dir / "inputs" / "target_pattern.png").exists(),
                                                               f"[{run_id}] inputs/target_pattern.png")
            _assert((run_dir / "inputs" / "search_image.png").exists(),
                                                               f"[{run_id}] inputs/search_image.png")
            _assert((run_dir / "inputs" / "ground_truth_entry.json").exists(),
                                                               f"[{run_id}] inputs/ground_truth_entry.json")
            search_mode = m["search_mode"]
            _assert((run_dir / "predictions" / f"{search_mode}.json").exists(),
                                                               f"[{run_id}] predictions/{search_mode}.json")
            _assert((run_dir / "prompts" / f"{search_mode}_prompt.md").exists(),
                                                               f"[{run_id}] prompts/{search_mode}_prompt.md")
            _assert((run_dir / "metrics" / "per_method_results.json").exists(),
                                                               f"[{run_id}] metrics/per_method_results.json")
            _assert((run_dir / "metrics" / "per_method_results.csv").exists(),
                                                               f"[{run_id}] metrics/per_method_results.csv")
            _assert((run_dir / "summary" / "run_summary.md").exists(),
                                                               f"[{run_id}] summary/run_summary.md")

        # --- Stage 5: prediction JSON has required keys ---
        for m in manifests:
            run_id      = m["run_id"]
            search_mode = m["search_mode"]
            pred_path   = smoke_out / "runs" / run_id / "predictions" / f"{search_mode}.json"
            pred        = json.loads(pred_path.read_text())
            for key in ("sample_id", "method_name", "found", "confidence",
                        "gt_target_present", "correct_found_status", "status"):
                _assert(key in pred, f"[{run_id}] prediction has key '{key}'")

        # --- Stage 6: aggregate ---
        agg = aggregate_runs(smoke_out)
        _assert(agg["n_runs"] == 2, "Aggregate sees 2 runs")
        save_aggregate_outputs(agg, smoke_out)
        _assert((smoke_out / "aggregate_summary.json").exists(),     "aggregate_summary.json saved")
        _assert((smoke_out / "per_sample_method_results.csv").exists(), "per_sample_method_results.csv saved")
        _assert((smoke_out / "summary_by_method.csv").exists(),      "summary_by_method.csv saved")

    print(f"\n  Smoke test complete: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


def _create_tiny_png(path: Path, color: tuple[int, int, int] = (128, 128, 128)) -> None:
    """Write a minimal 32×32 RGB PNG without requiring Pillow."""
    import struct, zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    w, h = 32, 32
    r, g, b = color
    raw = b"".join(b"\x00" + bytes([r, g, b] * w) for _ in range(h))
    compressed = zlib.compress(raw)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Case Study 4 — Visual Object/Pattern Retrieval in SEM Atlas Regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run",          action="store_true", help="Run GT entries (all or --sample)")
    p.add_argument("--sample",       help="Restrict to a single sample_id")
    p.add_argument("--search-mode",  choices=["atlas", "grid"], default="atlas",
                   help="atlas: single atlas image lookup; grid: scan all tiles in a region (default: atlas)")
    p.add_argument("--region",       type=int, default=0, help="Region number for grid scan (default: 0)")
    p.add_argument("--fw",           type=int, default=None, help="Field-width for grid scan (default: first available)")
    p.add_argument("--dry-run",      action="store_true", help="Preview prompts without VLM calls")
    p.add_argument("--smoke",        action="store_true", help="Smoke test with fake GT + mock VLM")
    p.add_argument("--list",         action="store_true", help="List GT entries and check image paths")
    p.add_argument("--aggregate",    action="store_true", help="Re-aggregate existing runs")
    p.add_argument("--mock-vlm",     action="store_true", help="Use mock VLM response (no API call)")
    p.add_argument("--gt-path",      default=str(_DEFAULT_GT_PATH), help="GT store JSON path")
    p.add_argument("--out-dir",      default=str(_DEFAULT_OUT_DIR),  help="Output root directory")
    p.add_argument("--delay",        type=int, default=1, help="Seconds between samples (default: 1)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args    = _parse_args(argv)
    out_dir = Path(args.out_dir)
    gt_path = Path(args.gt_path)

    # ------------------------------------------------------------------
    # --smoke
    # ------------------------------------------------------------------
    if args.smoke:
        _run_smoke_test(out_dir)
        return

    # ------------------------------------------------------------------
    # --list
    # ------------------------------------------------------------------
    if args.list:
        print("\nCase Study 4 — Visual Pattern Retrieval")
        print(f"  GT store : {gt_path}")
        print(f"  output   : {out_dir}")
        if not gt_path.exists():
            print("  [warn] GT store not found.", file=sys.stderr)
            return
        try:
            entries = filter_gt_entries(gt_path)
        except Exception as exc:
            print(f"  [error] Could not load GT store: {exc}", file=sys.stderr)
            return
        print(f"\nGT entries ({len(entries)}):")
        for e in entries:
            flag = "P" if e["target_present"] else "N"
            print(f"  [{flag}] {e['sample_id']:40s}  mode={e.get('search_mode', 'n/a'):25s}  tile={e.get('gt_tile') or e.get('gt_tiles')}")
        warnings = check_image_paths(gt_path)
        if warnings:
            print(f"\nImage path warnings ({len(warnings)}):")
            for w in warnings:
                print(f"  [warn] {w}")
        else:
            print("\nAll image paths OK.")
        return

    # ------------------------------------------------------------------
    # --aggregate
    # ------------------------------------------------------------------
    if args.aggregate:
        print(f"\nCase Study 4 — Aggregating runs in {out_dir}")
        _aggregate_and_print(out_dir)
        return

    # ------------------------------------------------------------------
    # --run / --dry-run
    # ------------------------------------------------------------------
    if not (args.run or args.dry_run or args.sample):
        # Default: treat as --run
        pass

    print(f"\nCase Study 4 — Visual Pattern Retrieval")
    print(f"  GT store : {gt_path}")
    print(f"  output   : {out_dir}")

    if not gt_path.exists():
        print(f"ERROR: GT store not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    try:
        entries = filter_gt_entries(
            gt_path,
            sample_ids=[args.sample] if args.sample else None,
        )
    except Exception as exc:
        print(f"ERROR: Could not load GT store: {exc}", file=sys.stderr)
        sys.exit(1)

    if not entries:
        print("No GT entries match the filters. Check --sample or populate the GT store.")
        return

    api_key, base_url, model, reasoning_effort = _load_openai_config()
    if not api_key and not args.dry_run and not args.mock_vlm:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or use --mock-vlm.", file=sys.stderr)
        sys.exit(1)

    search_mode_internal = _SEARCH_MODE_MAP[args.search_mode]
    print(f"  model       : {model}  (reasoning_effort={reasoning_effort})")
    print(f"  search-mode : {args.search_mode} ({search_mode_internal})")
    print(f"  entries     : {len(entries)}")
    if args.search_mode == "grid":
        print(f"  region      : {args.region if args.region is not None else 'auto from GT entry'}  fw={args.fw if args.fw is not None else 'auto'}")
    if args.dry_run:
        print("  [dry-run mode — no VLM calls will be made]")
    if args.mock_vlm:
        print("  [mock-vlm mode — hardcoded responses]")

    manifests: list[dict] = []
    for i, entry in enumerate(entries):
        entry = dict(entry)
        entry["search_mode"] = search_mode_internal

        if args.search_mode == "grid":
            manifest = run_sample_grid(
                gt_entry         = entry,
                region           = args.region,
                fw               = args.fw,
                out_dir          = out_dir,
                api_key          = api_key,
                base_url         = base_url,
                model            = model,
                reasoning_effort = reasoning_effort,
                dry_run          = args.dry_run,
                mock_vlm         = args.mock_vlm,
            )
        else:
            manifest = run_sample(
                gt_entry         = entry,
                out_dir          = out_dir,
                api_key          = api_key,
                base_url         = base_url,
                model            = model,
                reasoning_effort = reasoning_effort,
                dry_run          = args.dry_run,
                mock_vlm         = args.mock_vlm,
            )
        manifests.append(manifest)
        if i < len(entries) - 1:
            time.sleep(args.delay)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {len(manifests)} sample(s) processed")
    for m in manifests:
        ms = m.get("metrics_summary", {})
        print(
            f"  {m['sample_id']:40s}"
            f"  status={m.get('status', '?'):15s}"
            f"  correct_found={ms.get('correct_found_status', '?')}"
        )

    if not args.dry_run and len(manifests) > 0:
        _aggregate_and_print(out_dir)

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
