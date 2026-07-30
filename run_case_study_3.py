"""run_case_study_3.py - Case Study 3: SEM Atlas Path-to-Tile Reasoning

Tests whether VLMs can identify which SEM atlas tiles are traversed by a
drawn path, and whether visual aids (grid overlay, coordinate labels) improve
accuracy.

Nine experimental variants
----------------------------
  L0_path_only          - VLM sees atlas PNG with drawn path, no grid, no labels
  L0_grid               - VLM sees atlas PNG with path + tile grid overlay
  L0_grid_labels        - VLM sees atlas PNG with path + grid + coordinate labels
----------------------------
  L1_path_only          - VLM sees atlas PNG with drawn path, no grid, no labels. Virtual 2x2 subdivision per SEM tile (4x more cells to identify).
  L1_grid               - VLM sees atlas PNG with path + tile grid overlay. Virtual 2x2 subdivision per SEM tile (4x more cells to identify).
  L1_grid_labels        - VLM sees atlas PNG with path + grid + coordinate labels. Virtual 2x2 subdivision per SEM tile (4x more cells to identify).
----------------------------
  L2_path_only          - VLM sees atlas PNG with drawn path, no grid, no labels. Virtual 4x4 subdivision per SEM tile (16x more cells to identify). 
  L2_grid               - VLM sees atlas PNG with path + tile grid overlay. Virtual 4x4 subdivision per SEM tile (16x more cells to identify). 
  L2_grid_labels        - VLM sees atlas PNG with path + grid + coordinate labels. Virtual 4x4 subdivision per SEM tile (16x more cells to identify). 
----------------------------

Scientific question
-------------------
Can a VLM correctly identify which SEM atlas tiles a drawn path traverses?
Do tile grid overlays and coordinate labels improve VLM accuracy?

Prerequisites
-------------
1. sem-service running: docker-compose up -d  (or docker run -p 3000:3000 ...)
2. Enter atlas mode in the UI: load a tile grid, then switch to atlas mode
3. Draw a freehand path across tiles using the pen tool

Usage
-----
    # List variant definitions and check current atlas state
    python run_case_study_3.py --list

    # Run all 9 variants three times in one batch folder
    python run_case_study_3.py --run --n-runs 3

    # Run one variant three times in one batch folder
    python run_case_study_3.py --run --variant L0_grid --n-runs 3

    # Dry-run: preview prompts without calling the VLM API
    python run_case_study_3.py --dry-run

    # Run all 9 variants for the currently drawn path
    python run_case_study_3.py --run

    # Single variant
    python run_case_study_3.py --run --variant L0_path_only

    # Smoke test: L0_path_only only, minimal output
    python run_case_study_3.py --smoke

Service URLs (override via env or flags):
    SEM_SERVICE_URL - default http://localhost:3000
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request as _Req, urlopen
from urllib.error import HTTPError as _HTTPError

_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT / "outputs"))

from case_study_3.gt_extraction import gt_from_objects
from case_study_3.tile_parser    import parse_tile_ids
from case_study_3.metrics        import compute_prediction_metrics, aggregate_variant_metrics


# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------

SEM_URL = os.environ.get("SEM_SERVICE_URL", "http://localhost:3000")


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "L0_path_only": {
        "description":  "L0 original SEM-tile grid; path only, no grid or labels",
        "grid_level":   0,
        "visual_condition": "path_only",
        "overlay":      {"grid": False, "labels": False},
    },
    "L0_grid": {
        "description":  "L0 original SEM-tile grid; path + yellow grid overlay",
        "grid_level":   0,
        "visual_condition": "grid",
        "overlay":      {"grid": True, "labels": False},
    },
    "L0_grid_labels": {
        "description":  "L0 original SEM-tile grid; path + grid + coordinate labels",
        "grid_level":   0,
        "visual_condition": "grid_labels",
        "overlay":      {"grid": True, "labels": True},
    },
    "L1_path_only": {
        "description":  "L1 virtual 2x2 subdivision per SEM tile; path only, no subdivision grid or labels",
        "grid_level":   1,
        "visual_condition": "path_only",
        "overlay":      {"grid": False, "labels": False},
    },
    "L1_grid": {
        "description":  "L1 virtual 2x2 subdivision per SEM tile; path + subdivision grid",
        "grid_level":   1,
        "visual_condition": "grid",
        "overlay":      {"grid": True, "labels": False},
    },
    "L1_grid_labels": {
        "description":  "L1 virtual 2x2 subdivision per SEM tile; path + subdivision grid + coordinate labels",
        "grid_level":   1,
        "visual_condition": "grid_labels",
        "overlay":      {"grid": True, "labels": True},
    },
    "L2_grid": {
        "description":  "L2 virtual 4x4 subdivision per SEM tile; path + subdivision grid",
        "grid_level":   2,
        "visual_condition": "grid",
        "overlay":      {"grid": True, "labels": False},
    },
    "L2_grid_labels": {
        "description":  "L2 virtual 4x4 subdivision per SEM tile; path + subdivision grid + coordinate labels",
        "grid_level":   2,
        "visual_condition": "grid_labels",
        "overlay":      {"grid": True, "labels": True},
    },
    # Optional/exploratory: selectable with --variant L2_path_only, but excluded
    # from DEFAULT_VARIANTS because the 4x4 subdivision is invisible.
    "L2_path_only": {
        "description":  "L2 virtual 4x4 subdivision per SEM tile; path only, no subdivision grid or labels (exploratory)",
        "grid_level":   2,
        "visual_condition": "path_only",
        "overlay":      {"grid": False, "labels": False},
        "exploratory":  True,
    },
}

DEFAULT_VARIANTS: list[str] = [
    "L0_path_only",
    "L0_grid",
    "L0_grid_labels",
    "L1_path_only",
    "L1_grid",
    "L1_grid_labels",
    "L2_path_only",
    "L2_grid",
    "L2_grid_labels",
]

# Extra atlas overlay styling fields consumed by the sem-environment renderer.
# These make grid/label overlays visible after full-atlas PNG export and VLM resizing.
# The runner may override label size per grid level below.
ATLAS_OVERLAY_STYLE: dict[str, float | int] = {
    "gridLineWidth": 20,
    "gridLineAlpha": 0.85,
    "labelFontSize": 64,
    "labelBoxPadding": 14,
    "labelBoxAlpha": 0.70,
}

LABEL_FONT_SIZE_BY_GRID_LEVEL: dict[int, int] = {
    0: 64,
    1: 48,
    2: 34,
}


def variant_grid_level(variant_name: str) -> int:
    return int(VARIANTS[variant_name].get("grid_level", 0))


def variant_subdivision(variant_name: str) -> int:
    return 2 ** variant_grid_level(variant_name)


# ---------------------------------------------------------------------------
# VLM prompt
# ---------------------------------------------------------------------------

_PROMPT_SHARED = (
    "You are analyzing an SEM atlas image. A red path has been drawn on the image.\n"
    "Your task: identify which atlas analysis-grid cells the drawn path enters.\n"
    "Coordinates are written in the format \"(x,y)\", where x is the column index "
    "(0 = leftmost) and y is the row index (0 = topmost). For example, the "
    "top-left cell is \"(0,0)\", the cell to its right is \"(1,0)\", and the "
    "cell below it is \"(0,1)\".\n"
    "Examine the path carefully and list ALL cells that the path enters, including "
    "cells it just touches at an edge or corner.\n"
    "List cells in the order the path enters them.\n"
    "Respond ONLY with valid JSON in this exact format - no other text:\n"
    '{"cells_entered": ["(0,0)", "(1,0)", "(1,1)"]}'
)


def build_prompt(variant_name: str) -> str:
    cfg = VARIANTS[variant_name]
    grid_level = variant_grid_level(variant_name)
    subdivision = variant_subdivision(variant_name)
    visual_condition = cfg.get("visual_condition", "grid")

    if grid_level == 0:
        level_text = (
            "\n\nGrid level L0: the analysis grid is the original SEM acquisition-tile grid. "
            "Each coordinate refers to one original SEM atlas tile."
        )
    else:
        per_tile_cells = subdivision * subdivision
        level_text = (
            f"\n\nGrid level L{grid_level}: the analysis grid is a virtual subdivision of the atlas. "
            f"Each original SEM acquisition tile of 1920x1200 pixels is divided into {subdivision} by {subdivision} "
            f"smaller virtual cells ({per_tile_cells} cells per original tile). Coordinates refer "
            "to the resulting GLOBAL analysis grid across the whole atlas, not to local coordinates "
            "inside one original SEM tile."
        )

    if visual_condition == "path_only":
        if grid_level == 0:
            visual_text = (
                "\n\nNo grid lines or coordinate labels are drawn. Infer the original SEM tile "
                "boundaries from the regular structure of the stitched atlas."
            )
        else:
            visual_text = (
                f"\n\nNo subdivision grid lines or coordinate labels are drawn. You must mentally "
                f"subdivide each original SEM tile into a {subdivision} by {subdivision} grid "
                "and return the global virtual-cell coordinates entered by the path."
            )
    elif visual_condition == "grid":
        visual_text = (
            "\n\nYellow grid lines mark the analysis-grid cell boundaries. Use these lines "
            "to identify which cells the path enters."
        )
    elif visual_condition == "grid_labels":
        visual_text = (
            "\n\nYellow grid lines and coordinate labels are visible. Use both the grid "
            "boundaries and labels to identify which cells the path enters."
        )
    else:
        visual_text = ""

    return _PROMPT_SHARED + level_text + visual_text


# ---------------------------------------------------------------------------
# OpenAI config
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
# HTTP helpers
# ---------------------------------------------------------------------------



def aggregate_variant_metrics(metrics_list: list[dict]) -> dict:
    """Aggregate metrics across paths/runs for one experimental variant."""
    valid = [m for m in metrics_list if m and not m.get("parsing_error")]
    n_total = len(metrics_list)
    n_valid = len(valid)

    def values(key: str) -> list[float]:
        return [
            float(m[key])
            for m in valid
            if m.get(key) is not None
        ]

    def mean(xs: list[float]) -> float | None:
        if not xs:
            return None
        return round(sum(xs) / len(xs), 4)

    def variance(xs: list[float]) -> float | None:
        if len(xs) < 2:
            return 0.0 if len(xs) == 1 else None
        mu = sum(xs) / len(xs)
        return round(sum((x - mu) ** 2 for x in xs) / len(xs), 4)

    def sample_variance(xs: list[float]) -> float | None:
        if len(xs) < 2:
            return None
        mu = sum(xs) / len(xs)
        return round(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1), 4)

    def std(xs: list[float]) -> float | None:
        var = variance(xs)
        if var is None:
            return None
        return round(var ** 0.5, 4)

    def min_value(xs: list[float]) -> float | None:
        if not xs:
            return None
        return round(min(xs), 4)

    def max_value(xs: list[float]) -> float | None:
        if not xs:
            return None
        return round(max(xs), 4)

    def add_distribution_stats(out: dict, source_key: str, output_prefix: str) -> None:
        xs = values(source_key)
        out[f"mean_{output_prefix}"] = mean(xs)
        out[f"var_{output_prefix}"] = variance(xs)
        out[f"sample_var_{output_prefix}"] = sample_variance(xs)
        out[f"std_{output_prefix}"] = std(xs)
        out[f"min_{output_prefix}"] = min_value(xs)
        out[f"max_{output_prefix}"] = max_value(xs)

    out = {
        "n_paths": n_total,
        "n_parsed": n_valid,
        "n_parsing_failures": n_total - n_valid,
    }

    add_distribution_stats(out, "precision", "precision")
    add_distribution_stats(out, "recall", "recall")
    add_distribution_stats(out, "f1", "f1")
    add_distribution_stats(out, "tile_accuracy_percent", "tile_accuracy")
    add_distribution_stats(out, "missed_tile_rate", "missed_tile_rate")
    add_distribution_stats(out, "extra_tile_rate", "extra_tile_rate")

    exact_values = [
        1.0 if m.get("exact_match") else 0.0
        for m in valid
        if m.get("exact_match") is not None
    ]

    out["exact_match_rate"] = mean(exact_values)
    out["var_exact_match"] = variance(exact_values)
    out["sample_var_exact_match"] = sample_variance(exact_values)
    out["std_exact_match"] = std(exact_values)

    return out

def _http(method: str, url: str, body: dict | None = None, timeout: int = 60) -> dict:
    data    = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req     = _Req(url, data=data, headers=headers, method=method.upper())
    try:
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except _HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return {"error": str(e)}


def _http_binary(url: str, timeout: int = 180) -> bytes:
    try:
        with urlopen(url, timeout=timeout) as r:
            return r.read()
    except TimeoutError as exc:
        raise RuntimeError(
            f"Timed out after {timeout}s while fetching {url}. "
            "The SEM export endpoint is probably still rendering or stuck."
        ) from exc


def _sem(method: str, path: str, body: dict | None = None) -> dict:
    return _http(method, f"{SEM_URL}{path}", body)


def _check_service(url: str, label: str) -> bool:
    try:
        _http("GET", f"{url.rstrip('/')}/api/session/stats", timeout=5)
        return True
    except Exception as exc:
        print(f"  [warn] {label} not reachable at {url}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Trace helpers (agent_api.py / package_run.py compatible shape)
# ---------------------------------------------------------------------------

def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _safe_rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def _trace_call(
    *,
    tool: str,
    action: str | None,
    category: str,
    input_summary: str = "",
    result: str | dict | list | None = None,
    result_is_json: bool | None = None,
) -> dict:
    if result_is_json is None:
        result_is_json = isinstance(result, (dict, list))
    if isinstance(result, (dict, list)):
        result_text = json.dumps(_json_safe(result), ensure_ascii=False)
    elif result is None:
        result_text = ""
    else:
        result_text = str(result)
    if len(result_text) > 800:
        result_text = result_text[:800] + "…"
    return {
        "tool":           tool,
        "action":         action,
        "category":       category,
        "input_summary":  input_summary,
        "result":         result_text,
        "result_is_json": result_is_json,
    }


def _trace_step(step_no: int, thinking: str | None, calls: list[dict]) -> dict:
    return {
        "type":     "step",
        "step":     step_no,
        "thinking": thinking,
        "calls":    calls,
    }


def _write_trace_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    model: str,
    prompt: str,
    reply: str,
    steps: list[dict],
    global_traces_dir: Path | None = None,
) -> dict:
    """Write full_trace artifacts in the layout consumed by package_run.py.

    Also writes a raw trace copy to logs/traces by default so the trace is
    discoverable in the same place as traces emitted by agent_api.py.
    """
    trace_dir = run_dir / "full_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    raw_trace_dst = trace_dir / "agent_trace.json"
    model_msgs_dst = trace_dir / "model_messages.jsonl"
    tool_calls_dst = trace_dir / "tool_calls.jsonl"
    timeline_dst = trace_dir / "agent_timeline.jsonl"
    agent_reply_dst = trace_dir / "agent_reply.txt"

    trace_payload = {
        "run_id":       run_id,
        "case_study":   "case_study_3",
        "started_at":   started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "user_message": prompt,
        "model_name":   model,
        "reply":        reply,
        "steps":        steps,
    }
    _write_json(raw_trace_dst, trace_payload)

    global_trace_dst: Path | None = None
    if global_traces_dir is not None:
        global_traces_dir.mkdir(parents=True, exist_ok=True)
        safe_run_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_id)
        global_trace_dst = global_traces_dir / f"trace_{safe_run_id}.json"
        _write_json(global_trace_dst, trace_payload)

    agent_reply_dst.write_text(reply or "", encoding="utf-8")

    model_rows: list[dict] = []
    call_rows: list[dict] = []
    timeline_rows: list[dict] = []
    for step in steps:
        step_no = step.get("step")
        thinking = step.get("thinking")
        if thinking:
            model_rows.append({"step": step_no, "type": "thinking", "content": thinking})
            timeline_rows.append({"step": step_no, "event": "model_message", "content": thinking})
        for call in step.get("calls", []) or []:
            row = {
                "step":           step_no,
                "tool":           call.get("tool"),
                "action":         call.get("action"),
                "category":       call.get("category"),
                "input_summary":  call.get("input_summary"),
                "result":         call.get("result"),
                "result_is_json": call.get("result_is_json"),
            }
            call_rows.append(row)
            timeline_rows.append({"step": step_no, "event": "tool_call", **row})

    _write_jsonl(model_msgs_dst, model_rows)
    _write_jsonl(tool_calls_dst, call_rows)
    _write_jsonl(timeline_dst, timeline_rows)

    return {
        "raw_agent_trace": _safe_rel(raw_trace_dst, run_dir),
        "model_messages":  _safe_rel(model_msgs_dst, run_dir),
        "tool_calls":      _safe_rel(tool_calls_dst, run_dir),
        "agent_reply":     _safe_rel(agent_reply_dst, run_dir),
        "service_events":  _safe_rel(timeline_dst, run_dir),
        "global_trace":    str(global_trace_dst) if global_trace_dst else None,
    }


# ---------------------------------------------------------------------------
# VLM call (direct OpenAI vision - no agent loop)
# ---------------------------------------------------------------------------

def _call_vlm(png_path: Path, prompt: str, api_key: str, base_url: str, model: str,
              reasoning_effort: str = "medium") -> dict:
    """Call the OpenAI vision API with a PNG image and return the raw response.

    Returns
    -------
    dict with keys: ok, reply, model, usage, error
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {"ok": False, "reply": None, "error": "openai package not installed"}

    try:
        img_b64 = base64.b64encode(png_path.read_bytes()).decode()
    except Exception as exc:
        return {"ok": False, "reply": None, "error": f"Could not read PNG: {exc}"}

    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(api_key=api_key, **kwargs)

    create_kwargs: dict = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url":    f"data:image/png;base64,{img_b64}",
                        "detail": "high",
                    },
                },
            ],
        }],
    }
    # reasoning_effort is only supported by GPT-5.x models
    if reasoning_effort and reasoning_effort != "none" and "gpt-4" not in model:
        create_kwargs["reasoning_effort"] = reasoning_effort

    try:
        resp = client.chat.completions.create(**create_kwargs)
    except Exception as exc:
        return {"ok": False, "reply": None, "error": f"OpenAI call failed: {exc}"}

    reply = resp.choices[0].message.content or ""
    usage = {}
    if resp.usage:
        usage = {
            "prompt_tokens":     resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens":      resp.usage.total_tokens,
        }
    return {"ok": True, "reply": reply, "model": model, "usage": usage, "error": None}


# ---------------------------------------------------------------------------
# Atlas state + GT extraction
# ---------------------------------------------------------------------------

def _fetch_atlas_and_objects() -> tuple[dict, list[dict]]:
    """Return (atlas dict, objects list) from the currently running sem-service."""
    cam_state = _sem("GET", "/api/camera/state")
    if cam_state.get("uiMode") != "atlas":
        raise RuntimeError(
            f"sem-service is not in atlas mode (uiMode={cam_state.get('uiMode')!r}). "
            "Enter atlas mode first: load a tile grid, then click 'Atlas'."
        )
    atlas   = cam_state.get("atlas", {})
    objects = _sem("GET", "/api/objects")
    if not isinstance(objects, list):
        objects = objects.get("objects", []) if isinstance(objects, dict) else []
    return atlas, objects


def _assert_path_exists(objects: list[dict]) -> None:
    """Raise if no freehand path is drawn on the canvas."""
    paths = [o for o in objects if o.get("type") == "freehand"]
    if not paths:
        raise RuntimeError(
            "No freehand path found on the canvas. "
            "Draw a path across the atlas tiles using the pen tool, then re-run."
        )


# ---------------------------------------------------------------------------
# PNG export per variant
# ---------------------------------------------------------------------------

def _export_variant_png(
    variant_name: str,
    images_dir: Path,
    *,
    image_cache_dir: Path | None = None,
    use_image_cache: bool = True,
) -> tuple[Path, bool, list[dict]]:
    """Set overlay state, export PNG, return saved path, cache flag, and trace calls."""
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = images_dir / f"{variant_name}.png"
    trace_calls: list[dict] = []

    cache_path: Path | None = None
    if image_cache_dir is not None:
        image_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = image_cache_dir / f"{variant_name}.png"

    if use_image_cache and cache_path is not None and cache_path.exists():
        shutil.copy2(cache_path, out_path)
        size = out_path.stat().st_size
        print(f"    → using cached PNG {cache_path.name} ({size:,} bytes)")
        trace_calls.append(_trace_call(
            tool="filesystem",
            action="copy_cached_variant_png",
            category="image",
            input_summary=f"src={cache_path}, dst={out_path}",
            result={"ok": True, "bytes": size, "cache_hit": True},
        ))
        return out_path, True, trace_calls

    grid_level = variant_grid_level(variant_name)
    subdivision = variant_subdivision(variant_name)
    overlay = {
        **VARIANTS[variant_name]["overlay"],
        **ATLAS_OVERLAY_STYLE,
        "gridLevel": grid_level,
        "subdivision": subdivision,
        "labelFontSize": LABEL_FONT_SIZE_BY_GRID_LEVEL.get(grid_level, ATLAS_OVERLAY_STYLE["labelFontSize"]),
    }
    overlay_resp = _sem("POST", "/api/atlas/overlay", overlay)
    trace_calls.append(_trace_call(
        tool="sem_service",
        action="POST /api/atlas/overlay",
        category="image",
        input_summary=json.dumps(overlay),
        result=overlay_resp,
    ))
    time.sleep(1.0)   # let the renderer settle before export

    print(f"    → exporting PNG from sem-service...")
    t0 = time.time()
    png_bytes = _http_binary(f"{SEM_URL}/api/export/png", timeout=180)
    dt = time.time() - t0

    out_path.write_bytes(png_bytes)
    if cache_path is not None:
        cache_path.write_bytes(png_bytes)

    trace_calls.append(_trace_call(
        tool="sem_service",
        action="GET /api/export/png",
        category="image",
        input_summary=f"variant={variant_name}, timeout=180",
        result={
            "ok": True,
            "bytes": len(png_bytes),
            "seconds": round(dt, 3),
            "saved_to": str(out_path),
            "cache_path": str(cache_path) if cache_path else None,
            "cache_hit": False,
        },
    ))

    print(f"    → export finished in {dt:.1f}s ({len(png_bytes):,} bytes)")
    return out_path, False, trace_calls


def _reset_overlay() -> None:
    """Turn off all overlays after export."""
    try:
        _sem("POST", "/api/atlas/overlay", {"grid": False, "labels": False, "gridLevel": 0, "subdivision": 1})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Single variant run
# ---------------------------------------------------------------------------

def run_variant(
    variant_name: str,
    gt: dict,
    images_dir: Path,
    *,
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str = "medium",
    dry_run: bool = False,
    image_cache_dir: Path | None = None,
    use_image_cache: bool = True,
) -> dict:
    """Run one experimental variant. Returns a prediction dict."""
    v   = VARIANTS[variant_name]
    ts  = datetime.now(timezone.utc).isoformat()
    prompt = build_prompt(variant_name)
    grid_level = variant_grid_level(variant_name)
    subdivision = variant_subdivision(variant_name)
    visual_condition = v.get("visual_condition")

    print(f"\n  [{variant_name}]  {v['description']}")

    if dry_run:
        print(f"    → [dry-run] prompt preview:")
        print("      " + prompt[:200].replace("\n", "\n      ") + "…")
        return {
            "variant":                  variant_name,
            "grid_level":               grid_level,
            "subdivision":              subdivision,
            "visual_condition":         visual_condition,
            "timestamp":                ts,
            "status":                   "dry_run",
            "raw_reply":                None,
            "prompt":                   prompt,
            "predicted_tile_sequence":  [],
            "predicted_tile_set":       [],
            "parsing_error":            None,
            "metrics":                  None,
            "turn_count":               gt.get("turn_count"),
            "turn_difficulty":          gt.get("turn_difficulty"),
            "turn_tile_sequence_l0":    gt.get("turn_tile_sequence_l0"),
            "turn_coordinate_system":   gt.get("turn_coordinate_system"),
            "turn_base_tile_width":     gt.get("turn_base_tile_width"),
            "turn_base_tile_height":    gt.get("turn_base_tile_height"),
        }

    # Export image for this variant, or reuse the batch cache if available.
    png_path, image_cache_hit, trace_calls = _export_variant_png(
        variant_name,
        images_dir,
        image_cache_dir=image_cache_dir,
        use_image_cache=use_image_cache,
    )
    print(f"    → image ready {png_path.name}  ({png_path.stat().st_size:,} bytes)  cache_hit={image_cache_hit}")

    # Call VLM
    t0 = time.time()
    vlm_result = _call_vlm(png_path, prompt, api_key, base_url, model, reasoning_effort)
    trace_calls.append(_trace_call(
        tool="openai.chat.completions",
        action="vision_tile_prediction",
        category="vision",
        input_summary=f"model={model}, image={png_path.name}, variant={variant_name}",
        result={
            "ok": bool(vlm_result.get("ok")),
            "seconds": round(time.time() - t0, 3),
            "usage": vlm_result.get("usage"),
            "error": vlm_result.get("error"),
        },
    ))
    if not vlm_result["ok"]:
        print(f"    → [error] VLM call failed: {vlm_result['error']}", file=sys.stderr)
        return {
            "variant":                  variant_name,
            "grid_level":               grid_level,
            "subdivision":              subdivision,
            "visual_condition":         visual_condition,
            "timestamp":                ts,
            "status":                   "vlm_failed",
            "error":                    vlm_result["error"],
            "raw_reply":                None,
            "prompt":                   prompt,
            "predicted_tile_sequence":  [],
            "predicted_tile_set":       [],
            "parsing_error":            None,
            "metrics":                  compute_prediction_metrics(
                                            [], [], gt["gt_tile_sequence"], gt["gt_tile_set"],
                                            parsing_error="vlm_failed",
                                            turn_count=gt.get("turn_count"),
                                            turn_difficulty=gt.get("turn_difficulty"),
                                            turn_tile_sequence_l0=gt.get("turn_tile_sequence_l0"),
                                        ),
            "image_path":               str(png_path),
            "image_cache_hit":          image_cache_hit,
            "turn_count":               gt.get("turn_count"),
            "turn_difficulty":          gt.get("turn_difficulty"),
            "turn_tile_sequence_l0":    gt.get("turn_tile_sequence_l0"),
            "turn_coordinate_system":   gt.get("turn_coordinate_system"),
            "turn_base_tile_width":     gt.get("turn_base_tile_width"),
            "turn_base_tile_height":    gt.get("turn_base_tile_height"),
            "trace_calls":              trace_calls,
        }

    raw_reply = vlm_result["reply"]
    parsed    = parse_tile_ids(raw_reply)
    metrics   = compute_prediction_metrics(
        predicted_tile_sequence = parsed["predicted_tile_sequence"],
        predicted_tile_set      = parsed["predicted_tile_set"],
        gt_tile_sequence        = gt["gt_tile_sequence"],
        gt_tile_set             = gt["gt_tile_set"],
        parsing_error           = parsed["parsing_error"],
        turn_count              = gt.get("turn_count"),
        turn_difficulty         = gt.get("turn_difficulty"),
        turn_tile_sequence_l0   = gt.get("turn_tile_sequence_l0"),
    )

    status = "completed" if parsed["ok"] else "parse_failed"
    print(
        f"    → status={status}"
        f"  predicted={parsed['predicted_tile_set']}"
        f"  gt={gt['gt_tile_set']}"
        f"  f1={metrics['f1']}"
        f"  exact={metrics['exact_match']}"
    )
    if not parsed["ok"]:
        print(f"    → [warn] parsing failed: {parsed['parsing_error']}")
        print(f"    → [warn] raw reply (last 300 chars): …{raw_reply[-300:]}")

    return {
        "variant":                  variant_name,
            "grid_level":               grid_level,
            "subdivision":              subdivision,
            "visual_condition":         visual_condition,
        "timestamp":                ts,
        "status":                   status,
        "raw_reply":                raw_reply,
        "prompt":                   prompt,
        "model":                    vlm_result.get("model"),
        "usage":                    vlm_result.get("usage"),
        "predicted_tile_sequence":  parsed["predicted_tile_sequence"],
        "predicted_tile_set":       parsed["predicted_tile_set"],
        "predicted_cell_sequence":  parsed.get("predicted_cell_sequence", parsed["predicted_tile_sequence"]),
        "predicted_cell_set":       parsed.get("predicted_cell_set", parsed["predicted_tile_set"]),
        "gt_cell_sequence":         gt.get("gt_cell_sequence", gt["gt_tile_sequence"]),
        "gt_cell_set":              gt.get("gt_cell_set", gt["gt_tile_set"]),
        "cell_width":               gt.get("cell_width"),
        "cell_height":              gt.get("cell_height"),
        "effective_cols":           gt.get("effective_cols"),
        "effective_rows":           gt.get("effective_rows"),
        "turn_count":               gt.get("turn_count"),
        "turn_difficulty":          gt.get("turn_difficulty"),
        "turn_tile_sequence_l0":    gt.get("turn_tile_sequence_l0"),
        "turn_coordinate_system":   gt.get("turn_coordinate_system"),
        "turn_base_tile_width":     gt.get("turn_base_tile_width"),
        "turn_base_tile_height":    gt.get("turn_base_tile_height"),
        "parsing_error":            parsed["parsing_error"],
        "metrics":                  metrics,
        "image_path":               str(png_path),
        "image_cache_hit":          image_cache_hit,
        "trace_calls":              trace_calls,
    }


# ---------------------------------------------------------------------------
# Single path orchestrator (all variants)
# ---------------------------------------------------------------------------

def run_path(
    variant_names: list[str],
    *,
    out_dir: Path,
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str = "medium",
    dry_run: bool = False,
    inter_variant_delay: int = 1,
    run_dir: Path | None = None,
    run_id: str | None = None,
    image_cache_dir: Path | None = None,
    use_image_cache: bool = True,
    traces_dir: Path,
) -> dict:
    """Run all requested variants for the currently drawn atlas path.

    Saves all artifacts under out_dir/runs/<run_id>/  and returns the manifest.
    """
    ts_start = datetime.now(timezone.utc)
    ts_str = ts_start.strftime("%Y%m%d_%H%M%S")

    if run_id is None:
        run_id = f"case3_{ts_str}"

    if run_dir is None:
        run_dir = out_dir / "runs" / run_id

    # --- Fetch canvas state and compute GT ---
    atlas, objects = _fetch_atlas_and_objects()
    initial_trace_calls = [
        _trace_call(
            tool="sem_service",
            action="GET /api/camera/state + GET /api/objects",
            category="meta",
            input_summary="fetch current atlas state and canvas objects",
            result={
                "uiMode": "atlas",
                "atlas": atlas,
                "n_objects": len(objects),
                "n_freehand": sum(1 for o in objects if o.get("type") == "freehand"),
            },
        )
    ]
    _assert_path_exists(objects)

    n_paths = sum(1 for o in objects if o.get("type") == "freehand")
    gt_by_variant = {
        variant_name: gt_from_objects(
            objects,
            atlas,
            grid_level=variant_grid_level(variant_name),
        )
        for variant_name in variant_names
    }
    primary_gt = gt_by_variant[variant_names[0]] if variant_names else gt_from_objects(objects, atlas, grid_level=0)

    atlas_id = (
        f"region_{atlas.get('region', '?')}_fw_{atlas.get('fw', '?')}"
        if atlas.get("region") is not None else "atlas"
    )

    print(f"\n{'='*70}")
    print(f"  Case Study 3 - SEM Atlas Path-to-Tile Reasoning")
    print(f"  Run ID    : {run_id}")
    print(f"  Atlas     : {atlas_id}  "
          f"({atlas.get('cols', '?')}×{atlas.get('rows', '?')} tiles, "
          f"{atlas.get('tileWidth', '?')}×{atlas.get('tileHeight', '?')} px/tile)")
    print(f"  Paths     : {n_paths} freehand object(s)")
    print("  GT cells  : per variant/grid level (saved in ground_truth/gt_cells_by_variant.json)")
    print(f"  Variants  : {', '.join(variant_names)}")
    print(f"  Model     : {model}")
    print(f"{'='*70}")

    if not dry_run:
        images_dir = run_dir / "rendered_variants"
        images_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ground_truth").mkdir(exist_ok=True)
        (run_dir / "predictions").mkdir(exist_ok=True)
        (run_dir / "metrics").mkdir(exist_ok=True)
        (run_dir / "full_trace").mkdir(exist_ok=True)

        gt_payload = {
            "run_id":         run_id,
            "atlas_id":       atlas_id,
            "atlas":          atlas,
            "n_path_objects": n_paths,
            "variants": {
                variant_name: {
                    "grid_level": gt_by_variant[variant_name].get("grid_level"),
                    "subdivision": gt_by_variant[variant_name].get("subdivision"),
                    "visual_condition": VARIANTS[variant_name].get("visual_condition"),
                    "coordinate_system": gt_by_variant[variant_name].get("coordinate_system"),
                    "cell_width": gt_by_variant[variant_name].get("cell_width"),
                    "cell_height": gt_by_variant[variant_name].get("cell_height"),
                    "effective_cols": gt_by_variant[variant_name].get("effective_cols"),
                    "effective_rows": gt_by_variant[variant_name].get("effective_rows"),
                    "gt_cell_sequence": gt_by_variant[variant_name].get("gt_cell_sequence", gt_by_variant[variant_name]["gt_tile_sequence"]),
                    "gt_cell_set": gt_by_variant[variant_name].get("gt_cell_set", gt_by_variant[variant_name]["gt_tile_set"]),
                    "gt_tile_sequence": gt_by_variant[variant_name]["gt_tile_sequence"],
                    "gt_tile_set": gt_by_variant[variant_name]["gt_tile_set"],
                    "path_points": gt_by_variant[variant_name]["path_points"],
                    "turn_count": gt_by_variant[variant_name].get("turn_count"),
                    "turn_difficulty": gt_by_variant[variant_name].get("turn_difficulty"),
                    "turn_tile_sequence_l0": gt_by_variant[variant_name].get("turn_tile_sequence_l0"),
                    "turn_coordinate_system": gt_by_variant[variant_name].get("turn_coordinate_system"),
                    "turn_base_tile_width": gt_by_variant[variant_name].get("turn_base_tile_width"),
                    "turn_base_tile_height": gt_by_variant[variant_name].get("turn_base_tile_height"),
                }
                for variant_name in variant_names
            },
        }
        (run_dir / "ground_truth" / "gt_cells_by_variant.json").write_text(
            json.dumps(gt_payload, indent=2)
        )
        # Backward-compatible copy for older analysis scripts; contains all variants.
        (run_dir / "ground_truth" / "gt_tiles.json").write_text(
            json.dumps(gt_payload, indent=2)
        )
    else:
        images_dir = run_dir / "rendered_variants"  # won't be created

    predictions: list[dict] = []
    for i, variant_name in enumerate(variant_names):
        pred = run_variant(
            variant_name, gt_by_variant[variant_name], images_dir,
            api_key=api_key, base_url=base_url, model=model,
            reasoning_effort=reasoning_effort,
            dry_run=dry_run,
            image_cache_dir=image_cache_dir,
            use_image_cache=use_image_cache,
        )
        predictions.append(pred)

        if not dry_run:
            (run_dir / "predictions" / f"{variant_name}.json").write_text(
                json.dumps(pred, indent=2)
            )

        if i < len(variant_names) - 1:
            time.sleep(inter_variant_delay)

    # Reset overlay after all exports
    if not dry_run:
        _reset_overlay()

    # Build per-variant metrics summary
    variant_metrics = {
        p["variant"]: p["metrics"]
        for p in predictions
        if p["metrics"]
    }

    variant_aggregates = {
        p["variant"]: aggregate_variant_metrics([p["metrics"]])
        for p in predictions
        if p["metrics"] and p["status"] != "dry_run"
    }

    metrics_summary = {
        "n_variants": len(predictions),
        "n_completed": sum(1 for p in predictions if p["status"] == "completed"),
        "n_failed": sum(1 for p in predictions if p["status"] in ("vlm_failed", "parse_failed")),
        "per_variant": variant_metrics,
        "aggregate_per_variant": variant_aggregates,
    }

    if not dry_run:
        (run_dir / "metrics" / "per_variant_results.json").write_text(
            json.dumps(metrics_summary, indent=2)
        )

    logs: dict = {}
    if not dry_run:
        completed_at = datetime.now(timezone.utc)
        trace_steps: list[dict] = []
        step_no = 1
        trace_steps.append(_trace_step(
            step_no,
            "Fetched atlas state, canvas objects, derived grid-level ground-truth cell traversals, and classified original-tile turn difficulty.",
            initial_trace_calls,
        ))
        step_no += 1
        replies: list[str] = []
        for p in predictions:
            variant = p.get("variant", "unknown")
            replies.append(f"[{variant}]\n{p.get('raw_reply') or ''}")
            trace_steps.append(_trace_step(
                step_no,
                f"Ran Case Study 3 variant {variant}: exported or reused its rendered atlas image, then requested cell prediction from the VLM.",
                p.get("trace_calls", []),
            ))
            step_no += 1
        logs = _write_trace_artifacts(
            run_dir=run_dir,
            run_id=run_id,
            started_at=ts_start,
            completed_at=completed_at,
            model=model,
            prompt="\n\n--- VARIANT PROMPTS ---\n\n".join(build_prompt(v) for v in variant_names),
            reply="\n\n".join(replies),
            steps=trace_steps,
            global_traces_dir=traces_dir,
        )

    manifest = {
        "run_id":         run_id,
        "case_study":     "case_study_3",
        "atlas_id":       atlas_id,
        "started_at":     ts_start.isoformat(),
        "completed_at":   datetime.now(timezone.utc).isoformat(),
        "status":         "completed" if not dry_run else "dry_run",
        "model":          model,
        "gt_cell_set":    primary_gt.get("gt_cell_set", primary_gt["gt_tile_set"]),
        "gt_cell_sequence": primary_gt.get("gt_cell_sequence", primary_gt["gt_tile_sequence"]),
        "gt_tile_set":    primary_gt["gt_tile_set"],
        "gt_tile_sequence": primary_gt["gt_tile_sequence"],
        "turn_count":     primary_gt.get("turn_count"),
        "turn_difficulty": primary_gt.get("turn_difficulty"),
        "turn_tile_sequence_l0": primary_gt.get("turn_tile_sequence_l0"),
        "turn_coordinate_system": primary_gt.get("turn_coordinate_system"),
        "turn_base_tile_width": primary_gt.get("turn_base_tile_width"),
        "turn_base_tile_height": primary_gt.get("turn_base_tile_height"),
        "gt_by_variant": {
            variant_name: {
                "grid_level": gt_by_variant[variant_name].get("grid_level"),
                "subdivision": gt_by_variant[variant_name].get("subdivision"),
                "visual_condition": VARIANTS[variant_name].get("visual_condition"),
                "cell_width": gt_by_variant[variant_name].get("cell_width"),
                "cell_height": gt_by_variant[variant_name].get("cell_height"),
                "effective_cols": gt_by_variant[variant_name].get("effective_cols"),
                "effective_rows": gt_by_variant[variant_name].get("effective_rows"),
                "gt_cell_set": gt_by_variant[variant_name].get("gt_cell_set", gt_by_variant[variant_name]["gt_tile_set"]),
                "gt_cell_sequence": gt_by_variant[variant_name].get("gt_cell_sequence", gt_by_variant[variant_name]["gt_tile_sequence"]),
                "turn_count": gt_by_variant[variant_name].get("turn_count"),
                "turn_difficulty": gt_by_variant[variant_name].get("turn_difficulty"),
                "turn_tile_sequence_l0": gt_by_variant[variant_name].get("turn_tile_sequence_l0"),
                "turn_coordinate_system": gt_by_variant[variant_name].get("turn_coordinate_system"),
            }
            for variant_name in variant_names
        },
        "n_path_objects": n_paths,
        "variants_run":   variant_names,
        "predictions":    {p["variant"]: p["status"] for p in predictions},
        "image_cache": {
            "enabled": bool(use_image_cache and image_cache_dir is not None),
            "cache_dir": str(image_cache_dir) if image_cache_dir else None,
            "per_variant": {
                p["variant"]: {
                    "cache_hit": p.get("image_cache_hit"),
                    "image_path": p.get("image_path"),
                    "grid_level": p.get("grid_level"),
                    "subdivision": p.get("subdivision"),
                    "visual_condition": p.get("visual_condition"),
                }
                for p in predictions
            },
        },
        "logs": logs,
        "metrics_summary": metrics_summary,
    }

    if not dry_run:
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\n  Run saved → {run_dir}")
        print(f"  Images   → {images_dir}/")
        if logs:
            print(f"  Trace    → {run_dir / 'full_trace'}")
            if logs.get('global_trace'):
                print(f"  Global trace → {logs['global_trace']}")

    return manifest

# ---------------------------------------------------------------------------
# Batch multiple runs
# ---------------------------------------------------------------------------


def run_batch(
    variant_names: list[str],
    *,
    n_runs: int,
    out_dir: Path,
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str = "medium",
    dry_run: bool = False,
    inter_variant_delay: int = 1,
    use_image_cache: bool = True,
    traces_dir: Path,
) -> dict:
    """Run multiple repetitions into one batch folder and aggregate by variant."""
    ts_start = datetime.now(timezone.utc)
    ts_str = ts_start.strftime("%Y%m%d_%H%M%S")
    batch_id = f"case3_batch_{ts_str}"
    batch_dir = out_dir / "runs" / batch_id
    image_cache_dir = batch_dir / "render_cache" if use_image_cache else None

    if not dry_run:
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "metrics").mkdir(exist_ok=True)
        if image_cache_dir is not None:
            image_cache_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print("  Case Study 3 - Batch Run")
    print(f"  Batch ID  : {batch_id}")
    print(f"  Runs      : {n_runs}")
    print(f"  Variants  : {', '.join(variant_names)}")
    print(f"  Output    : {batch_dir}")
    print(f"  PNG cache : {image_cache_dir if image_cache_dir is not None else 'disabled'}")
    print(f"{'='*70}")

    manifests: list[dict] = []

    for run_idx in range(1, n_runs + 1):
        run_label = f"run_{run_idx:03d}"
        child_run_id = f"{batch_id}_{run_label}"
        child_run_dir = batch_dir / run_label

        print(f"\n{'-'*70}")
        print(f"  Batch repetition {run_idx}/{n_runs} - {run_label}")
        print(f"{'-'*70}")

        manifest = run_path(
            variant_names,
            out_dir=out_dir,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            dry_run=dry_run,
            inter_variant_delay=inter_variant_delay,
            run_dir=child_run_dir,
            run_id=child_run_id,
            image_cache_dir=image_cache_dir,
            use_image_cache=use_image_cache,
            traces_dir=traces_dir,
        )
        manifests.append(manifest)

    aggregate_per_variant = aggregate_manifests_by_variant(manifests)

    batch_manifest = {
        "batch_id": batch_id,
        "case_study": "case_study_3",
        "started_at": ts_start.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if not dry_run else "dry_run",
        "model": model,
        "n_runs_requested": n_runs,
        "n_runs_completed": sum(
            1 for m in manifests
            if m.get("status") in ("completed", "dry_run")
        ),
        "variants_run": variant_names,
        "variant_metadata": {
            variant_name: {
                "grid_level": variant_grid_level(variant_name),
                "subdivision": variant_subdivision(variant_name),
                "visual_condition": VARIANTS[variant_name].get("visual_condition"),
                "exploratory": bool(VARIANTS[variant_name].get("exploratory", False)),
            }
            for variant_name in variant_names
        },
        "run_ids": [m.get("run_id") for m in manifests],
        "image_cache": {
            "enabled": bool(use_image_cache and image_cache_dir is not None),
            "cache_dir": str(image_cache_dir) if image_cache_dir else None,
            "cached_variants": sorted(p.name for p in image_cache_dir.glob("*.png"))
                if image_cache_dir and image_cache_dir.exists() else [],
        },
        "aggregate_per_variant": aggregate_per_variant,
    }

    if not dry_run:
        (batch_dir / "batch_manifest.json").write_text(
            json.dumps(batch_manifest, indent=2)
        )
        (batch_dir / "metrics" / "aggregate_per_variant.json").write_text(
            json.dumps(aggregate_per_variant, indent=2)
        )

        print(f"\n  Batch saved      → {batch_dir}")
        print(f"  Batch aggregate  → {batch_dir / 'metrics' / 'aggregate_per_variant.json'}")

    return batch_manifest



# ---------------------------------------------------------------------------
# Aggregate across completed runs
# ---------------------------------------------------------------------------

def aggregate_runs(out_dir: Path) -> dict[str, dict]:
    """Read all run_manifest.json files and compute per-variant aggregate statistics."""
    runs_dir = out_dir / "runs"
    if not runs_dir.exists():
        return {}

    all_metrics: dict[str, list] = {}

    for manifest_path in sorted(runs_dir.rglob("run_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue

        for v_name, m in manifest.get("metrics_summary", {}).get("per_variant", {}).items():
            if m and not m.get("parsing_error"):
                all_metrics.setdefault(v_name, []).append(m)

    return {
        v_name: aggregate_variant_metrics(metrics_list)
        for v_name, metrics_list in all_metrics.items()
    }


def aggregate_manifests_by_variant(manifests: list[dict]) -> dict[str, dict]:
    """Aggregate metrics across a provided list of run manifests, grouped by variant."""
    all_metrics: dict[str, list[dict]] = {}

    for manifest in manifests:
        per_variant = manifest.get("metrics_summary", {}).get("per_variant", {})
        for v_name, m in per_variant.items():
            if m and not m.get("parsing_error"):
                all_metrics.setdefault(v_name, []).append(m)

    return {
        v_name: aggregate_variant_metrics(metrics_list)
        for v_name, metrics_list in all_metrics.items()
    }


def _assert_trace_dir_writable(traces_dir: Path) -> None:
    """Fail early if global trace logging cannot write to logs/traces."""
    try:
        traces_dir.mkdir(parents=True, exist_ok=True)
        test_path = traces_dir / ".case3_trace_write_test"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Trace logging is required, but {traces_dir} is not writable: {exc}\n\n"
            "Fix one of these before running:\n"
            f"  sudo chown -R \"$USER:$USER\" {traces_dir}\n"
            f"  chmod -R u+rwX {traces_dir}\n"
            "or set --traces-dir to a writable directory."
        ) from exc

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Case Study 3 - SEM Atlas Path-to-Tile Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="Run variants for the currently drawn atlas path",
    )
    p.add_argument(
        "--variant",
        choices=list(VARIANTS),
        help="Run a single variant (default: main grid-level matrix; excludes exploratory L2_path_only)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: L0_path_only only, exits after one variant",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List variant definitions and check atlas state, then exit",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts without calling the VLM API or exporting images",
    )
    p.add_argument(
        "--out-dir",
        default=str(_PROJECT_ROOT / "outputs" / "case_study_3"),
        help="Output root directory (default: outputs/case_study_3)",
    )
    p.add_argument(
        "--sem-url",
        default=SEM_URL,
        help="sem-service base URL (default: http://localhost:3000)",
    )
    p.add_argument(
        "--delay",
        type=int,
        default=1,
        help="Seconds to wait between variant exports (default: 1)",
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of repeated runs to execute for the selected variant(s) (default: 1)",
    )
    p.add_argument(
        "--no-image-cache",
        action="store_true",
        help="Disable the batch-level rendered PNG cache and re-export images every run",
    )
    p.add_argument(
    "--traces-dir",
    default=str(_PROJECT_ROOT / "logs" / "traces"),
    help="Required global trace output directory (default: logs/traces)",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global SEM_URL

    args    = _parse_args(argv)
    SEM_URL = args.sem_url
    out_dir = Path(args.out_dir)
    traces_dir = Path(args.traces_dir)
    if not args.dry_run:
        _assert_trace_dir_writable(traces_dir)    
    # ------------------------------------------------------------------
    # --list
    # ------------------------------------------------------------------
    if args.list:
        print("\nCase Study 3 - SEM Atlas Path-to-Tile Reasoning")
        print(f"  sem-service : {SEM_URL}")
        print(f"\nVariants ({len(VARIANTS)}):")
        for v_name, cfg in VARIANTS.items():
            print(f"  {v_name:35s}  L{cfg.get('grid_level', 0)}  {cfg.get('visual_condition', '?'):12s}  {cfg['description']}")

        print("\nAtlas state:")
        try:
            cam = _sem("GET", "/api/camera/state")
            mode = cam.get("uiMode", "?")
            atlas = cam.get("atlas", {})
            print(f"  uiMode  : {mode}")
            if mode == "atlas":
                print(
                    f"  atlas   : {atlas.get('cols', '?')}×{atlas.get('rows', '?')} tiles"
                    f"  ({atlas.get('tileWidth', '?')}×{atlas.get('tileHeight', '?')} px/tile)"
                )
                objects = _sem("GET", "/api/objects")
                if isinstance(objects, dict):
                    objects = objects.get("objects", [])
                n_paths = sum(1 for o in objects if o.get("type") == "freehand")
                print(f"  paths   : {n_paths} freehand object(s) on canvas")
            else:
                print("  [!] Not in atlas mode - enter atlas mode first")
        except Exception as exc:
            print(f"  [error] Could not reach sem-service: {exc}", file=sys.stderr)
        return

    print("\nCase Study 3 - SEM Atlas Path-to-Tile Reasoning")
    print(f"  sem-service : {SEM_URL}")
    print(f"  output      : {out_dir}")

    # ------------------------------------------------------------------
    # Determine variants to run
    # ------------------------------------------------------------------
    if args.smoke:
        variant_names = ["L0_path_only"]
    elif args.variant:
        variant_names = [args.variant]
    else:
        variant_names = list(DEFAULT_VARIANTS)

    # ------------------------------------------------------------------
    # Pre-flight service check
    # ------------------------------------------------------------------
    if not args.dry_run:
        if not _check_service(SEM_URL, "sem-service"):
            print("\nERROR: sem-service not reachable. Aborting.", file=sys.stderr)
            sys.exit(1)
        print("  Services    : OK")
    else:
        print("  [dry-run mode - no services will be called for VLM, but sem-service needed for GT]")

    # ------------------------------------------------------------------
    # Load OpenAI config
    # ------------------------------------------------------------------
    api_key, base_url, model, reasoning_effort = _load_openai_config()
    if not api_key and not args.dry_run:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or environment.", file=sys.stderr)
        sys.exit(1)
    print(f"  model       : {model}  (reasoning_effort={reasoning_effort})")
    print(f"  variants    : {variant_names}")



    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    if not (args.run or args.smoke or args.dry_run or args.variant):
        # Default behaviour: run all variants (equivalent to --run)
        pass
    
    if args.n_runs < 1:
        print("ERROR: --n-runs must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.n_runs == 1:
        result = run_path(
            variant_names,
            out_dir=out_dir,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            dry_run=args.dry_run,
            inter_variant_delay=args.delay,
            image_cache_dir=None,
            use_image_cache=not args.no_image_cache,
            traces_dir=traces_dir,
        )
        is_batch = False
    else:
        result = run_batch(
            variant_names,
            n_runs=args.n_runs,
            out_dir=out_dir,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            dry_run=args.dry_run,
            inter_variant_delay=args.delay,
            use_image_cache=not args.no_image_cache,
            traces_dir=traces_dir,
        )
        is_batch = True

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")

    if is_batch:
        print(f"  SUMMARY - batch_id: {result['batch_id']}")
        print(f"  runs      : {result.get('n_runs_completed', '?')}/{result.get('n_runs_requested', '?')}")
        print(f"  variants  : {', '.join(result.get('variants_run', []))}")

        agg = result.get("aggregate_per_variant", {})
        if agg:
            print("\n  Aggregate across batch, grouped by variant:")
            for v_name, stats in agg.items():
                print(
                    f"  {v_name:35s}"
                    f"  n={stats.get('n_paths', '?')}"
                    f"  mean_precision={stats.get('mean_precision', '?')}±{stats.get('std_precision', '?')}"
                    f"  mean_recall={stats.get('mean_recall', '?')}±{stats.get('std_recall', '?')}"
                    f"  mean_f1={stats.get('mean_f1', '?')}±{stats.get('std_f1', '?')}"
                    f"  exact_match_rate={stats.get('exact_match_rate', '?')}"
                )
    else:
        manifest = result
        print(f"  SUMMARY - run_id: {manifest['run_id']}")
        print(f"  GT cells  : {manifest.get('gt_cell_set', manifest.get('gt_tile_set'))}")
        print(
            f"  Turns     : {manifest.get('turn_count', '?')} "
            f"({manifest.get('turn_difficulty', '?')})"
        )

        ms = manifest.get("metrics_summary", {})
        print(f"  completed : {ms.get('n_completed', '?')}/{ms.get('n_variants', '?')}")

        if not args.dry_run:
            pv = ms.get("per_variant", {})
            for v_name in variant_names:
                m = pv.get(v_name, {})
                if m:
                    print(
                        f"  {v_name:35s}"
                        f"  pred={m.get('predicted_cell_set', m.get('predicted_tile_set', '?'))}"
                        f"  f1={m.get('f1', '?')}"
                        f"  exact={m.get('exact_match', '?')}"
                        f"  turns={m.get('turn_count', '?')}"
                        f"/{m.get('turn_difficulty', '?')}"
                    )

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
