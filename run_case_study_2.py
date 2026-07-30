"""run_case_study_2.py - Case Study 2: SEM Particle Counting Evaluation

Tests whether SAM2 segmentation preprocessing improves VLM particle counting
accuracy on labeled SEM particle images from Dataset_Images_Labeled_W_Metadata/Particles/.

Seven experimental conditions
------------------------------
  E1  raw_vlm              - VLM counts directly from the raw image (baseline)
  E2  sam2_deterministic   - SAM2 colour mask -> shared deterministic mask count (no VLM)
  E3  imggen_deterministic - ImgGen colour instance mask -> shared deterministic mask count
  E4  sam2_overlay_vlm     - SAM2 colour mask overlaid on image -> VLM counts annotated image
  E5  imggen_overlay_vlm   - ImgGen colour overlay on canvas -> VLM counts annotated regions
  E6  sam3_deterministic   - SAM3 colour mask -> shared deterministic mask count (no VLM)
  E7  sam3_overlay_vlm     - SAM3 colour overlay on canvas -> VLM counts annotated regions

Ground truth
------------
Annotate particle counts manually in data/case_study_2/particle_gt.json:

    {
        "L2_000b4469b73e3fb3558d20b33b91fcb0": {
            "gt_count": 23,
            "gt_count_mode": "point",
            "notes": "clearly separated particles, good contrast"
        },
        "Tile_luca_001": {
            "gt_count": 6,
            "gt_uncertain_count": 2,
            "gt_count_mode": "interval",
            "notes": "6 confident labels, 2 uncertain labels; accepted interval is [6, 8]"
        },
        ...
    }

Usage
-----
    # List GT-annotated samples and variants
    python run_case_study_2.py --list

    # Dry-run: preview prompts without calling services
    python run_case_study_2.py --sample-id L2_xxx --variant E1_raw_vlm --dry-run

    # Single sample, single variant
    python run_case_study_2.py --sample-id L2_xxx --variant E1_raw_vlm

    # Registered dataset: first n GT-annotated images from Grid_Scan_Paper
    python run_case_study_2.py --dataset grid_scan_paper --n 5 --variant E7_sam3_overlay_vlm

    # Single sample, all 5 variants
    python run_case_study_2.py --sample-id L2_xxx --all-variants

    # All GT-annotated samples x all variants
    python run_case_study_2.py --all

    # Smoke test: first available particle image, E1 only
    python run_case_study_2.py --smoke

Service URLs (override via env or flags):
    SEM_SERVICE_URL  - default http://localhost:3000
    AGENT_API_URL    - default http://localhost:3001
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import shutil
import socket
import sys
import time
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request as _Req, urlopen

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Service endpoints (overridable via env / CLI flags)
# ---------------------------------------------------------------------------

SEM_URL   = os.environ.get("SEM_SERVICE_URL", "http://localhost:3000")
AGENT_URL = os.environ.get("AGENT_API_URL",   "http://localhost:3001")

# Retry/backoff defaults for long CS2 batches. These can be overridden by CLI
# flags or environment variables. The defaults intentionally make rate-limit
# failures much less likely when running all VLM variants over many samples.
HTTP_RETRIES = int(os.environ.get("CS2_HTTP_RETRIES", "4"))
HTTP_BACKOFF_SECONDS = float(os.environ.get("CS2_HTTP_BACKOFF", "5"))
HTTP_BACKOFF_MAX_SECONDS = float(os.environ.get("CS2_HTTP_BACKOFF_MAX", "120"))
AGENT_CALL_COOLDOWN_SECONDS = float(os.environ.get("CS2_AGENT_CALL_COOLDOWN", "8"))
_LAST_AGENT_CHAT_COMPLETED_AT = 0.0

_PROJECT_ROOT = Path(__file__).parent
GT_PATH       = _PROJECT_ROOT / "outputs" / "case_study_2" / "particle_gt.json"

# What to segment - shared subject injected into both ImgGen and SAM3 prompts.
# Override via --imggen-prompt / --sam3-prompt (or set both the same with one flag).
IMGGEN_SEGMENT_PROMPT = "particle"   # subject for ImgGen binary/colour mask
SAM3_TEXT_PROMPT      = "particle"   # subject for SAM3 text-prompted segmentation


# Datasets addressable through sem-service /api/dataset/load.
# sem-service expects unlabeled categories as "{split}/{subfolder}".
#
# Prompt strategy:
# - ImgGen and VLM get dataset-specific human-readable object wording.
# - SAM3 keeps compact candidate labels and falls back until it finds a usable mask.
DATASET_REGISTRY: dict[str, dict] = {
    "grid_scan_paper": {
        "description": "Grid Scan Paper unlabeled SEM segmentation SSL train set",
        "source": "unlabeled",
        "category": "train/Grid_Scan_Paper",
        "local_dir": (
            _PROJECT_ROOT
            / "Dataset_images_Unlabeled"
            / "dataset"
            / "sem_segmentation_ssl"
            / "train"
            / "Grid_Scan_Paper"
        ),
        "extensions": (".tif", ".tiff"),
        "prompts": {
            # ImgGen/VLM should describe the GridScan targets as clumps/deposits,
            # not clean round particles.
            "imggen_subject": "bright material clumps",
            "vlm_singular": "bright material clump",
            "vlm_plural": "bright material clumps",
            "count_instruction": (
                "Count each spatially separated bright material clump or deposit as one object. "
                "If a region is one fused clump with internal texture or attached fragments, count the whole clump as one object. "
                "Include partially visible clumps at the image edges."
            ),
            # Keep SAM3 prompts compact. Your observed run showed "clump" worked
            # after particle/agglomerate/deposit failed, so keep this ordered list.
            "sam3_subject": "clump",
            "sam3_subjects": ["particle", "particles", "agglomerate", "deposit", "clump"],
        },
    },
    "labeled_particles": {
        "description": "Original labeled SEM Particles dataset",
        "source": "labeled",
        "category": "Particles",
        "local_dir": (
            _PROJECT_ROOT
            / "Dataset_Images_Labeled_W_Metadata"
            / "Particles"
        ),
        "extensions": (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        "prompts": {
            "imggen_subject": "bright particles",
            "vlm_singular": "particle",
            "vlm_plural": "particles",
            "count_instruction": (
                "Count each distinct visible particle object, including partially visible ones at the edges. "
                "When particles touch but remain visually separable, count them separately."
            ),
            # This restores the old working SAM3 wording for the labeled particles set.
            "sam3_subject": "particle",
            "sam3_subjects": ["particle", "particles"],
        },
    },
    "validation_unc_luca": {
        "description": "Luca UNC validation set with high- and low-magnification SEM samples",
        "source": "validation_unc_luca",
        "category": "Merged_Samples",
        "category_candidates": (
            "Merged_Samples",
        ),
        "local_dir": (
            _PROJECT_ROOT
            / "Validation_Unc_Dataset_Images_Luca_CS2"
            / "Merged_Samples"
        ),
        "extensions": (".tif", ".tiff"),
        "prompts": {
            "imggen_subject": "bright material clumps",
            "vlm_singular": "bright material clump",
            "vlm_plural": "bright material clumps",
            "count_instruction": (
                "Count each spatially separated bright material clump or deposit as one object. "
                "If a region is one fused clump with internal texture or attached fragments, count the whole clump as one object. "
                "Include partially visible clumps at the image edges."
            ),
            "sam3_subject": "clump",
            "sam3_subjects": ["particle", "particles", "agglomerate", "deposit", "clump"],
        },
    },
    "validation_andrea_grid_scan_mirror": {
        "description": "Andrea mirrored grid-scan validation set, extracted SEM images only",
        "source": "validation_andrea",
        "category": "extracted_images",
        "category_candidates": (
            "extracted_images",
        ),
        "local_dir": (
            _PROJECT_ROOT
            / "Validation_Andrea_Dataset_Images_CS2"
            / "extracted_images"
        ),
        "extensions": (".tif", ".tiff"),
        "prompts": {
            "imggen_subject": "bright material clumps",
            "vlm_singular": "bright material clump",
            "vlm_plural": "bright material clumps",
            "count_instruction": (
                "Count each spatially separated bright material clump or deposit as one object. "
                "If a region is one fused clump with internal texture or attached fragments, count the whole clump as one object. "
                "Include partially visible clumps at the image edges."
            ),
            "sam3_subject": "clump",
            "sam3_subjects": ["particle", "particles", "agglomerate", "deposit", "clump"],
        },
    },
}


def _dataset_category_candidates(cfg: dict) -> tuple[str, ...]:
    """Return possible sem-service category names for a registered dataset."""
    candidates = cfg.get("category_candidates") or (cfg["category"],)
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = str(candidate).strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return tuple(out)


def _dataset_local_dir_candidates(cfg: dict) -> tuple[Path, ...]:
    """Return possible local directories for host-side preprocessing."""
    candidates = cfg.get("local_dir_candidates") or (cfg["local_dir"],)
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        p = Path(candidate)
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return tuple(out)


def _resolve_local_image_path(cfg: dict, filename: str) -> Path:
    """Return a local image path, preferring an existing file when possible."""
    candidates = [local_dir / filename for local_dir in _dataset_local_dir_candidates(cfg)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _prompt_config_for_sample(sample_spec: dict | None = None) -> dict:
    """Return dataset-specific object labels and segmentation prompts."""
    default = {
        "imggen_subject": IMGGEN_SEGMENT_PROMPT,
        "vlm_singular": "particle",
        "vlm_plural": "particles",
        "count_instruction": (
            "Count each distinct visible particle object, including partially visible ones at the edges."
        ),
        "sam3_subject": SAM3_TEXT_PROMPT,
        "sam3_subjects": [SAM3_TEXT_PROMPT, "particle"],
    }
    dataset_name = (sample_spec or {}).get("dataset_name")
    if dataset_name in DATASET_REGISTRY:
        merged = dict(default)
        merged.update(DATASET_REGISTRY[dataset_name].get("prompts", {}))
        return merged
    return default

# ---------------------------------------------------------------------------
# OpenAI config - loaded lazily so the script works without a .env for --list
# ---------------------------------------------------------------------------

def _load_openai_config() -> tuple[str, str, str]:
    """Return (api_key, base_url, model_name) from .env / environment."""
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key  = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    # IMGGEN_MODEL selects the image-edit model; defaults to gpt-image-2.
    model    = os.environ.get("IMGGEN_MODEL", "gpt-image-2")
    return api_key, base_url, model


# ---------------------------------------------------------------------------
# Experimental variant definitions
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "E1_raw_vlm": {
        "description":        "VLM counts directly from raw image (baseline, no segmentation)",
        "needs_segmentation": False,
        "implemented":        True,
    },
    "E2_sam2_deterministic": {
        "description":        "SAM2 colour mask -> shared deterministic mask count (no VLM)",
        "needs_segmentation": False,
        "implemented":        True,
    },
    "E3_imggen_deterministic": {
        "description":        "ImgGen hard unique-colour instance label mask -> shared deterministic mask count",
        "needs_segmentation": False,
        "implemented":        True,
    },
    "E4_sam2_overlay_vlm": {
        "description":        "SAM2 colour mask overlaid on image -> VLM counts annotated regions",
        "needs_segmentation": True,
        "implemented":        True,
    },
    "E5_imggen_overlay_vlm": {
        "description":        "ImgGen colour overlay composited on canvas -> VLM counts annotated regions",
        "needs_segmentation": False,
        "implemented":        True,
    },
    "E6_sam3_deterministic": {
        "description":        "SAM3 colour mask -> shared deterministic mask count (no VLM)",
        "needs_segmentation": False,
        "implemented":        True,
    },
    "E7_sam3_overlay_vlm": {
        "description":        "SAM3 colour overlay composited on canvas -> VLM counts annotated regions",
        "needs_segmentation": False,
        "implemented":        True,
    },
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_prompt(variant_name: str, sample_id: str, sample_spec: dict | None = None) -> str:
    """Return the agent chat prompt for a given variant.

    Dataset-specific object wording is used so the same experiment can target
    particles, clumps/deposits, fibres, or other SEM object types without changing
    the core variant logic.
    """
    pc = _prompt_config_for_sample(sample_spec)
    vlm_singular = pc["vlm_singular"]
    vlm_plural = pc["vlm_plural"]
    count_instruction = pc["count_instruction"]

    if variant_name == "E1_raw_vlm":
        return (
            "You are performing Case Study 2 - SEM Object Counting (E1: raw VLM baseline).\n\n"
            f"A SEM image has been loaded on the canvas (sample: {sample_id}).\n\n"
            f"Task: count all clearly visible {vlm_plural} in the image.\n\n"
            "Protocol:\n"
            "1. Call get_canvas_image to capture the current view of the SEM image.\n"
            "2. Call analyze_sandbox_image with the captured image path and the question:\n"
            f'   "How many {vlm_plural} are visible in this SEM image? '
            f"{count_instruction} "
            'Respond with only a single integer."\n'
            "3. State your final answer as a single integer on its own line, "
            "prefixed exactly with 'FINAL COUNT:' (e.g. 'FINAL COUNT: 17').\n\n"
            "IMPORTANT: Output a single integer after 'FINAL COUNT:' - no ranges, no uncertainty."
        )

    if variant_name == "E2_sam2_deterministic":
        return (
            "You are performing Case Study 2 - SEM Object Counting (E2: SAM2 deterministic).\n\n"
            f"A SEM image has been loaded on the canvas (sample: {sample_id}).\n\n"
            f"Task: use the SAM2 colour mask to count {vlm_plural} with the shared "
            "deterministic mask counter.\n\n"
            "Protocol:\n"
            "1. Call segment_viewport(centroids=False, bboxes=False, mask=True).\n"
            "2. Count the returned colour mask with the shared deterministic counter.\n"
            "3. State that count as a single integer on its own line, "
            "prefixed exactly with 'FINAL COUNT:' (e.g. 'FINAL COUNT: 17').\n\n"
            "IMPORTANT: Count the mask output with the same deterministic rule used for ImgGen and SAM3."
        )

    if variant_name == "E4_sam2_overlay_vlm":
        return (
            "You are performing Case Study 2 - SEM Object Counting (E4: SAM2 overlay + VLM).\n\n"
            f"A SEM image has been loaded on the canvas (sample: {sample_id}).\n\n"
            "Task: use SAM2 to annotate the image with a colour mask, "
            f"then use your VLM to count the annotated {vlm_plural}.\n\n"
            "Protocol:\n"
            "1. Call segment_viewport(centroids=False, bboxes=False, mask=True) to overlay "
            "coloured regions on the canvas.\n"
            "2. Call get_canvas_image to capture the annotated view.\n"
            "3. Call analyze_sandbox_image with the captured image path and the question:\n"
            f'   "How many distinct coloured regions are visible in this annotated SEM image? '
            f"Each coloured region represents one {vlm_singular}. "
            'Respond with only a single integer."\n'
            "4. State your final answer as a single integer on its own line, "
            "prefixed exactly with 'FINAL COUNT:' (e.g. 'FINAL COUNT: 17').\n\n"
            f"IMPORTANT: Count distinct coloured mask regions - one count per {vlm_singular}."
        )

    if variant_name == "E5_imggen_overlay_vlm":
        return (
            "You are performing Case Study 2 - SEM Object Counting (E5: ImgGen overlay + VLM).\n\n"
            f"A SEM image has been pre-annotated with an AI-generated colour segmentation overlay "
            f"and loaded onto the canvas (sample: {sample_id}).\n"
            f"Each {vlm_singular} has been highlighted with a distinct colour by the image generation model.\n\n"
            f"Task: count the number of distinct coloured {vlm_plural} in the annotated image.\n\n"
            "Protocol:\n"
            "1. Call get_canvas_image to capture the annotated view (the overlay is already applied).\n"
            "2. Call analyze_sandbox_image with the captured image path and the question:\n"
            f'   "How many distinct coloured regions are visible in this annotated SEM image? '
            f"Each coloured region represents one {vlm_singular}. "
            'Respond with only a single integer."\n'
            "3. State your final answer as a single integer on its own line, "
            "prefixed exactly with 'FINAL COUNT:' (e.g. 'FINAL COUNT: 17').\n\n"
            f"IMPORTANT: Count distinct coloured regions only - one region per {vlm_singular}. Do NOT call segment_viewport."
        )

    if variant_name == "E7_sam3_overlay_vlm":
        return (
            "You are performing Case Study 2 - SEM Object Counting (E7: SAM3 overlay + VLM).\n\n"
            f"A SEM image has been pre-annotated with a SAM3 text-prompted colour segmentation overlay "
            f"and loaded onto the canvas (sample: {sample_id}).\n"
            f"Each {vlm_singular} region detected by SAM3 has been highlighted with a distinct colour.\n\n"
            f"Task: count the number of distinct coloured {vlm_plural} in the annotated image.\n\n"
            "Protocol:\n"
            "1. Call get_canvas_image to capture the annotated view (the overlay is already applied).\n"
            "2. Call analyze_sandbox_image with the captured image path and the question:\n"
            f'   "How many distinct coloured regions are visible in this annotated SEM image? '
            f"Each coloured region represents one {vlm_singular} detected by SAM3. "
            'Respond with only a single integer."\n'
            "3. State your final answer as a single integer on its own line, "
            "prefixed exactly with 'FINAL COUNT:' (e.g. 'FINAL COUNT: 17').\n\n"
            f"IMPORTANT: Count distinct coloured regions only - one region per {vlm_singular}. Do NOT call segment_viewport."
        )

    raise ValueError(f"No prompt defined for implemented variant {variant_name!r}")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _parse_retry_after_seconds(value: str | None) -> float | None:
    """Parse a Retry-After header as seconds, if present and valid."""
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _retry_delay_seconds(attempt_no: int, *, retry_after: str | None = None) -> float:
    """Return exponential backoff with small jitter, respecting Retry-After."""
    header_delay = _parse_retry_after_seconds(retry_after)
    if header_delay is not None:
        return min(header_delay, HTTP_BACKOFF_MAX_SECONDS)

    base = HTTP_BACKOFF_SECONDS * (2 ** max(0, attempt_no - 1))
    jitter = random.uniform(0.0, min(1.0, HTTP_BACKOFF_SECONDS * 0.25))
    return min(base + jitter, HTTP_BACKOFF_MAX_SECONDS)


def _is_retryable_exception(exc: Exception) -> tuple[bool, int | None, str | None]:
    """Return (retryable, http_status, retry_after_header)."""
    if isinstance(exc, HTTPError):
        retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        return exc.code in retryable_statuses, exc.code, exc.headers.get("Retry-After")

    if isinstance(exc, TimeoutError) or isinstance(exc, socket.timeout):
        return True, None, None

    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError) or isinstance(reason, socket.timeout):
            return True, None, None
        text = str(reason or exc).lower()
        retryable_terms = ("timed out", "timeout", "temporarily unavailable", "connection reset")
        return any(term in text for term in retryable_terms), None, None

    return False, None, None


def _http(
    method: str,
    url: str,
    body: dict | None = None,
    timeout: int = 300,
    *,
    retries: int | None = None,
) -> dict:
    """JSON HTTP helper with retry/backoff for transient failures and 429s."""
    attempts = max(0, HTTP_RETRIES if retries is None else retries) + 1
    last_exc: Exception | None = None

    for attempt_idx in range(attempts):
        attempt_no = attempt_idx + 1
        data    = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        req     = _Req(url, data=data, headers=headers, method=method.upper())

        try:
            with urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except Exception as exc:
            last_exc = exc
            retryable, status, retry_after = _is_retryable_exception(exc)
            is_last_attempt = attempt_no >= attempts
            if not retryable or is_last_attempt:
                raise

            delay = _retry_delay_seconds(attempt_no, retry_after=retry_after)
            status_part = f" HTTP {status}" if status is not None else ""
            retry_after_part = f" Retry-After={retry_after!r}" if retry_after else ""
            print(
                f"  [warn] transient HTTP failure{status_part} on {method.upper()} {url}; "
                f"retry {attempt_no}/{attempts - 1} in {delay:.1f}s.{retry_after_part} error={exc}",
                file=sys.stderr,
            )
            time.sleep(delay)

    # Unreachable, but keeps type checkers and linters happy.
    assert last_exc is not None
    raise last_exc


def _sem(method: str, path: str, body: dict | None = None, timeout: int = 300) -> dict:
    return _http(method, f"{SEM_URL}{path}", body, timeout=timeout)


def _agent(method: str, path: str, body: dict | None = None, timeout: int = 900) -> dict:
    return _http(method, f"{AGENT_URL}{path}", body, timeout=timeout)


def _agent_chat_with_cooldown(prompt: str, timeout: int) -> dict:
    """Throttle /chat calls slightly so long all-variant runs do not hammer the agent API."""
    global _LAST_AGENT_CHAT_COMPLETED_AT

    if AGENT_CALL_COOLDOWN_SECONDS > 0 and _LAST_AGENT_CHAT_COMPLETED_AT > 0:
        elapsed = time.monotonic() - _LAST_AGENT_CHAT_COMPLETED_AT
        remaining = AGENT_CALL_COOLDOWN_SECONDS - elapsed
        if remaining > 0:
            print(f"    -> cooldown before agent /chat: {remaining:.1f}s")
            time.sleep(remaining)

    resp = _agent("POST", "/chat", {"message": prompt}, timeout=timeout)
    _LAST_AGENT_CHAT_COMPLETED_AT = time.monotonic()
    return resp


def _check_service(url: str, label: str) -> bool:
    endpoint = (
        f"{url.rstrip('/')}/api/session/stats"
        if "3000" in url
        else f"{url.rstrip('/')}/status"
    )
    try:
        _http("GET", endpoint, timeout=5)
        return True
    except Exception as exc:
        print(f"  [warn] {label} not reachable at {url}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Ground truth store
# ---------------------------------------------------------------------------

def load_gt_store(path: str | Path = GT_PATH) -> dict[str, dict]:
    """Load the ground truth store. Returns {} if the file is missing or empty."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Count parsing
# ---------------------------------------------------------------------------

def parse_count(reply: str) -> int | None:
    """Extract integer count from agent reply.

    Preference order:
      1. 'FINAL COUNT: N' - explicit anchor we asked the agent to use
      2. Last standalone integer in the reply (fallback)
    """
    m = re.search(r"FINAL\s+COUNT\s*[:=]\s*(\d+)", reply, re.IGNORECASE)
    if m:
        return int(m.group(1))
    all_ints = re.findall(r"\b(\d+)\b", reply)
    return int(all_ints[-1]) if all_ints else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _resolve_gt_interval(gt_entry: dict | None) -> dict:
    """Normalize point and interval GT entries into one internal representation."""
    if not gt_entry:
        return {
            "gt_count_mode": "missing",
            "gt_count": None,
            "gt_uncertain_count": None,
            "gt_interval": None,
        }

    mode = gt_entry.get("gt_count_mode", "point")

    # Backward-compatible scalar GT.
    if mode == "point":
        gt = gt_entry.get("gt_count")
        return {
            "gt_count_mode": "point",
            "gt_count": gt,
            "gt_uncertain_count": 0,
            "gt_interval": [gt, gt] if gt is not None else None,
        }

    if mode == "interval":
        confident = gt_entry.get("gt_count")
        uncertain = gt_entry.get("gt_uncertain_count", 0)

        if confident is None:
            return {
                "gt_count_mode": "interval",
                "gt_count": None,
                "gt_uncertain_count": uncertain,
                "gt_interval": None,
            }

        lower = int(confident)
        uncertain = int(uncertain)
        if uncertain < 0:
            raise ValueError(f"gt_uncertain_count must be >= 0, got {uncertain}")
        upper = lower + uncertain

        return {
            "gt_count_mode": "interval",
            "gt_count": lower,
            "gt_uncertain_count": uncertain,
            "gt_interval": [lower, upper],
        }

    raise ValueError(f"Unknown gt_count_mode: {mode!r}")


def _format_gt_for_display(gt_entry: dict | None) -> str:
    """Return a readable GT string for logs and CLI output."""
    gt_info = _resolve_gt_interval(gt_entry)
    mode = gt_info["gt_count_mode"]

    if mode == "missing":
        return "not annotated"

    if mode == "interval":
        interval = gt_info["gt_interval"]
        if interval is None:
            return "interval unavailable"
        return f"{interval[0]}-{interval[1]}"

    return str(gt_info["gt_count"])







def compute_metrics(predicted: int | None, gt_entry: dict | None) -> dict:
    """Compute prediction vs. point or interval ground truth metrics."""
    gt_info = _resolve_gt_interval(gt_entry)
    interval = gt_info["gt_interval"]

    base = {
        "comparable": False,
        "predicted_count": predicted,
        **gt_info,
    }

    if predicted is None or interval is None:
        return base

    lower, upper = interval

    if predicted < lower:
        signed_error = predicted - lower
        abs_error = lower - predicted
    elif predicted > upper:
        signed_error = predicted - upper
        abs_error = predicted - upper
    else:
        signed_error = 0
        abs_error = 0

    # Use interval midpoint for relative error denominator, or lower bound if preferred.
    # Midpoint is usually fairer for interval GT.
    gt_reference = max((lower + upper) / 2, 1)
    rel_error = abs_error / gt_reference

    return {
        **base,
        "comparable": True,
        "signed_error": signed_error,
        "absolute_error": abs_error,
        "relative_error": round(rel_error, 4),
        "inside_gt_interval": lower <= predicted <= upper,
        "exact_match": lower <= predicted <= upper if gt_info["gt_count_mode"] == "interval" else predicted == lower,
        "within_1": abs_error <= 1,
        "within_2": abs_error <= 2,
        "within_3": abs_error <= 3,
        "within_5_pct": rel_error <= 0.05,
        "within_10_pct": rel_error <= 0.10,
    }


def _json_safe(value):
    """Convert values such as Path/numpy scalars into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class _Tee:
    """Write stdout to multiple streams while preserving terminal output."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def _tee_stdout(log_path: Path):
    """Save exactly the script's stdout print output to log_path.

    This records only what the script prints; it does not record the CLI command
    unless the script explicitly prints that command.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        with redirect_stdout(_Tee(sys.stdout, f)):
            yield



def _iso_to_dt(iso: str) -> datetime | None:
    """Parse an ISO timestamp into a UTC datetime when possible."""
    if not iso:
        return None
    try:
        cleaned = iso.rstrip("Z")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _extract_model_name(trace_data: dict) -> str | None:
    """Best-effort model name extraction from agent trace payloads."""
    for key in ("model_name", "model", "llm_model"):
        value = trace_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = trace_data.get("metadata")
    if isinstance(metadata, dict):
        for key in ("model_name", "model", "llm_model"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _normalise_trace_steps(trace_data: dict) -> list[dict]:
    """Return trace steps in the structured format emitted by agent_api.py.

    The current agent API usually persists {steps: [{step, thinking, calls}]}.
    Older traces may use a flat {trace: [...]} shape; those are converted to a
    compatible step list so CS2 can write the same JSONL artifacts as CS1.
    """
    steps = trace_data.get("steps")
    if isinstance(steps, list):
        return [step for step in steps if isinstance(step, dict)]

    flat = trace_data.get("trace")
    if not isinstance(flat, list):
        return []

    converted: list[dict] = []
    for i, entry in enumerate(flat, 1):
        if not isinstance(entry, dict):
            continue
        typ = entry.get("type")
        if typ == "tool_call":
            converted.append({
                "type": "step",
                "step": i,
                "thinking": None,
                "calls": [{
                    "tool": entry.get("tool"),
                    "action": entry.get("action"),
                    "category": entry.get("category", "unknown"),
                    "input_summary": entry.get("input"),
                    "result": entry.get("result"),
                    "result_is_json": entry.get("result_is_json"),
                }],
            })
        else:
            converted.append({
                "type": "step",
                "step": i,
                "thinking": entry.get("content") or entry.get("thinking"),
                "calls": [],
            })
    return converted


def _select_trace_file(traces_dir: Path, completed_dt: datetime) -> Path | None:
    """Select the trace whose completion timestamp is closest to completed_dt."""
    candidates = sorted(traces_dir.glob("trace_*.json"))
    if not candidates:
        return None

    best: tuple[float, Path] | None = None
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        trace_dt = _iso_to_dt(data.get("completed_at", ""))
        if trace_dt is None:
            continue
        delta = abs((trace_dt - completed_dt).total_seconds())
        if best is None or delta < best[0]:
            best = (delta, path)

    return best[1] if best else candidates[-1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _rel_to_run(path: Path | None, run_dir: Path) -> str | None:
    """Return a run-relative path for manifests, falling back to absolute."""
    if path is None:
        return None
    try:
        return str(path.relative_to(run_dir))
    except Exception:
        return str(path)


def _extract_agent_trace_for_variant(
    *,
    variant_name: str,
    completed_dt: datetime,
    traces_dir: Path,
    run_dir: Path,
    reply: str | None = None,
) -> dict:
    """Copy and normalize the agent trace for one CS2 VLM-counting variant.

    Output layout mirrors the CS1 package_run.py artifacts while keeping each
    variant isolated:

        full_trace/<variant>/agent_trace.json
        full_trace/<variant>/model_messages.jsonl
        full_trace/<variant>/tool_calls.jsonl
        full_trace/<variant>/agent_timeline.jsonl
        full_trace/<variant>/agent_reply.txt
    """
    warnings: list[str] = []
    variant_trace_dir = run_dir / "full_trace" / variant_name
    variant_trace_dir.mkdir(parents=True, exist_ok=True)

    raw_trace_dst = variant_trace_dir / "agent_trace.json"
    model_msgs_dst = variant_trace_dir / "model_messages.jsonl"
    tool_calls_dst = variant_trace_dir / "tool_calls.jsonl"
    timeline_dst = variant_trace_dir / "agent_timeline.jsonl"
    agent_reply_dst = variant_trace_dir / "agent_reply.txt"

    if reply is not None:
        agent_reply_dst.write_text(reply, encoding="utf-8")

    trace_file: Path | None = None
    trace_data: dict | None = None
    model_name: str | None = None
    model_rows: list[dict] = []
    call_rows: list[dict] = []
    timeline_rows: list[dict] = []

    if not traces_dir.exists():
        warnings.append(f"trace_missing: traces directory does not exist: {traces_dir}")
    else:
        trace_file = _select_trace_file(traces_dir, completed_dt)
        if trace_file is None:
            warnings.append(f"trace_missing: no trace_*.json found in {traces_dir}")
        else:
            try:
                trace_data = json.loads(trace_file.read_text())
                model_name = _extract_model_name(trace_data)
                steps = _normalise_trace_steps(trace_data)

                trace_completed = _iso_to_dt(trace_data.get("completed_at", ""))
                if trace_completed and abs((trace_completed - completed_dt).total_seconds()) > 600:
                    warnings.append(f"trace_timestamp_mismatch: {trace_file.name} may not belong to {variant_name}")

                shutil.copy2(trace_file, raw_trace_dst)

                for step in steps:
                    step_no = step.get("step")
                    thinking = step.get("thinking")
                    if thinking:
                        model_rows.append({
                            "step": step_no,
                            "type": "thinking",
                            "content": thinking,
                        })
                        timeline_rows.append({
                            "step": step_no,
                            "event": "model_message",
                            "content": thinking,
                        })

                    for call in step.get("calls", []) or []:
                        call_row = {
                            "step": step_no,
                            "tool": call.get("tool"),
                            "action": call.get("action"),
                            "category": call.get("category"),
                            "input_summary": call.get("input_summary"),
                            "result": call.get("result"),
                            "result_is_json": call.get("result_is_json"),
                        }
                        call_rows.append(call_row)
                        timeline_rows.append({
                            "step": step_no,
                            "event": "tool_call",
                            "tool": call.get("tool"),
                            "action": call.get("action"),
                            "category": call.get("category"),
                            "input_summary": call.get("input_summary"),
                            "result": call.get("result"),
                        })

                _write_jsonl(model_msgs_dst, model_rows)
                _write_jsonl(tool_calls_dst, call_rows)
                _write_jsonl(timeline_dst, timeline_rows)

                if not model_rows:
                    warnings.append("model_messages_empty: trace contained no thinking/model text")
                if not call_rows:
                    warnings.append("tool_calls_empty: trace contained no tool calls")

                print(f"    -> trace saved: {trace_file.name}")
            except Exception as exc:
                warnings.append(f"trace_parse_error: {exc}")

    return {
        "variant": variant_name,
        "trace_file": str(trace_file) if trace_file else None,
        "model_name": model_name,
        "logs": {
            "raw_agent_trace": _rel_to_run(raw_trace_dst, run_dir) if raw_trace_dst.exists() else None,
            "model_messages": _rel_to_run(model_msgs_dst, run_dir) if model_msgs_dst.exists() else None,
            "tool_calls": _rel_to_run(tool_calls_dst, run_dir) if tool_calls_dst.exists() else None,
            "agent_reply": _rel_to_run(agent_reply_dst, run_dir) if agent_reply_dst.exists() else None,
            "service_events": _rel_to_run(timeline_dst, run_dir) if timeline_dst.exists() else None,
        },
        "n_model_messages": len(model_rows),
        "n_tool_calls": len(call_rows),
        "warnings": warnings,
    }

def _sha256_file(path: str | Path) -> str | None:
    """Return the SHA-256 hash of a file, or None if unavailable."""
    try:
        p = Path(path)
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _repro_manifest(
    *,
    sample_id: str,
    sample_spec: dict | None,
    variant_names: list[str],
) -> dict:
    """Build a reproducibility manifest for the run.

    It records the input image hash, dataset-specific prompts, selected model
    names, service URLs, and deterministic counter settings.
    """
    image_path = _image_path_for_sample(sample_id, sample_spec)
    prompt_cfg = _prompt_config_for_sample(sample_spec)
    _api_key, base_url, imggen_model = _load_openai_config()

    manifest = {
        "sample_id": sample_id,
        "sample_spec": _json_safe(sample_spec if sample_spec else _default_labeled_sample_spec(sample_id)),
        "input_image_path": str(image_path),
        "input_image_sha256": _sha256_file(image_path),
        "variants": list(variant_names),
        "service_urls": {
            "sem_service": SEM_URL,
            "agent_api": AGENT_URL,
        },
        "models": {
            "imggen_model": imggen_model,
            "imggen_base_url_set": bool(base_url),
            "sam2_endpoint": f"{AGENT_URL}/segment",
            "sam3_endpoint": f"{AGENT_URL}/segment-sam3",
        },
        "prompts": {
            "imggen_subject": prompt_cfg["imggen_subject"],
            "imggen_colour_prompt": _build_imggen_colour_prompt(prompt_cfg["imggen_subject"]),
            "vlm_singular": prompt_cfg["vlm_singular"],
            "vlm_plural": prompt_cfg["vlm_plural"],
            "count_instruction": prompt_cfg["count_instruction"],
            "sam3_subject": prompt_cfg["sam3_subject"],
            "sam3_subjects": prompt_cfg["sam3_subjects"],
            "sam3_prompt_candidates": _sam3_prompt_candidates(prompt_cfg),
        },
        "deterministic_counter": {
            "name": "foreground_islands_then_colour_split",
            "dark_threshold": 30,
            "colour_bin_size": 32,
            "min_colour_fraction": 0.10,
            "colour_merge_distance": 2.0,
            "foreground_connectivity": 8,
            "notes": (
                "Near-black pixels are background; foreground islands are counted "
                "unless an eroded island core contains multiple meaningful colours."
            ),
        },
        "image_preprocessing": {
            "sam2": "PIL opens the source, converts to RGB, re-encodes as real PNG data URL before POST /segment",
            "sam3": "PIL opens the source, converts to RGB, re-encodes as real PNG data URL before POST /segment-sam3",
            "imggen": "PIL opens the source, converts to RGB, resizes to 1024x1024 with LANCZOS, sends PNG to images.edit",
        },
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "numpy_version": np.__version__,
        },
    }
    return _json_safe(manifest)


# ---------------------------------------------------------------------------
# ImgGen helpers (used for E3 and E5)
# ---------------------------------------------------------------------------




def _build_imggen_colour_prompt(subject: str) -> str:
    """Build the hard unique-colour instance label-mask prompt for a subject."""
    return (
        f"Create a pure hard instance segmentation label mask for this SEM image. "
        f"Segment only: {subject}. "
        "Output a flat PNG mask, not an overlay and not a natural image. "
        "The background must be pure black RGB(0,0,0). "
        "Each detected object instance must be filled with one unique solid RGB colour. "
        "For a given object, every pixel must have exactly the same RGB value. "
        "Different objects must use clearly different RGB colours. "
        "All pixels must be fully opaque. "
        "Use no transparency, no alpha blending, no gradients, no shadows, no highlights, "
        "no texture, no blur, no antialiasing, no soft edges, no colour variation within an object, "
        "no outlines, no text, no labels, no borders, and no decorations. "
        "Preserve the object shapes and spatial layout exactly. "
        "Return only the mask image."
    )


def _generate_imggen_mask(
    image_path: str | Path,
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
) -> dict:
    """Call the OpenAI images.edit API (gpt-image-2) on a local image.

    The image is resized to fit within 1024x1024 before sending (gpt-image-2
    minimum supported size).  Returns the raw base64 PNG from the response.

    Returns:
        {"ok": True,  "mask_b64": "<base64 PNG string>"}
        {"ok": False, "error": "<message>"}
    """
    try:
        from openai import OpenAI  # imported here to keep top-level imports minimal
    except ImportError:
        return {"ok": False, "error": "openai package not installed"}

    try:
        src = Image.open(Path(image_path)).convert("RGB")
        src = src.resize((1024, 1024), Image.LANCZOS)
        buf = BytesIO()
        src.save(buf, "PNG")
        img_bytes = buf.getvalue()
    except Exception as exc:
        return {"ok": False, "error": f"Could not read/resize image: {exc}"}

    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(api_key=api_key, **kwargs)

    try:
        response = client.images.edit(
            model=model,
            image=("sem.png", img_bytes, "image/png"),
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
    except Exception as exc:
        return {"ok": False, "error": f"images.edit call failed: {exc}"}

    data = response.data
    if not data or not data[0].b64_json:
        return {"ok": False, "error": "No b64_json in images.edit response"}

    return {"ok": True, "mask_b64": data[0].b64_json}


def _imggen_cache_key(sample_id: str, subject: str, model: str) -> str:
    """Return a stable key for reusing an ImgGen colour mask within one run."""
    return f"{sample_id}::{subject}::{model}"


def _get_or_generate_imggen_colour_mask(
    *,
    sample_id: str,
    image_path: str | Path,
    api_key: str,
    base_url: str,
    model: str,
    cache: dict[str, str] | None = None,
    subject: str | None = None,
) -> dict:
    """Return an ImgGen hard unique-colour instance label mask, reusing it within a sample run.

    E3 and E5 should use the same generated colour mask whenever they are run
    in the same run_sample(...) call. This isolates the comparison to the
    counting method instead of mixing in stochastic differences between two
    separate ImgGen calls.

    Returns:
        {"ok": True, "mask_b64": "...", "prompt": "...", "cache_hit": bool}
        {"ok": False, "error": "...", "prompt": "...", "cache_hit": False}
    """
    subject = subject or IMGGEN_SEGMENT_PROMPT
    colour_prompt = _build_imggen_colour_prompt(subject)
    key = _imggen_cache_key(sample_id, subject, model)

    if cache is not None and key in cache:
        return {
            "ok": True,
            "mask_b64": cache[key],
            "prompt": colour_prompt,
            "cache_hit": True,
        }

    result = _generate_imggen_mask(image_path, colour_prompt, api_key, base_url, model)
    if not result.get("ok"):
        return {
            "ok": False,
            "error": result.get("error", "unknown ImgGen error"),
            "prompt": colour_prompt,
            "cache_hit": False,
        }

    mask_b64 = result["mask_b64"]
    if cache is not None:
        cache[key] = mask_b64

    return {
        "ok": True,
        "mask_b64": mask_b64,
        "prompt": colour_prompt,
        "cache_hit": False,
    }


def _decode_mask_image(mask_b64_or_data_url: str) -> Image.Image:
    """Decode a raw base64 image string or data URL into an RGB PIL image."""
    payload = mask_b64_or_data_url
    if payload.lstrip().startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")


def _count_instances_from_colour_mask_image(
    img: Image.Image,
    *,
    dark_threshold: int = 30,
    colour_bin_size: int = 32,
    min_colour_fraction: float = 0.10,
    colour_merge_distance: float = 2.0,
) -> int | None:
    """Count instances from a colour segmentation mask using one shared rule.

    Shared deterministic protocol for SAM2, ImgGen, and SAM3 masks:
      1. Treat near-black pixels as background.
      2. Find connected foreground islands first, independent of colour.
      3. Count each isolated foreground island as one instance unless its eroded
         colour core contains multiple meaningful colours.
      4. For mixed-colour islands, count the merged meaningful colour groups
         inside that island.

    This intentionally does not discard components based on absolute area,
    border contact, or relative image area. The one-pixel erosion and colour
    quantization are only used to avoid treating anti-aliased boundaries as
    extra instances.
    """
    try:
        import scipy.ndimage as ndi
    except ImportError:
        return None

    try:
        rgb = np.asarray(img.convert("RGB")).astype(np.int16)
        max_channel = rgb.max(axis=2)
        foreground = max_channel > dark_threshold

        if not foreground.any():
            return 0

        # Use 8-connectivity so diagonal contact is treated as touching.
        structure = np.ones((3, 3), dtype=bool)
        island_labels, n_islands = ndi.label(foreground, structure=structure)

        # Snap near-identical colours together to suppress antialiasing without
        # removing actual foreground components.
        quantized = (rgb // colour_bin_size).astype(np.int16)

        total_count = 0

        for island_id in range(1, n_islands + 1):
            island = island_labels == island_id
            island_area = int(island.sum())
            if island_area == 0:
                continue

            # Look for meaningful colours in the island interior rather than at
            # the boundary. This is the key step that prevents soft/anti-aliased
            # edges from becoming separate colour instances.
            core = ndi.binary_erosion(island, structure=structure, iterations=1)
            if not core.any():
                core = island

            core_colours = quantized[core].reshape(-1, 3)
            unique_colours, colour_counts = np.unique(
                core_colours,
                axis=0,
                return_counts=True,
            )

            min_pixels_for_colour = max(1, int(round(core_colours.shape[0] * min_colour_fraction)))
            meaningful = colour_counts >= min_pixels_for_colour
            meaningful_colours = unique_colours[meaningful]
            meaningful_counts = colour_counts[meaningful]

            # Merge neighbouring quantized colour bins. A single rendered object
            # can still straddle adjacent RGB bins, especially around purple or
            # orange colours, even when it is visually one solid region.
            if len(meaningful_colours) > 1:
                order = np.argsort(-meaningful_counts)
                merged_colours: list[np.ndarray] = []
                merged_weights: list[int] = []
                for idx in order:
                    colour = meaningful_colours[idx].astype(np.float64)
                    weight = int(meaningful_counts[idx])
                    matched = False
                    for j, centre in enumerate(merged_colours):
                        if float(np.linalg.norm(colour - centre)) <= colour_merge_distance:
                            total_weight = merged_weights[j] + weight
                            merged_colours[j] = ((centre * merged_weights[j]) + (colour * weight)) / total_weight
                            merged_weights[j] = total_weight
                            matched = True
                            break
                    if not matched:
                        merged_colours.append(colour)
                        merged_weights.append(weight)
                meaningful_colours = np.rint(np.vstack(merged_colours)).astype(np.int16)

            # One separated island with one meaningful interior colour is one
            # segment, regardless of boundary colour variation.
            if len(meaningful_colours) <= 1:
                total_count += 1
                continue

            # Mixed island: the island is one connected blob against the black
            # background, but it contains several meaningful interior colours.
            # Count those colour groups directly. This avoids over-splitting one
            # particle because of small disconnected edge fragments assigned to
            # the same colour.
            total_count += int(len(meaningful_colours))

        return int(total_count)

    except Exception:
        return None

def _count_from_colour_instance_mask_b64(
    mask_b64: str,
    *,
    dark_threshold: int = 30,
    colour_bin_size: int = 32,
    min_colour_fraction: float = 0.10,
    colour_merge_distance: float = 2.0,
) -> int | None:
    """Decode a base64/data-URL colour mask and apply the shared deterministic counter."""
    try:
        img = _decode_mask_image(mask_b64)
    except Exception:
        return None
    return _count_instances_from_colour_mask_image(
        img,
        dark_threshold=dark_threshold,
        colour_bin_size=colour_bin_size,
        min_colour_fraction=min_colour_fraction,
        colour_merge_distance=colour_merge_distance,
    )

def _rgba_data_url_from_b64(img_b64: str) -> str:
    """Convert a raw base64 PNG/JPEG to an RGBA data URL for canvas overlay push."""
    img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _run_sam3_segmentation(
    image_path: str | Path,
    text_prompt: str,
    return_overlay: bool = True,
) -> dict:
    """Call the SAM3 segmentation endpoint on the agent-api service.

    The endpoint is ``POST {AGENT_URL}/segment-sam3`` and runs SAM3
    (facebook/sam3) text-prompted instance segmentation inside the container.

    Parameters
    ----------
    image_path    : path to the JPG on the host filesystem.
    text_prompt   : text describing the objects to detect (e.g. ``'particle'``).
    return_overlay: if True, the endpoint also returns an RGBA composite overlay PNG.

    Returns
    -------
    dict with keys:
        ok, count, text_prompt, width, height
        composite_overlay  (data URL, only when return_overlay=True)
        instances          (list of {score, centroid, bbox})
        error              (only on failure)
    """
    try:
        # Always send a real PNG to the SAM3 service.  The previous code used
        # image/png for every non-JPEG file while still sending the original
        # bytes.  That works poorly for .tif/.tiff datasets because the data URL
        # says PNG but the payload is TIFF.  Re-encoding also normalizes grayscale
        # and TIFF images while preserving the old behaviour for JPG inputs.
        src = Image.open(Path(image_path)).convert("RGB")
        buf = BytesIO()
        src.save(buf, format="PNG")
        img_bytes = buf.getvalue()
    except Exception as exc:
        return {"ok": False, "error": f"Could not read/encode image for SAM3: {exc}"}

    img_b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode()

    body = {
        "image_b64":        img_b64,
        "text_prompt":      text_prompt,
        "return_overlay":   return_overlay,
        "return_instances": True,
    }
    try:
        return _agent("POST", "/segment-sam3", body, timeout=300)
    except Exception as exc:
        return {"ok": False, "error": f"/segment-sam3 call failed: {exc}"}


def _dedupe_prompts(prompts: list[str]) -> list[str]:
    """Return de-duplicated non-empty SAM3 prompts while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        cleaned = str(prompt).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def _sam3_prompt_candidates(prompt_cfg: dict) -> list[str]:
    """Return SAM3 prompt candidates, starting with the old working prompt.

    SAM3 appears to be very sensitive to phrasing.  The earlier working version
    used the compact prompt 'particle'.  Keep that behaviour as the first
    candidate, then try dataset-specific fallbacks only if the first one returns
    zero/no instances.
    """
    candidates: list[str] = []
    candidates.extend(prompt_cfg.get("sam3_subjects", []))
    candidates.append(prompt_cfg.get("sam3_subject", SAM3_TEXT_PROMPT))
    candidates.append(SAM3_TEXT_PROMPT)
    candidates.append("particle")
    return _dedupe_prompts(candidates)


def _run_sam3_segmentation_with_fallbacks(
    image_path: str | Path,
    prompt_cfg: dict,
    return_overlay: bool = True,
) -> dict:
    """Run SAM3 with short prompt fallbacks and select the first usable mask."""
    attempts: list[dict] = []
    best_ok: dict | None = None

    for text_prompt in _sam3_prompt_candidates(prompt_cfg):
        result = _run_sam3_segmentation(
            image_path,
            text_prompt,
            return_overlay=return_overlay,
        )
        count = result.get("count") if result.get("ok") else None
        has_overlay = bool(result.get("composite_overlay"))
        attempts.append({
            "text_prompt": text_prompt,
            "ok": bool(result.get("ok")),
            "count": count,
            "has_composite_overlay": has_overlay,
            "error": result.get("error"),
        })

        if not result.get("ok"):
            continue

        result["sam3_selected_prompt"] = text_prompt
        result["sam3_attempts"] = attempts

        # This is the path that used to work: choose the first prompt that
        # actually gives SAM3 instances.  For E7, an overlay is also required.
        if count and count > 0 and (has_overlay or not return_overlay):
            return result

        if best_ok is None:
            best_ok = result

    if best_ok is not None:
        best_ok["sam3_attempts"] = attempts
        best_ok.setdefault("sam3_selected_prompt", attempts[0]["text_prompt"] if attempts else SAM3_TEXT_PROMPT)
        return best_ok

    return {
        "ok": False,
        "error": "All SAM3 prompt attempts failed",
        "sam3_attempts": attempts,
        "sam3_selected_prompt": SAM3_TEXT_PROMPT,
    }


def _run_sam2_segmentation(image_path: str | Path) -> dict:
    """Call /segment on agent-api and return count + mask PNG in one shot.

    Always re-encode the source image as a real PNG before sending it. This
    avoids mislabeled data URLs for .tif/.tiff inputs, where raw TIFF bytes were
    previously sent with an image/png MIME type.

    Returns dict with keys: ok, count, mask_png, error.
    """
    try:
        src = Image.open(Path(image_path)).convert("RGB")
        buf = BytesIO()
        src.save(buf, format="PNG")
        img_bytes = buf.getvalue()
    except Exception as exc:
        return {"ok": False, "error": f"Could not read/encode image for SAM2: {exc}"}

    img_b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode()

    try:
        return _agent(
            "POST", "/segment",
            {"image_b64": img_b64, "centroids": False, "bboxes": False, "mask": True},
            timeout=120,
        )
    except Exception as exc:
        return {"ok": False, "error": f"/segment call failed: {exc}"}


# ---------------------------------------------------------------------------
# Dataset/sample resolution helpers
# ---------------------------------------------------------------------------

def _sample_id_from_filename(filename: str) -> str:
    """Return the experiment sample_id used for GT lookup and output folders."""
    return Path(filename).stem


def _default_labeled_sample_spec(sample_id: str) -> dict:
    """Return the original labeled Particles sample specification."""
    filename = f"{sample_id}.jpg"
    return {
        "sample_id": sample_id,
        "source": "labeled",
        "category": "Particles",
        "filename": filename,
        "local_path": (
            _PROJECT_ROOT
            / "Dataset_Images_Labeled_W_Metadata"
            / "Particles"
            / filename
        ),
        "dataset_name": "labeled_particles",
    }


def _resolve_dataset_sample_specs(
    dataset_name: str,
    n: int,
    gt_store: dict[str, dict],
    sample_id_filter: str | None = None,
) -> list[dict]:
    """Resolve images for a registered sem-service dataset.

    If sample_id_filter is provided, resolve that exact filename stem from the
    dataset. Otherwise, resolve the first n GT-annotated images.

    This intentionally uses sem-service /api/dataset/list to discover filenames,
    then later /api/dataset/load to put each exact image on the canvas. The local
    path is only used for host-side preprocessing calls such as SAM3 and ImgGen.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Available: {', '.join(DATASET_REGISTRY)}"
        )
    if n <= 0:
        raise ValueError("--n must be a positive integer")

    cfg = DATASET_REGISTRY[dataset_name]
    resp = _sem("GET", "/api/dataset/list")
    source = cfg["source"]
    categories = resp.get(source, [])
    category_candidates = _dataset_category_candidates(cfg)
    cat = next((c for c in categories if c.get("name") in category_candidates), None)
    if cat is None:
        available = ", ".join(c.get("name", "?") for c in categories) or "none"
        expected = ", ".join(repr(c) for c in category_candidates)
        raise RuntimeError(
            f"Dataset category for {dataset_name!r} not found in sem-service source "
            f"{source!r}. Tried: {expected}. Available categories: {available}"
        )
    resolved_category = cat.get("name", cfg["category"])

    images = cat.get("images", [])
    exts = tuple(e.lower() for e in cfg.get("extensions", ()))
    if exts:
        images = [
            img for img in images
            if Path(img.get("filename", "")).suffix.lower() in exts
        ]

    specs: list[dict] = []
    skipped_without_gt = 0
    for img in images:
        filename = img["filename"]
        sample_id = _sample_id_from_filename(filename)
        if sample_id_filter and sample_id != sample_id_filter:
            continue
        if sample_id not in gt_store:
            skipped_without_gt += 1
            continue
        specs.append({
            "sample_id": sample_id,
            "source": source,
            "category": resolved_category,
            "filename": filename,
            "local_path": _resolve_local_image_path(cfg, filename),
            "dataset_name": dataset_name,
            "dataset_description": cfg.get("description"),
        })
        if sample_id_filter or len(specs) >= n:
            break

    if not specs:
        if sample_id_filter:
            raise RuntimeError(
                f"Sample {sample_id_filter!r} was not found as a GT-annotated image "
                f"in dataset {dataset_name!r}. Check that the filename stem exists in "
                f"sem-service category {resolved_category!r} and in the GT store."
            )
        raise RuntimeError(
            f"No GT-annotated images found for dataset {dataset_name!r}. "
            f"Add entries to the GT store using filename stems such as "
            f"{_sample_id_from_filename(images[0]['filename']) if images else '<sample_id>'}."
        )

    if not sample_id_filter and len(specs) < n:
        print(
            f"  [warn] Requested n={n}, but only found {len(specs)} GT-annotated "
            f"image(s) in dataset {dataset_name!r}; skipped {skipped_without_gt} without GT.",
            file=sys.stderr,
        )

    return specs


# ---------------------------------------------------------------------------
# Canvas setup
# ---------------------------------------------------------------------------

def setup_canvas(
    sample_id: str,
    needs_segmentation: bool,
    sample_spec: dict | None = None,
) -> None:
    """Load the sample image through sem-service and configure the segmentation gate."""
    spec = sample_spec or _default_labeled_sample_spec(sample_id)
    resp = _sem("POST", "/api/dataset/load", {
        "source":   spec["source"],
        "category": spec["category"],
        "filename": spec["filename"],
    })
    if not resp.get("ok"):
        raise RuntimeError(
            f"Failed to load {spec['source']}:{spec['category']}/{spec['filename']}: "
            f"{resp.get('error', resp)}"
        )

    # Clear any segmentation overlay left from the prior variant run
    try:
        _sem("DELETE", "/api/canvas/segmentation")
    except Exception:
        pass  # non-fatal

    # Gate: must be enabled for E2/E4 or segment_viewport tool will refuse.
    # E3/E5/E6/E7 keep it disabled (no agent-side segment_viewport call needed).
    _sem("PUT", "/api/canvas/segmentation-enabled", {"enabled": needs_segmentation})


def _image_path_for_sample(
    sample_id: str,
    sample_spec: dict | None = None,
) -> Path:
    """Return the absolute local path to the sample image on the host."""
    spec = sample_spec or _default_labeled_sample_spec(sample_id)
    return Path(spec["local_path"])


# ---------------------------------------------------------------------------
# Image artifact helpers
# ---------------------------------------------------------------------------

def _save_original(image_path: Path, images_dir: Path) -> None:
    """Copy the source image into the run's images/ folder."""
    import shutil
    dest = images_dir / "original.jpg"
    if dest.exists():
        return
    if image_path.exists():
        shutil.copy2(image_path, dest)


def _composite_overlay_on_original(image_path: Path, overlay_img: Image.Image) -> Image.Image:
    """Alpha-composite an RGBA overlay PNG on top of the original RGB image.

    The overlay is resized to match the original image dimensions so the
    composite is always returned at the original image resolution.
    """
    orig = Image.open(image_path).convert("RGBA")
    if overlay_img.size != orig.size:
        overlay_img = overlay_img.resize(orig.size, Image.LANCZOS)
    return Image.alpha_composite(orig, overlay_img).convert("RGB")


def _save_mask_and_overlay(
    variant_name: str,
    images_dir: Path,
    image_path: Path,
    *,
    mask_b64: str | None = None,       # raw base64 PNG (no data-URL prefix)
    overlay_data_url: str | None = None,  # data URL of RGBA composite PNG
) -> None:
    """Save mask PNG and mask-composited-over-original PNG for a variant.

    Parameters
    ----------
    mask_b64 : raw base64 PNG used as the mask (binary or colour)
    overlay_data_url : RGBA data URL already composited by the model/SAM3
    """
    mask_path    = images_dir / f"{variant_name}_mask.png"
    overlay_path = images_dir / f"{variant_name}_overlay.png"

    mask_img: Image.Image | None = None

    if mask_b64:
        try:
            mask_img = Image.open(BytesIO(base64.b64decode(mask_b64)))
            mask_img.save(mask_path)
        except Exception:
            mask_img = None

    if overlay_data_url:
        try:
            # overlay_data_url is already composited (RGBA SAM3/ImgGen overlay)
            _hdr, _b64 = overlay_data_url.split(",", 1)
            overlay_rgba = Image.open(BytesIO(base64.b64decode(_b64))).convert("RGBA")
            if not mask_path.exists():
                overlay_rgba.convert("RGB").save(mask_path)  # save as mask too
            # composite over original
            if image_path.exists():
                composite = _composite_overlay_on_original(image_path, overlay_rgba)
                composite.save(overlay_path)
            return
        except Exception:
            pass

    # Fall back: composite mask_img over original
    if mask_img is not None and image_path.exists():
        try:
            mask_rgba = mask_img.convert("RGBA")
            # Make mask semi-transparent (alpha 160)
            r, g, b, a = mask_rgba.split()
            a = a.point(lambda px: min(px, 160))
            mask_rgba = Image.merge("RGBA", (r, g, b, a))
            composite = _composite_overlay_on_original(image_path, mask_rgba)
            composite.save(overlay_path)
        except Exception:
            pass


def _save_canvas_overlay(variant_name: str, images_dir: Path, image_path: Path) -> None:
    """Capture the current sem-service canvas as PNG and save as mask + overlay."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{SEM_URL}/api/export/png", timeout=15) as r:
            png_bytes = r.read()
        # Save the raw canvas render as the mask
        mask_path = images_dir / f"{variant_name}_mask.png"
        mask_path.write_bytes(png_bytes)
        # Composite: the canvas PNG already includes the image + overlay,
        # so just save it as the overlay image too.
        overlay_path = images_dir / f"{variant_name}_overlay.png"
        overlay_path.write_bytes(png_bytes)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Single-variant run
# ---------------------------------------------------------------------------

def run_variant(
    sample_id: str,
    variant_name: str,
    gt_entry: dict | None,
    *,
    agent_timeout: int = 900,
    dry_run: bool = False,
    skip_reset: bool = False,
    images_dir: Path | None = None,
    imggen_colour_cache: dict[str, str] | None = None,
    sample_spec: dict | None = None,
    traces_dir: Path | None = None,
    run_dir: Path | None = None,
) -> dict:
    """Run one experimental variant for one sample. Returns a prediction dict.

    Dispatch logic
    --------------
    E1, E4              : pure agent runs (no host-side pre-processing)
    E2 sam2_det         : SAM2 /segment -> shared deterministic colour-mask count (no agent)
    E3 imggen_det       : ImgGen colour instance mask -> shared deterministic colour-mask count (no agent)
    E5 imggen_overlay   : ImgGen colour overlay -> push to canvas -> agent VLM
    E6 sam3_det         : SAM3 text-prompted -> shared deterministic colour-mask count (no agent)
    E7 sam3_overlay     : SAM3 text-prompted -> colour overlay -> push to canvas -> agent VLM
    """
    v   = VARIANTS[variant_name]
    ts  = datetime.now(timezone.utc).isoformat()
    gt_count   = gt_entry["gt_count"] if gt_entry else None
    gt_display = _format_gt_for_display(gt_entry)
    image_path = _image_path_for_sample(sample_id, sample_spec)
    prompt_cfg = _prompt_config_for_sample(sample_spec)
    imggen_subject = prompt_cfg["imggen_subject"]

    print(f"\n  [{variant_name}]  {v['description']}")

    # Not-implemented variants - record without calling services
    if not v["implemented"]:
        print(f"    -> not_implemented (skipping)")
        return {
            "variant":         variant_name,
            "sample_id":       sample_id,
            "timestamp":       ts,
            "status":          "not_implemented",
            "reason":          v["description"],
            "predicted_count": None,
            "raw_reply":       None,
            "prompt":          None,
            "metrics":         compute_metrics(None, gt_entry),
        }

    # E2, E3, and E6 are fully deterministic - they never call the agent
    _is_deterministic = variant_name in ("E2_sam2_deterministic", "E3_imggen_deterministic", "E6_sam3_deterministic")

    if dry_run:
        if _is_deterministic:
            print(f"    -> [dry-run] deterministic variant - no agent prompt")
            if variant_name == "E2_sam2_deterministic":
                print(f"    -> would POST /segment mask=True and count the returned mask with the shared deterministic counter")
            elif variant_name == "E6_sam3_deterministic":
                print(f"    -> would POST /segment-sam3 text_prompt={SAM3_TEXT_PROMPT!r} and count the returned mask with the shared deterministic counter")
            else:
                print(
                        f"    -> would call gpt-image-2 images.edit for colour instance mask "
                        f"subject={imggen_subject!r}"
                    )
            return {
                "variant":         variant_name,
                "sample_id":       sample_id,
                "timestamp":       ts,
                "status":          "dry_run",
                "predicted_count": None,
                "raw_reply":       None,
                "prompt":          None,
                "metrics":         compute_metrics(None, gt_entry),
            }
        else:
            prompt = build_prompt(variant_name, sample_id, sample_spec)
            print(f"    -> [dry-run] prompt preview:")
            print("      " + prompt[:200].replace("\n", "\n      ") + "…")
            return {
                "variant":         variant_name,
                "sample_id":       sample_id,
                "timestamp":       ts,
                "status":          "dry_run",
                "predicted_count": None,
                "raw_reply":       None,
                "prompt":          prompt,
                "metrics":         compute_metrics(None, gt_entry),
            }

    # Setup canvas (all variants: loads image, clears any prior overlay)
    try:
        setup_canvas(sample_id, v["needs_segmentation"], sample_spec)
        time.sleep(0.5)  # brief pause for canvas state to propagate
    except Exception as exc:
        print(f"    -> [error] canvas setup failed: {exc}", file=sys.stderr)
        return {
            "variant":         variant_name,
            "sample_id":       sample_id,
            "timestamp":       ts,
            "status":          "setup_failed",
            "error":           str(exc),
            "predicted_count": None,
            "raw_reply":       None,
            "prompt":          None,
            "metrics":         compute_metrics(None, gt_entry),
        }

    # ----------------------------------------------------------------
    # E2: SAM2 -> shared deterministic colour-mask count  (no agent call)
    # ----------------------------------------------------------------
    if variant_name == "E2_sam2_deterministic":
        result = _run_sam2_segmentation(image_path)
        if not result.get("ok"):
            print(f"    -> [error] SAM2 failed: {result.get('error')}", file=sys.stderr)
            return {
                "variant":         variant_name,
                "sample_id":       sample_id,
                "timestamp":       ts,
                "status":          "sam2_failed",
                "error":           result.get("error"),
                "predicted_count": None,
                "raw_reply":       None,
                "prompt":          None,
                "metrics":         compute_metrics(None, gt_entry),
            }
        mask_png = result.get("mask_png")
        predicted = _count_from_colour_instance_mask_b64(mask_png) if mask_png else None
        status    = "completed" if predicted is not None else "parse_failed"
        if images_dir and mask_png:
            _save_mask_and_overlay(
                variant_name, images_dir, image_path,
                overlay_data_url=mask_png,
            )
        print(
            f"    -> predicted={predicted}  gt={gt_display}  status={status}  "
            f"(shared deterministic mask count; service_count={result.get('count')})"
        )
        return {
            "variant":         variant_name,
            "sample_id":       sample_id,
            "timestamp":       ts,
            "status":          status,
            "predicted_count": predicted,
            "raw_reply":       (
                f"[shared deterministic colour-mask count: {predicted}; "
                f"SAM2 service count: {result.get('count')}]"
            ),
            "prompt":          None,
            "deterministic_counter": "foreground_islands_then_colour_split",
            "sam2_service_count": result.get("count"),
            "metrics":         compute_metrics(predicted, gt_entry),
        }


    # ----------------------------------------------------------------
    # E3: ImgGen -> colour instance mask -> shared deterministic colour-mask count
    # ----------------------------------------------------------------
    if variant_name == "E3_imggen_deterministic":
        api_key, base_url, model = _load_openai_config()
        result = _get_or_generate_imggen_colour_mask(
            sample_id=sample_id,
            image_path=image_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            cache=imggen_colour_cache,
            subject=imggen_subject,
        )
        colour_prompt = result["prompt"]

        if not result.get("ok"):
            print(f"    -> [error] ImgGen failed: {result.get('error')}", file=sys.stderr)
            return {
                "variant":         variant_name,
                "sample_id":       sample_id,
                "timestamp":       ts,
                "status":          "imggen_failed",
                "error":           result.get("error"),
                "predicted_count": None,
                "raw_reply":       None,
                "prompt":          colour_prompt,
                "imggen_segment_prompt": imggen_subject,
                "metrics":         compute_metrics(None, gt_entry),
            }

        predicted = _count_from_colour_instance_mask_b64(result["mask_b64"])
        status    = "completed" if predicted is not None else "parse_failed"
        if images_dir:
            _save_mask_and_overlay(
                variant_name, images_dir, image_path, mask_b64=result["mask_b64"]
            )
        print(
            f"    -> predicted={predicted}  gt={gt_display}  status={status}  "
            f"(ImgGen colour mask cache_hit={result.get('cache_hit', False)})"
        )
        return {
            "variant":         variant_name,
            "sample_id":       sample_id,
            "timestamp":       ts,
            "status":          status,
            "predicted_count": predicted,
            "raw_reply":       "[shared deterministic colour-mask count from ImgGen colour instance mask]",
            "prompt":          colour_prompt,
            "imggen_segment_prompt": imggen_subject,
            "imggen_cache_hit": result.get("cache_hit", False),
            "deterministic_counter": "foreground_islands_then_colour_split",
            "metrics":         compute_metrics(predicted, gt_entry),
        }

    # ----------------------------------------------------------------
    # E5: ImgGen -> colour overlay -> push to canvas -> agent VLM
    # ----------------------------------------------------------------
    if variant_name == "E5_imggen_overlay_vlm":
        api_key, base_url, model = _load_openai_config()
        result = _get_or_generate_imggen_colour_mask(
            sample_id=sample_id,
            image_path=image_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            cache=imggen_colour_cache,
            subject=imggen_subject,
        )
        colour_prompt = result["prompt"]

        if not result.get("ok"):
            print(f"    -> [error] ImgGen colour overlay failed: {result.get('error')}", file=sys.stderr)
            return {
                "variant":              variant_name,
                "sample_id":            sample_id,
                "timestamp":            ts,
                "status":               "imggen_failed",
                "error":                result.get("error"),
                "predicted_count":      None,
                "raw_reply":            None,
                "prompt":               colour_prompt,
                "imggen_segment_prompt": imggen_subject,
                "imggen_cache_hit":      result.get("cache_hit", False),
                "metrics":              compute_metrics(None, gt_entry),
            }

        try:
            _sem("PUT", "/api/canvas/segmentation",
                 {"mask_png": _rgba_data_url_from_b64(result["mask_b64"])})
        except Exception as exc:
            print(f"    -> [warn] overlay push failed: {exc} - continuing without overlay",
                  file=sys.stderr)
        if images_dir:
            _save_mask_and_overlay(
                variant_name, images_dir, image_path, mask_b64=result["mask_b64"]
            )
        print(f"    -> ImgGen colour overlay ready  cache_hit={result.get('cache_hit', False)}")
        # Fall through to agent call below

    # ----------------------------------------------------------------
    # E6: SAM3 -> shared deterministic colour-mask count  (no agent call)
    # ----------------------------------------------------------------
    if variant_name == "E6_sam3_deterministic":
        result = _run_sam3_segmentation_with_fallbacks(
            image_path, prompt_cfg, return_overlay=True
        )
        if not result["ok"]:
            print(f"    -> [error] SAM3 failed: {result.get('error')}", file=sys.stderr)
            return {
                "variant":          variant_name,
                "sample_id":        sample_id,
                "timestamp":        ts,
                "status":           "sam3_failed",
                "error":            result.get("error"),
                "predicted_count":  None,
                "raw_reply":        None,
                "prompt":           result.get("sam3_selected_prompt", SAM3_TEXT_PROMPT),
                "sam3_text_prompt": result.get("sam3_selected_prompt", SAM3_TEXT_PROMPT),
                "sam3_attempts":    result.get("sam3_attempts", []),
                "metrics":          compute_metrics(None, gt_entry),
            }
        composite_overlay = result.get("composite_overlay")
        predicted = _count_from_colour_instance_mask_b64(composite_overlay) if composite_overlay else None
        status    = "completed" if predicted is not None else "parse_failed"
        if images_dir and composite_overlay:
            _save_mask_and_overlay(
                variant_name, images_dir, image_path,
                overlay_data_url=composite_overlay,
            )
        selected_sam3_prompt = result.get("sam3_selected_prompt", SAM3_TEXT_PROMPT)
        print(
            f"    -> predicted={predicted}  gt={gt_display}  status={status}  "
            f"(shared deterministic mask count; SAM3 prompt={selected_sam3_prompt!r}; "
            f"service_count={result.get('count')})"
        )
        return {
            "variant":          variant_name,
            "sample_id":        sample_id,
            "timestamp":        ts,
            "status":           status,
            "predicted_count":  predicted,
            "raw_reply":        (
                f"[shared deterministic colour-mask count: {predicted}; "
                f"SAM3 service count: {result.get('count')}]"
            ),
            "prompt":           selected_sam3_prompt,
            "sam3_text_prompt": selected_sam3_prompt,
            "sam3_attempts":    result.get("sam3_attempts", []),
            "deterministic_counter": "foreground_islands_then_colour_split",
            "sam3_service_count": result.get("count"),
            "metrics":          compute_metrics(predicted, gt_entry),
        }

    # ----------------------------------------------------------------
    # E7: SAM3 -> colour overlay -> push to canvas -> agent VLM
    # ----------------------------------------------------------------
    if variant_name == "E7_sam3_overlay_vlm":
        result = _run_sam3_segmentation_with_fallbacks(
            image_path, prompt_cfg, return_overlay=True
        )
        if not result["ok"]:
            print(f"    -> [error] SAM3 failed: {result.get('error')}", file=sys.stderr)
            return {
                "variant":          variant_name,
                "sample_id":        sample_id,
                "timestamp":        ts,
                "status":           "sam3_failed",
                "error":            result.get("error"),
                "predicted_count":  None,
                "raw_reply":        None,
                "prompt":           result.get("sam3_selected_prompt", SAM3_TEXT_PROMPT),
                "sam3_text_prompt": result.get("sam3_selected_prompt", SAM3_TEXT_PROMPT),
                "sam3_attempts":    result.get("sam3_attempts", []),
                "metrics":          compute_metrics(None, gt_entry),
            }
        if result.get("composite_overlay"):
            try:
                _sem("PUT", "/api/canvas/segmentation",
                     {"mask_png": result["composite_overlay"]})
            except Exception as exc:
                print(f"    -> [warn] overlay push failed: {exc} - continuing without overlay",
                      file=sys.stderr)
            if images_dir:
                _save_mask_and_overlay(
                    variant_name, images_dir, image_path,
                    overlay_data_url=result["composite_overlay"],
                )
        selected_sam3_prompt = result.get("sam3_selected_prompt", SAM3_TEXT_PROMPT)
        print(
            f"    -> SAM3 overlay ready  count={result.get('count')}  "
            f"prompt={selected_sam3_prompt!r}"
        )
        # Fall through to agent call below

    # ----------------------------------------------------------------
    # Agent call: E1, E2, E4 (direct), E5 (after overlay), E7 (after overlay)
    # ----------------------------------------------------------------
    if not skip_reset:
        try:
            _agent("POST", "/reset", timeout=15)
        except Exception:
            pass  # non-fatal - agent may still work

    prompt = build_prompt(variant_name, sample_id, sample_spec)
    try:
        resp = _agent_chat_with_cooldown(prompt, timeout=agent_timeout)
        agent_completed_dt = datetime.now(timezone.utc)
    except Exception as exc:
        print(f"    -> [error] agent call failed: {exc}", file=sys.stderr)
        return {
            "variant":         variant_name,
            "sample_id":       sample_id,
            "timestamp":       ts,
            "status":          "agent_failed",
            "error":           str(exc),
            "predicted_count": None,
            "raw_reply":       None,
            "prompt":          prompt,
            "metrics":         compute_metrics(None, gt_entry),
        }

    reply     = resp.get("reply", "")
    predicted = parse_count(reply)
    status    = "completed" if predicted is not None else "parse_failed"

    trace_info: dict | None = None
    if traces_dir is not None and run_dir is not None:
        trace_info = _extract_agent_trace_for_variant(
            variant_name=variant_name,
            completed_dt=agent_completed_dt,
            traces_dir=traces_dir,
            run_dir=run_dir,
            reply=reply,
        )

    # E4: capture the canvas (which has the SAM2 overlay from the agent)
    if images_dir and variant_name == "E4_sam2_overlay_vlm":
        _save_canvas_overlay(variant_name, images_dir, image_path)

    print(f"    -> predicted={predicted}  gt={gt_display}  status={status}")
    if predicted is None:
        print(f"    -> [warn] could not parse count from reply (last 200 chars): …{reply[-200:]}")

    result_out = {
        "variant":         variant_name,
        "sample_id":       sample_id,
        "timestamp":       ts,
        "status":          status,
        "predicted_count": predicted,
        "raw_reply":       reply,
        "prompt":          prompt,
        "metrics":         compute_metrics(predicted, gt_entry),
    }
    if trace_info is not None:
        result_out["trace"] = trace_info
        if trace_info.get("model_name"):
            result_out["model_name"] = trace_info["model_name"]
    if variant_name == "E5_imggen_overlay_vlm":
        result_out["imggen_segment_prompt"] = imggen_subject
        # Whether the overlay came from the per-sample ImgGen colour mask cache
        # is printed above; deterministic E3 records the same field in its JSON.
    if variant_name == "E7_sam3_overlay_vlm":
        result_out["sam3_text_prompt"] = selected_sam3_prompt
        result_out["sam3_attempts"] = result.get("sam3_attempts", [])
    return result_out


# ---------------------------------------------------------------------------
# Single sample orchestrator (all variants)
# ---------------------------------------------------------------------------

def run_sample(
    sample_id: str,
    variant_names: list[str],
    gt_entry: dict | None,
    *,
    out_dir: Path,
    agent_timeout: int,
    dry_run: bool,
    skip_reset: bool,
    inter_variant_delay: int = 2,
    sample_spec: dict | None = None,
    traces_dir: Path | None = None,
) -> dict:
    """Run all requested variants for one sample. Saves artifacts; returns manifest."""
    ts_start = datetime.now(timezone.utc)
    ts_str   = ts_start.strftime('%Y%m%d_%H%M%S')
    run_id   = f"{sample_id}_{ts_str}"
    # All runs for the same sample live under runs/{sample_id}/{timestamp}/
    run_dir  = out_dir / "runs" / sample_id / ts_str

    if dry_run:
        return _run_sample_inner(
            sample_id,
            variant_names,
            gt_entry,
            out_dir=out_dir,
            run_dir=run_dir,
            run_id=run_id,
            ts_start=ts_start,
            agent_timeout=agent_timeout,
            dry_run=dry_run,
            skip_reset=skip_reset,
            inter_variant_delay=inter_variant_delay,
            sample_spec=sample_spec,
            traces_dir=traces_dir,
        )

    log_path = run_dir / "logs" / "run_stdout.txt"
    with _tee_stdout(log_path):
        manifest = _run_sample_inner(
            sample_id,
            variant_names,
            gt_entry,
            out_dir=out_dir,
            run_dir=run_dir,
            run_id=run_id,
            ts_start=ts_start,
            agent_timeout=agent_timeout,
            dry_run=dry_run,
            skip_reset=skip_reset,
            inter_variant_delay=inter_variant_delay,
            sample_spec=sample_spec,
            traces_dir=traces_dir,
        )
        print(f"  Log      -> {log_path}")
        return manifest


def _run_sample_inner(
    sample_id: str,
    variant_names: list[str],
    gt_entry: dict | None,
    *,
    out_dir: Path,
    run_dir: Path,
    run_id: str,
    ts_start: datetime,
    agent_timeout: int,
    dry_run: bool,
    skip_reset: bool,
    inter_variant_delay: int = 2,
    sample_spec: dict | None = None,
    traces_dir: Path | None = None,
) -> dict:
    """Internal implementation for one sample run.

    The public run_sample wrapper adds stdout tee logging for non-dry runs.
    """
    print(f"\n{'='*70}")
    print(f"  Case Study 2 - sample : {sample_id}")
    print(f"  Run ID                : {run_id}")
    print(f"  Variants              : {', '.join(variant_names)}")
    print(f"  GT count              : {_format_gt_for_display(gt_entry)}")
    if sample_spec:
        print(
            f"  Dataset image         : "
            f"{sample_spec['source']}:{sample_spec['category']}/{sample_spec['filename']}"
        )
        pc = _prompt_config_for_sample(sample_spec)
        print(
            f"  Object prompt         : imggen={pc['imggen_subject']!r}  "
            f"vlm={pc['vlm_plural']!r}  sam3_candidates={pc['sam3_subjects']!r}"
        )
    print(f"{'='*70}")

    if not dry_run:
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        full_trace_dir = run_dir / "full_trace"
        full_trace_dir.mkdir(exist_ok=True)
        images_dir = run_dir / "images"
        images_dir.mkdir(exist_ok=True)
        # Save original image once for the whole run
        _save_original(_image_path_for_sample(sample_id, sample_spec), images_dir)
        repro = _repro_manifest(
            sample_id=sample_id,
            sample_spec=sample_spec,
            variant_names=variant_names,
        )
        (logs_dir / "repro_manifest.json").write_text(
            json.dumps(repro, indent=2)
        )
    else:
        images_dir = None

    effective_traces_dir = traces_dir or (_PROJECT_ROOT / "logs" / "traces")

    # Shared within this sample run so E3 and E5 use the exact same ImgGen mask.
    imggen_colour_cache: dict[str, str] = {}

    predictions: list[dict] = []
    for i, variant_name in enumerate(variant_names):
        pred = run_variant(
            sample_id, variant_name, gt_entry,
            agent_timeout=agent_timeout,
            dry_run=dry_run,
            skip_reset=skip_reset,
            images_dir=images_dir,
            imggen_colour_cache=imggen_colour_cache,
            sample_spec=sample_spec,
            traces_dir=effective_traces_dir if not dry_run else None,
            run_dir=run_dir if not dry_run else None,
        )
        predictions.append(pred)

        if not dry_run:
            pred_path = run_dir / "predictions" / f"{variant_name}.json"
            pred_path.write_text(json.dumps(pred, indent=2))

        if i < len(variant_names) - 1:
            time.sleep(inter_variant_delay)

    # Build metrics summary
    comparable = [
        p for p in predictions
        if p["metrics"].get("comparable")
    ]
    metrics_summary: dict = {
        "n_variants":    len(predictions),
        "n_implemented": sum(1 for p in predictions if p["status"] != "not_implemented"),
        "n_completed":   sum(1 for p in predictions if p["status"] == "completed"),
        "n_failed":      sum(
            1 for p in predictions
            if p["status"] in (
                "agent_failed", "setup_failed", "parse_failed",
                "imggen_failed", "sam3_failed",
            )
        ),
        "n_not_impl":    sum(1 for p in predictions if p["status"] == "not_implemented"),
        "per_variant":   {p["variant"]: p["metrics"] for p in predictions},
    }
    if comparable:
        abs_errors = [p["metrics"]["absolute_error"] for p in comparable]
        metrics_summary["mae"] = round(sum(abs_errors) / len(abs_errors), 4)

    if not dry_run:
        (run_dir / "metrics" / "per_variant_results.json").write_text(
            json.dumps(metrics_summary, indent=2)
        )

    gt_info = _resolve_gt_interval(gt_entry)

    manifest = {
        "run_id":           run_id,
        "case_study":       "case_study_2",
        "sample_id":        sample_id,
        "sample_source":    _json_safe(sample_spec if sample_spec else _default_labeled_sample_spec(sample_id)),
        "started_at":       ts_start.isoformat(),
        "completed_at":     datetime.now(timezone.utc).isoformat(),
        "status":           (
            "dry_run" if dry_run else
            "completed" if metrics_summary["n_failed"] == 0 else
            "completed_with_failures"
        ),
        "gt_count":         gt_info["gt_count"],
        "gt_count_mode":    gt_info["gt_count_mode"],
        "gt_uncertain_count": gt_info["gt_uncertain_count"],
        "gt_interval":      gt_info["gt_interval"],
        "gt_notes":         gt_entry.get("notes") if gt_entry else None,
        "variants_run":     variant_names,
        "predictions":      {p["variant"]: p["status"] for p in predictions},
        "logs": {
            "runner_console": "logs/run_stdout.txt" if not dry_run else None,
            "repro_manifest": "logs/repro_manifest.json" if not dry_run else None,
            "per_variant_traces": {
                p["variant"]: p.get("trace", {}).get("logs", {})
                for p in predictions
                if p.get("trace")
            },
        },
        "trace_warnings": {
            p["variant"]: p.get("trace", {}).get("warnings", [])
            for p in predictions
            if p.get("trace", {}).get("warnings")
        },
        "metrics_summary":  metrics_summary,
    }

    if not dry_run:
        (run_dir / "run_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2))
        print(f"\n  Run saved -> {run_dir}")
        print(f"  Images   -> {run_dir / 'images'}/")
        print(f"  Trace    -> {run_dir / 'full_trace'}/")
        print(f"  Repro    -> {run_dir / 'logs' / 'repro_manifest.json'}")

    return manifest

# ---------------------------------------------------------------------------
# Aggregate across completed runs
# ---------------------------------------------------------------------------

def _aggregate_metric_list(metric_rows: list[dict]) -> dict:
    """Compute aggregate stats for a list of comparable metric dicts."""
    abs_errors = [m["absolute_error"] for m in metric_rows]
    n = len(metric_rows)

    return {
        "n_samples":       n,
        "mae":             round(sum(abs_errors) / n, 4),
        "exact_match_pct": round(sum(m["exact_match"] for m in metric_rows) / n * 100, 1),
        "inside_gt_interval_pct": round(
            sum(m.get("inside_gt_interval", m.get("exact_match", False)) for m in metric_rows) / n * 100,
            1,
        ),
        "within_1_pct":    round(sum(m["within_1"]     for m in metric_rows) / n * 100, 1),
        "within_2_pct":    round(sum(m["within_2"]     for m in metric_rows) / n * 100, 1),
        "within_3_pct":    round(sum(m["within_3"]     for m in metric_rows) / n * 100, 1),
        "within_5p_pct":   round(sum(m["within_5_pct"] for m in metric_rows) / n * 100, 1),
        "within_10p_pct":  round(sum(m["within_10_pct"] for m in metric_rows) / n * 100, 1),
    }


def _dataset_name_from_manifest(manifest: dict) -> str:
    """Return a stable dataset label for grouping aggregate metrics."""
    sample_source = manifest.get("sample_source") or {}
    dataset_name = sample_source.get("dataset_name")
    if dataset_name:
        return dataset_name

    source = sample_source.get("source", "unknown_source")
    category = sample_source.get("category", "unknown_category")
    return f"{source}:{category}"


def aggregate_runs_by_dataset(
    out_dir: Path,
    *,
    latest_per_sample_variant: bool = True,
    completed_only: bool = True,
) -> dict[str, dict[str, dict]]:
    """Compute per-dataset, per-variant aggregate statistics.

    By default this uses only the latest completed comparable result for each
    (dataset, sample_id, variant). This prevents old reruns, repair runs, and
    failed historical runs from inflating n or mixing datasets.
    """
    runs_dir = out_dir / "runs"
    if not runs_dir.exists():
        return {}

    selected: dict[tuple[str, str, str], dict] = {}
    all_metrics_by_dataset_variant: dict[str, dict[str, list]] = {}

    for manifest_path in sorted(runs_dir.glob("*/*/run_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue

        dataset_name = _dataset_name_from_manifest(manifest)
        sample_id = manifest.get("sample_id")
        started_at = manifest.get("started_at") or manifest_path.parent.name
        run_dir = manifest_path.parent

        per_variant = manifest.get("metrics_summary", {}).get("per_variant", {})
        for variant_name, metrics in per_variant.items():
            if not metrics.get("comparable"):
                continue

            if completed_only:
                pred_path = run_dir / "predictions" / f"{variant_name}.json"
                if not pred_path.exists():
                    continue
                try:
                    pred = json.loads(pred_path.read_text())
                except Exception:
                    continue
                if pred.get("status") != "completed":
                    continue

            if latest_per_sample_variant:
                key = (dataset_name, sample_id, variant_name)
                old = selected.get(key)
                if old is None or started_at > old["started_at"]:
                    selected[key] = {
                        "dataset_name": dataset_name,
                        "sample_id": sample_id,
                        "variant_name": variant_name,
                        "started_at": started_at,
                        "metrics": metrics,
                    }
            else:
                all_metrics_by_dataset_variant.setdefault(dataset_name, {}).setdefault(
                    variant_name, []
                ).append(metrics)

    if latest_per_sample_variant:
        for row in selected.values():
            all_metrics_by_dataset_variant.setdefault(row["dataset_name"], {}).setdefault(
                row["variant_name"], []
            ).append(row["metrics"])

    summary: dict[str, dict[str, dict]] = {}
    for dataset_name, by_variant in all_metrics_by_dataset_variant.items():
        summary[dataset_name] = {}
        for variant_name, metric_rows in by_variant.items():
            if metric_rows:
                summary[dataset_name][variant_name] = _aggregate_metric_list(metric_rows)

    return summary


def aggregate_runs(out_dir: Path) -> dict[str, dict]:
    """Backward-compatible aggregate across all datasets.

    Prefer aggregate_runs_by_dataset(...) for reporting. This function keeps the
    old return shape by flattening the dataset-aware aggregate across all
    datasets, latest completed result per (dataset, sample_id, variant).
    """
    by_dataset = aggregate_runs_by_dataset(out_dir)
    all_rows: dict[str, list] = {}

    for _dataset_name, by_variant in by_dataset.items():
        for variant_name, stats in by_variant.items():
            # Cannot reconstruct exact metric rows from stats, so this function is
            # kept only as a placeholder for callers that expect the old name.
            # The main CLI now uses aggregate_runs_by_dataset directly.
            all_rows.setdefault(variant_name, []).append(stats)

    # Merge dataset-level stats approximately weighted by n_samples. This path is
    # not used by the CLI; it exists to avoid breaking imports of aggregate_runs.
    merged: dict[str, dict] = {}
    for variant_name, rows in all_rows.items():
        n_total = sum(r["n_samples"] for r in rows)
        if n_total <= 0:
            continue
        merged[variant_name] = {
            "n_samples": n_total,
            "mae": round(sum(r["mae"] * r["n_samples"] for r in rows) / n_total, 4),
            "exact_match_pct": round(sum(r["exact_match_pct"] * r["n_samples"] for r in rows) / n_total, 1),
            "inside_gt_interval_pct": round(sum(r["inside_gt_interval_pct"] * r["n_samples"] for r in rows) / n_total, 1),
            "within_1_pct": round(sum(r["within_1_pct"] * r["n_samples"] for r in rows) / n_total, 1),
            "within_2_pct": round(sum(r["within_2_pct"] * r["n_samples"] for r in rows) / n_total, 1),
            "within_3_pct": round(sum(r["within_3_pct"] * r["n_samples"] for r in rows) / n_total, 1),
            "within_5p_pct": round(sum(r["within_5p_pct"] * r["n_samples"] for r in rows) / n_total, 1),
            "within_10p_pct": round(sum(r["within_10p_pct"] * r["n_samples"] for r in rows) / n_total, 1),
        }
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Case Study 2 - SEM Particle Counting Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--sample-id",
        help="sample_id to run (filename stem, without extension)",
    )
    p.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY),
        help=(
            "Run images from a registered sem-service dataset. With --sample-id, runs that "
            "exact filename stem; otherwise runs the first --n GT-annotated images. "
            f"Available: {', '.join(DATASET_REGISTRY)}."
        ),
    )
    p.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of GT-annotated dataset images to run when --dataset is used (default: 1).",
    )
    p.add_argument(
        "--variant",
        default="E1_raw_vlm",
        choices=list(VARIANTS),
        help="Single variant to run (default: E1_raw_vlm)",
    )
    p.add_argument(
        "--all-variants",
        action="store_true",
        help="Run all variants for the specified sample",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all GT-annotated samples × all variants",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: first available particle image, E1 only",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List GT-annotated samples and variant definitions, then exit",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts without calling the agent or sem-service",
    )
    p.add_argument(
        "--out-dir",
        default=str(_PROJECT_ROOT / "outputs" / "case_study_2"),
        help="Output root directory (default: outputs/case_study_2)",
    )
    p.add_argument(
        "--traces-dir",
        default=str(_PROJECT_ROOT / "logs" / "traces"),
        help="Directory containing agent trace_*.json files (default: logs/traces).",
    )
    p.add_argument(
        "--gt-store",
        default=str(GT_PATH),
        help="Path to particle_gt.json",
    )
    p.add_argument(
        "--sem-url",
        default=SEM_URL,
        help="sem-service base URL (default: http://localhost:3000)",
    )
    p.add_argument(
        "--agent-url",
        default=AGENT_URL,
        help="agent-api base URL (default: http://localhost:3001)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Agent /chat timeout in seconds per attempt (default: 900 = 15 min).",
    )
    p.add_argument(
        "--http-retries",
        type=int,
        default=HTTP_RETRIES,
        help=(
            "Retries per HTTP call after transient errors such as 429, 5xx, or timeouts "
            "(default: %(default)s; env CS2_HTTP_RETRIES)."
        ),
    )
    p.add_argument(
        "--http-backoff",
        type=float,
        default=HTTP_BACKOFF_SECONDS,
        help=(
            "Initial retry backoff in seconds; doubles per retry unless Retry-After is provided "
            "(default: %(default)s; env CS2_HTTP_BACKOFF)."
        ),
    )
    p.add_argument(
        "--http-backoff-max",
        type=float,
        default=HTTP_BACKOFF_MAX_SECONDS,
        help=(
            "Maximum retry sleep in seconds (default: %(default)s; env CS2_HTTP_BACKOFF_MAX)."
        ),
    )
    p.add_argument(
        "--agent-call-cooldown",
        type=float,
        default=AGENT_CALL_COOLDOWN_SECONDS,
        help=(
            "Minimum seconds between agent /chat calls across VLM variants "
            "(default: %(default)s; env CS2_AGENT_CALL_COOLDOWN). Use 0 to disable."
        ),
    )
    p.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip POST /reset between variants (faster for debugging)",
    )
    p.add_argument(
        "--delay",
        type=int,
        default=2,
        help="Seconds to wait between variant runs (default: 2)",
    )
    p.add_argument(
        "--imggen-prompt",
        default=IMGGEN_SEGMENT_PROMPT,
        help=(
            "Subject injected into ImgGen segmentation prompts for E3/E5 (default: '%(default)s'). "
            "Describes what to segment, e.g. 'bright circular particle', 'fibre material'."
        ),
    )
    p.add_argument(
        "--sam3-prompt",
        default=SAM3_TEXT_PROMPT,
        help=(
            "Text prompt injected into SAM3 for E6/E7 (default: '%(default)s'). "
            "Describes what to segment, e.g. 'bright circular particle', 'fibre material'."
        ),
    )
    p.add_argument(
        "--sample",
        action="store_true",
        help=(
            "Pick a random labeled Particles image via POST /api/dataset/sample "
            "and use it as --sample-id. Useful for quick ad-hoc tests."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global SEM_URL, AGENT_URL, IMGGEN_SEGMENT_PROMPT, SAM3_TEXT_PROMPT
    global HTTP_RETRIES, HTTP_BACKOFF_SECONDS, HTTP_BACKOFF_MAX_SECONDS, AGENT_CALL_COOLDOWN_SECONDS

    args                 = _parse_args(argv)
    SEM_URL              = args.sem_url
    AGENT_URL            = args.agent_url
    IMGGEN_SEGMENT_PROMPT = args.imggen_prompt
    SAM3_TEXT_PROMPT     = args.sam3_prompt
    HTTP_RETRIES         = max(0, args.http_retries)
    HTTP_BACKOFF_SECONDS = max(0.0, args.http_backoff)
    HTTP_BACKOFF_MAX_SECONDS = max(0.0, args.http_backoff_max)
    AGENT_CALL_COOLDOWN_SECONDS = max(0.0, args.agent_call_cooldown)
    out_dir   = Path(args.out_dir)
    gt_store  = load_gt_store(args.gt_store)

    # ------------------------------------------------------------------
    # --list
    # ------------------------------------------------------------------
    if args.list:
        print("\nCase Study 2 - GT Store")
        print(f"  Path    : {args.gt_store}")
        print(f"  Samples : {len(gt_store)}")
        if gt_store:
            for sid, entry in sorted(gt_store.items()):
                gt_info = _resolve_gt_interval(entry)
                print(
                    f"    {sid:60s}  "
                    f"gt={_format_gt_for_display(entry)}  "
                    f"mode={gt_info['gt_count_mode']}  "
                    f"uncertain={gt_info['gt_uncertain_count']}  "
                    f"notes={entry.get('notes', '')}"
                )
        else:
            print("    (empty - add entries to data/case_study_2/particle_gt.json)")
        print(f"\nDatasets ({len(DATASET_REGISTRY)}):")
        for d_name, cfg in DATASET_REGISTRY.items():
            candidates = ", ".join(_dataset_category_candidates(cfg))
            print(
                f"  {d_name:40s}  source={cfg['source']:<9s}  "
                f"categories={candidates}"
            )
            print(f"    {cfg.get('description', '')}")
        print(f"\nVariants ({len(VARIANTS)}):")
        for v_name, cfg in VARIANTS.items():
            impl = "implemented" if cfg["implemented"] else "not_implemented"
            print(f"  {v_name:35s}  {impl}  - {cfg['description']}")
        return

    # ------------------------------------------------------------------
    # Pre-flight service checks (skipped in dry-run)
    # ------------------------------------------------------------------
    print("\nCase Study 2 - SEM Particle Counting Evaluation")
    print(f"  sem-service : {SEM_URL}")
    print(f"  agent-api   : {AGENT_URL}")

    if not args.dry_run:
        sem_ok   = _check_service(SEM_URL,   "sem-service")
        agent_ok = _check_service(AGENT_URL, "agent-api")
        if not sem_ok or not agent_ok:
            print("\nERROR: Required services not reachable. Aborting.", file=sys.stderr)
            sys.exit(1)
        print("  Services    : OK")
    else:
        print("  [dry-run mode - no services will be called]")

    # ------------------------------------------------------------------
    # Determine sample(s) and variant(s) to run
    # ------------------------------------------------------------------
    sample_ids:    list[str]
    sample_specs:  dict[str, dict] = {}
    variant_names: list[str]

    # --dataset: run n GT-annotated images from a registered sem-service dataset.
    if args.dataset:
        try:
            specs = _resolve_dataset_sample_specs(
                args.dataset,
                args.n,
                gt_store,
                sample_id_filter=args.sample_id,
            )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        sample_ids = [spec["sample_id"] for spec in specs]
        sample_specs = {spec["sample_id"]: spec for spec in specs}
        variant_names = list(VARIANTS.keys()) if args.all_variants else [args.variant]

    # --sample: pick a random labeled Particles image via /api/dataset/list
    elif getattr(args, "sample", False):
        import random as _random
        resp = _sem("GET", "/api/dataset/list")
        particles_cat = next(
            (c for c in resp.get("labeled", []) if c["name"] == "Particles"), None
        )
        if not particles_cat or not particles_cat["images"]:
            print("ERROR: No Particles images found in labeled dataset", file=sys.stderr)
            sys.exit(1)
        pick = _random.choice(particles_cat["images"])
        picked = Path(pick["filename"]).stem
        # Load it onto the canvas so the UI reflects the selection
        _sem("POST", "/api/dataset/load", {
            "source": "labeled", "category": "Particles", "filename": pick["filename"]
        })
        print(f"  [--sample] picked {picked}  (category=Particles, {len(particles_cat['images'])} candidates)")
        args.sample_id    = picked
        args.all_variants = True
        sample_specs[picked] = {
            "sample_id": picked,
            "source": "labeled",
            "category": "Particles",
            "filename": pick["filename"],
            "local_path": _default_labeled_sample_spec(picked)["local_path"],
            "dataset_name": "labeled_particles",
        }

    elif args.smoke:
        # Use the first GT-annotated sample if any; otherwise pick the first particle JPG
        annotated = [sid for sid, e in gt_store.items() if e.get("gt_count") is not None]
        if annotated:
            sample_ids = [annotated[0]]
        else:
            particles_dir = _PROJECT_ROOT / "Dataset_Images_Labeled_W_Metadata" / "Particles"
            jpegs = sorted(p.stem for p in particles_dir.glob("L2_*.jpg"))
            if not jpegs:
                print(
                    "ERROR: No particle images found in "
                    "Dataset_Images_Labeled_W_Metadata/Particles/",
                    file=sys.stderr,
                )
                sys.exit(1)
            sample_ids = [jpegs[0]]
            print(f"  [smoke] No GT annotations yet - using first image: {jpegs[0]}")
        variant_names = ["E1_raw_vlm"]

    elif args.all:
        annotated = sorted(sid for sid, e in gt_store.items() if e.get("gt_count") is not None)
        if not annotated:
            print(
                "ERROR: No GT-annotated samples in store. "
                "Add entries to data/case_study_2/particle_gt.json first.",
                file=sys.stderr,
            )
            sys.exit(1)
        sample_ids    = annotated
        variant_names = list(VARIANTS.keys())

    elif args.sample_id:
        sample_ids    = [args.sample_id]
        variant_names = list(VARIANTS.keys()) if args.all_variants else [args.variant]

    else:
        print(
            "ERROR: Specify --sample-id <id>, --all, --smoke, or --list.",
            file=sys.stderr,
        )
        sys.exit(1)

    for sid in sample_ids:
        sample_specs.setdefault(sid, _default_labeled_sample_spec(sid))

    print(f"  Samples     : {len(sample_ids)}")
    print(f"  Variants    : {variant_names}")
    if args.dataset:
        print(f"  Dataset     : {args.dataset}  n={len(sample_ids)}")
    print(f"  GT store    : {args.gt_store}  ({len(gt_store)} entries)")
    print(f"  Output      : {out_dir}")
    if not args.dry_run:
        print(f"  Traces dir  : {args.traces_dir}")
        print(
            f"  Retry cfg   : retries={HTTP_RETRIES}  backoff={HTTP_BACKOFF_SECONDS:g}s  "
            f"max_backoff={HTTP_BACKOFF_MAX_SECONDS:g}s  agent_cooldown={AGENT_CALL_COOLDOWN_SECONDS:g}s"
        )

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    all_manifests: list[dict] = []
    failed = 0

    for i, sample_id in enumerate(sample_ids):
        gt_entry = gt_store.get(sample_id)
        if gt_entry is None:
            print(f"\n  [warn] {sample_id} not in GT store - running without ground truth")

        manifest = run_sample(
            sample_id,
            variant_names,
            gt_entry,
            out_dir=out_dir,
            agent_timeout=args.timeout,
            dry_run=args.dry_run,
            skip_reset=args.skip_reset,
            inter_variant_delay=args.delay,
            sample_spec=sample_specs.get(sample_id),
            traces_dir=Path(args.traces_dir),
        )
        all_manifests.append(manifest)
        if manifest.get("status") not in ("completed", "dry_run"):
            failed += 1

        if i < len(sample_ids) - 1:
            time.sleep(args.delay)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  SUMMARY - {len(all_manifests)} sample(s),  {failed} failed")
    print(f"{'='*70}")

    if args.dry_run or not all_manifests:
        return

    # Per-sample quick stats
    for m in all_manifests:
        ms = m.get("metrics_summary", {})
        mae_str = f"  MAE={ms['mae']}" if "mae" in ms else ""
        print(
            f"  {m['sample_id']:55s}"
            f"  completed={ms.get('n_completed', '?')}/{ms.get('n_implemented', '?')}"
            f"{mae_str}"
        )

    # Aggregate (only meaningful with multiple samples and GT)
    if len(all_manifests) > 1:
        agg_by_dataset = aggregate_runs_by_dataset(out_dir)
        if agg_by_dataset:
            for dataset_name, agg in sorted(agg_by_dataset.items()):
                print(f"\n  Aggregate metrics - dataset={dataset_name}:")
                for v_name in VARIANTS:
                    stats = agg.get(v_name)
                    if not stats:
                        continue
                    print(
                        f"    {v_name:35s}"
                        f"  n={stats['n_samples']}"
                        f"  MAE={stats['mae']}"
                        f"  exact={stats['exact_match_pct']}%"
                        f"  inside_interval={stats.get('inside_gt_interval_pct', stats['exact_match_pct'])}%"
                        f"  within_1={stats['within_1_pct']}%"
                        f"  within_10%={stats['within_10p_pct']}%"
                    )
        else:
            agg_by_dataset = {}

        # Save multi-run summary
        ts            = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path  = out_dir / f"multi_run_summary_{ts}.json"
        summary_path.write_text(json.dumps(_json_safe({
            "generated_at": ts,
            "n_samples":    len(all_manifests),
            "n_failed":     failed,
            "variants":     variant_names,
            "aggregate_by_dataset": agg_by_dataset,
            "runs":         all_manifests,
        }), indent=2))
        print(f"\n  Multi-run summary -> {summary_path}")


if __name__ == "__main__":
    main()
