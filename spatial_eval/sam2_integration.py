"""SAM2 helpers for SpatialEval inference.

This module adds a SAM2 segmentation pre-pass before the GPT answer step.
It is task-aware for:
- mazenav
- spatialgrid
- spatialmap

The flow is:
1. Build a fixed grid canvas from the image.
2. Ask the vision model for JSON landmarks.
3. Scale landmarks back to original image space.
4. Run SAM2 point/box segmentation for the landmarks.
5. Build a composite image (original + overlay) and return metadata.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
CANVAS_SIZE = 1000
GRID_STEP = 100

TASK_PROMPTS = {
    "mazenav": """\
Look at this maze image carefully.
The image has been resized to exactly {cw} x {ch} pixels (width x height).
A grid is overlaid with lines every {step} pixels; axis labels show the pixel positions.
Use these grid lines to give accurate pixel coordinates.

Colour coding used in this maze:
- GREEN block  = START position
- RED block    = END / EXIT position
- BLUE blocks  = the solution path connecting start to end

Identify the following:
  1. The GREEN start block — label "start"
  2. The RED exit/end block — label "exit"
  3. For every straight section of the BLUE path (each uninterrupted horizontal or vertical run
     of blue blocks), place ONE point at the middle of that segment — label "segment_1", "segment_2", etc.
     A new segment begins every time the path changes direction.

Return ONLY a JSON array — no markdown, no prose — where each element has:
  "label"  : "start", "exit", or "segment_N"
  "x"      : pixel x-coordinate of the centre of the block (integer, 0 = left edge, {cw} = right edge)
  "y"      : pixel y-coordinate of the centre of the block (integer, 0 = top edge, {ch} = bottom edge)
  "box"    : [x_min, y_min, x_max, y_max] tight bounding box around just that block
             (use null if you cannot estimate a box)

Example: [{{"label": "start", "x": 120, "y": 340, "box": [100, 320, 140, 360]}}, ...]
""",
    "spatialgrid": """\
Look at this grid-based spatial reasoning image carefully.
The image has been resized to exactly {cw} x {ch} pixels (width x height).
A grid is overlaid with lines every {step} pixels; axis labels show the pixel positions.
Use these grid lines to give accurate pixel coordinates.

Your job is to identify the visually distinct landmarks or regions that are likely to matter for
answering a spatial question about this grid. Focus on colored blocks, occupied cells, marked
regions, boundaries, arrows, icons, and other clearly distinct salient features.

Identify up to 8 salient landmarks. For each landmark, place one point roughly at its centre and
(optionally) a tight box around it.

Return ONLY a JSON array — no markdown, no prose — where each element has:
  "label"  : "landmark_1", "landmark_2", ...
  "x"      : pixel x-coordinate of the centre of the landmark
  "y"      : pixel y-coordinate of the centre of the landmark
  "box"    : [x_min, y_min, x_max, y_max] tight bounding box around that landmark
             (use null if you cannot estimate a box)

Do not invent landmarks that are not visible. If the image is sparse, return fewer landmarks.
Example: [{{"label": "landmark_1", "x": 120, "y": 340, "box": [100, 320, 140, 360]}}, ...]
""",
    "spatialmap": """\
Look at this map-based spatial reasoning image carefully.
The image has been resized to exactly {cw} x {ch} pixels (width x height).
A grid is overlaid with lines every {step} pixels; axis labels show the pixel positions.
Use these grid lines to give accurate pixel coordinates.

Your job is to identify the salient map landmarks that are likely to matter for answering spatial
questions about this map. Focus on pins, markers, labels, symbols, region boundaries, route
endpoints, and other clearly distinct map features.

Identify up to 8 salient landmarks. For each landmark, place one point roughly at its centre and
(optionally) a tight box around it.

Return ONLY a JSON array — no markdown, no prose — where each element has:
  "label"  : "landmark_1", "landmark_2", ...
  "x"      : pixel x-coordinate of the centre of the landmark
  "y"      : pixel y-coordinate of the centre of the landmark
  "box"    : [x_min, y_min, x_max, y_max] tight bounding box around that landmark
             (use null if you cannot estimate a box)

Do not invent landmarks that are not visible. If the image is sparse, return fewer landmarks.
Example: [{{"label": "landmark_1", "x": 120, "y": 340, "box": [100, 320, 140, 360]}}, ...]
""",
}

QA_SYSTEM_TEMPLATE = """\
You are an expert at analysing {task_name} images.

You will be shown TWO images:
  1. The original image
  2. The same image with SAM2 segmentation overlays

SAM2 measured the following pixel centroids (x from left, y from top):
{centroid_info}

To answer spatial-relationship questions:
- Use the SAM2 centroid coordinates above as the ground-truth positions.
- "Directly above" = |cx(B) - cx(A)| ≤ 20 px and B has smaller y.
- "Directly to the left" = |cy(B) - cy(A)| ≤ 20 px and B has smaller x.
- Trust the numbers over visual appearance.

Think step by step, then end with your final answer on its own last line as ONLY
the letter and option text (e.g. 'C. 3').
"""

PALETTE = {
    "start": np.array([0, 220, 80], dtype=float),
    "exit": np.array([220, 30, 30], dtype=float),
    "path": np.array([30, 120, 220], dtype=float),
    "dead_end": np.array([200, 200, 200], dtype=float),
}


@dataclass
class Sam2TaskResult:
    task: str
    landmarks_canvas: List[dict]
    landmarks: List[dict]
    centroids: Dict[str, List[Tuple[int, int, float]]]
    boxes: Dict[str, List[Tuple[int, int, int, int, float]]]
    overlay_image: Image.Image
    composite_image: Image.Image


def load_sam2_predictor(
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str | None = None,
) -> tuple:
    """Load SAM2 predictor once for reuse across many items.

    Disables the broken cuDNN backend (CUDNN_STATUS_NOT_INITIALIZED on this
    system) and falls back to native CUDA algorithms which work fine.
    Returns (predictor, effective_device).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        # cuDNN is broken on this host; disable it so conv2d uses native CUDA.
        torch.backends.cudnn.enabled = False
        free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
        total_gb = torch.cuda.mem_get_info(0)[1] / 1e9
        print(f"[sam2] GPU: {torch.cuda.get_device_name(0)} — {free_gb:.1f}GB free / {total_gb:.1f}GB total (cuDNN disabled)")
    print(f"[sam2] Loading {sam2_model_id} on {device}...")
    t0 = time.perf_counter()
    try:
        predictor = SAM2ImagePredictor.from_pretrained(sam2_model_id, device=device)
        print(f"[sam2] Predictor ready on {device} ({time.perf_counter() - t0:.1f}s)")
        return predictor, device
    except (RuntimeError, Exception) as exc:
        if device != "cpu" and ("cuda" in device.lower() or "cuda" in str(exc).lower()):
            print(f"[sam2] CUDA load failed ({exc}), falling back to CPU...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            predictor = SAM2ImagePredictor.from_pretrained(sam2_model_id, device="cpu")
            print(f"[sam2] Predictor ready on CPU ({time.perf_counter() - t0:.1f}s)")
            return predictor, "cpu"
        raise


def _load_font(size: int = 14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_grid_canvas(image: Image.Image, canvas_size: int = CANVAS_SIZE, grid_step: int = GRID_STEP) -> Image.Image:
    canvas = image.resize((canvas_size, canvas_size), Image.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(14)
    line_colour = (180, 180, 180)
    label_colour = (255, 255, 0)
    for pos in range(0, canvas_size + 1, grid_step):
        draw.line([(pos, 0), (pos, canvas_size)], fill=line_colour, width=1)
        draw.text((pos + 2, 2), str(pos), fill=label_colour, font=font)
        draw.line([(0, pos), (canvas_size, pos)], fill=line_colour, width=1)
        draw.text((2, pos + 2), str(pos), fill=label_colour, font=font)
    return canvas


def scale_coords(val: int, canvas_dim: int, orig_dim: int) -> int:
    return int(round(val * orig_dim / canvas_dim))


def scale_landmark(lm: dict, cw: int, ch: int, orig_w: int, orig_h: int) -> dict:
    out = dict(lm)
    out["x"] = scale_coords(lm["x"], cw, orig_w)
    out["y"] = scale_coords(lm["y"], ch, orig_h)
    if lm.get("box"):
        x0, y0, x1, y1 = lm["box"]
        out["box"] = [
            scale_coords(x0, cw, orig_w),
            scale_coords(y0, ch, orig_h),
            scale_coords(x1, cw, orig_w),
            scale_coords(y1, ch, orig_h),
        ]
    return out


def parse_landmark_json(raw_text: str) -> List[dict]:
    json_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    data = json.loads(json_text)
    if not isinstance(data, list):
        raise ValueError("Expected landmark JSON to be a list")
    return data


def _task_prompt(task: str, cw: int, ch: int, grid_step: int) -> str:
    task_key = (task or "").lower()
    template = TASK_PROMPTS.get(task_key, TASK_PROMPTS["spatialgrid"])
    return template.format(cw=cw, ch=ch, step=grid_step)


def ask_model_for_landmarks(
    generate_fn: Callable[[str, Image.Image, float], Tuple[str, str]],
    image: Image.Image,
    *,
    task: str,
    temperature: float = 0.2,
    canvas_size: int = CANVAS_SIZE,
    grid_step: int = GRID_STEP,
) -> Tuple[List[dict], List[dict], Image.Image]:
    orig_w, orig_h = image.size
    canvas = make_grid_canvas(image, canvas_size=canvas_size, grid_step=grid_step)
    cw, ch = canvas.size
    prompt_text = _task_prompt(task, cw, ch, grid_step)
    _, raw = generate_fn(prompt_text, canvas, temperature)
    landmarks_canvas = parse_landmark_json(raw)
    landmarks = [scale_landmark(lm, cw, ch, orig_w, orig_h) for lm in landmarks_canvas]
    return landmarks_canvas, landmarks, canvas


def _build_side_by_side(original: Image.Image, overlay: Image.Image) -> Image.Image:
    original = original.convert("RGB")
    overlay = overlay.convert("RGB")
    gap = 12
    title_h = 28
    out_w = original.width + overlay.width + gap
    out_h = max(original.height, overlay.height) + title_h
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    canvas.paste(original, (0, title_h))
    canvas.paste(overlay, (original.width + gap, title_h))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(16)
    draw.rectangle((0, 0, original.width, 24), fill=(0, 0, 0))
    draw.text((6, 4), "Original", fill=(255, 255, 255), font=font)
    draw.rectangle((original.width + gap, 0, out_w, 24), fill=(0, 0, 0))
    draw.text((original.width + gap + 6, 4), "SAM2 overlay", fill=(255, 255, 255), font=font)
    return canvas


def _run_sam2_on_device(
    image_np: np.ndarray,
    landmarks: Sequence[dict],
    sam2_model_id: str,
    device: str,
    dtype,
    threshold: float,
    rng: random.Random,
    predictor=None,  # Optional pre-loaded predictor; loaded from HuggingFace if None
) -> Tuple[Dict[str, List[Tuple[int, int, float]]], Dict[str, List[Tuple[int, int, int, int, float]]], np.ndarray]:
    """Core SAM2 inference on a specific device. Separated to allow CUDA→CPU fallback."""
    if predictor is None:
        predictor = SAM2ImagePredictor.from_pretrained(sam2_model_id, device=device)
    overlay = image_np.copy().astype(float)
    centroids: Dict[str, List[Tuple[int, int, float]]] = {}
    boxes: Dict[str, List[Tuple[int, int, int, int, float]]] = {}

    with torch.inference_mode(), torch.autocast(device, dtype=dtype):
        predictor.set_image(image_np)
        for lm in landmarks:
            point_coords = np.array([[lm["x"], lm["y"]]])
            point_labels = np.array([1])
            box = np.array(lm["box"]) if lm.get("box") else None

            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

            best = int(np.argmax(scores))
            mask = masks[best].astype(bool)
            score = float(scores[best])
            if score < threshold:
                continue

            label = lm["label"]
            base_col = PALETTE.get(label, np.array([200, 200, 200], dtype=float))
            col = np.clip(base_col + rng.randint(-20, 20), 0, 255)
            overlay[mask] = overlay[mask] * 0.35 + col * 0.65

            ys, xs = np.where(mask)
            if ys.size:
                centroids.setdefault(label, []).append((int(xs.mean()), int(ys.mean()), round(score, 3)))
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            else:
                bbox = (0, 0, 0, 0)
            boxes.setdefault(label, []).append((*bbox, round(score, 3)))

    return centroids, boxes, overlay


def run_sam2_on_landmarks(
    image: Image.Image,
    landmarks: Sequence[dict],
    *,
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str | None = None,
    threshold: float = 0.4,
    predictor=None,        # Optional pre-loaded predictor (call load_sam2_predictor once)
    predictor_device: str | None = None,  # Device the predictor was loaded on
) -> Tuple[Dict[str, List[Tuple[int, int, float]]], Dict[str, List[Tuple[int, int, int, int, float]]], Image.Image]:
    # Use predictor's device if known; otherwise auto-detect
    if predictor is not None and predictor_device is not None:
        device = predictor_device
    else:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # bfloat16 triggers cuDNN which is broken on this host

    image_np = np.array(image.convert("RGB"))
    rng = random.Random(42)

    try:
        centroids, boxes, overlay = _run_sam2_on_device(
            image_np, landmarks, sam2_model_id, device, dtype, threshold, rng,
            predictor=predictor,
        )
    except (RuntimeError, Exception) as exc:
        if device != "cpu" and ("cuda" in device.lower() or "cudnn" in str(exc).lower() or "cuda" in str(exc).lower()):
            print(f"[sam2] CUDA inference failed ({type(exc).__name__}: {exc}), retrying on CPU...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            device = "cpu"
            dtype = torch.float32
            # Create a fresh CPU predictor for this fallback (cached predictor stays on CUDA)
            centroids, boxes, overlay = _run_sam2_on_device(
                image_np, landmarks, sam2_model_id, device, dtype, threshold, rng,
                predictor=None,
            )
        else:
            raise

    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    return centroids, boxes, overlay_img


def build_sam2_result(
    generate_fn: Callable[[str, Image.Image, float], Tuple[str, str]],
    image: Image.Image,
    *,
    task: str,
    temperature: float = 0.2,
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str | None = None,
    predictor=None,        # Optional pre-loaded predictor (from load_sam2_predictor)
    predictor_device: str | None = None,
) -> Sam2TaskResult:
    t_lm = time.perf_counter()
    landmarks_canvas, landmarks, _ = ask_model_for_landmarks(
        generate_fn,
        image,
        task=task,
        temperature=temperature,
    )
    print(f"  [sam2] landmark detection: {time.perf_counter() - t_lm:.1f}s  ({len(landmarks)} landmarks)")

    t_seg = time.perf_counter()
    centroids, boxes, overlay_img = run_sam2_on_landmarks(
        image,
        landmarks,
        sam2_model_id=sam2_model_id,
        device=device,
        predictor=predictor,
        predictor_device=predictor_device,
    )
    print(f"  [sam2] segmentation:       {time.perf_counter() - t_seg:.1f}s")

    composite = _build_side_by_side(image, overlay_img)
    return Sam2TaskResult(
        task=task,
        landmarks_canvas=landmarks_canvas,
        landmarks=landmarks,
        centroids=centroids,
        boxes=boxes,
        overlay_image=overlay_img,
        composite_image=composite,
    )


def _dedup_masks(
    centroids: Dict[str, List[Tuple[int, int, float]]],
    boxes: Dict[str, List[Tuple[int, int, int, int, float]]],
    min_dist: int = 30,
) -> Tuple[Dict, Dict]:
    """Remove masks whose centroid is within min_dist px of a higher-scored mask."""
    all_masks = [
        (score, label, i, cx, cy)
        for label, instances in centroids.items()
        for i, (cx, cy, score) in enumerate(instances)
    ]
    all_masks.sort(reverse=True)  # highest score first
    kept: list = []
    for score, label, i, cx, cy in all_masks:
        if not any(((cx - kx) ** 2 + (cy - ky) ** 2) < min_dist ** 2
                   for _, _, _, kx, ky in kept):
            kept.append((score, label, i, cx, cy))
    kept_set = {(lbl, idx) for _, lbl, idx, _, _ in kept}
    new_c: Dict[str, List] = {}
    new_b: Dict[str, List] = {}
    for label, instances in centroids.items():
        for i, entry in enumerate(instances):
            if (label, i) in kept_set:
                new_c.setdefault(label, []).append(entry)
    for label, instances in boxes.items():
        for i, entry in enumerate(instances):
            if (label, i) in kept_set:
                new_b.setdefault(label, []).append(entry)
    return new_c, new_b


def build_sam2_result_basic(
    image: Image.Image,
    *,
    task: str,
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str | None = None,
    predictor=None,
    predictor_device: str | None = None,
    grid_n: int = 4,
    threshold: float = 0.4,
) -> Sam2TaskResult:
    """SAM2 segmentation using a fixed N\u00d7N grid of prompt points.

    No landmark-detection API call — SAM2 is run undirected over a uniform
    grid of points evenly spaced across the image.  Fast and model-agnostic.
    """
    w, h = image.size
    step_x = w / (grid_n + 1)
    step_y = h / (grid_n + 1)
    landmarks = [
        {"label": f"r{i}_c{j}", "x": int(step_x * j), "y": int(step_y * i), "box": None}
        for i in range(1, grid_n + 1)
        for j in range(1, grid_n + 1)
    ]

    eff_device = predictor_device or device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"  [sam2-basic] model={sam2_model_id}  device={eff_device}  "
        f"grid={grid_n}\u00d7{grid_n} ({len(landmarks)} pts)  threshold={threshold}"
    )

    t0 = time.perf_counter()
    centroids, boxes, overlay_img = run_sam2_on_landmarks(
        image,
        landmarks,
        sam2_model_id=sam2_model_id,
        device=device,
        predictor=predictor,
        predictor_device=predictor_device,
        threshold=threshold,
    )
    centroids, boxes = _dedup_masks(centroids, boxes)
    n_masks = sum(len(v) for v in centroids.values())
    print(f"  [sam2-basic] segmentation: {time.perf_counter() - t0:.1f}s  ({n_masks} unique masks above threshold)")

    composite = _build_side_by_side(image, overlay_img)
    return Sam2TaskResult(
        task=task,
        landmarks_canvas=landmarks,
        landmarks=landmarks,
        centroids=centroids,
        boxes=boxes,
        overlay_image=overlay_img,
        composite_image=composite,
    )


# Backwards-compatible alias.
def build_sam2_maze_result(
    generate_fn: Callable[[str, Image.Image, float], Tuple[str, str]],
    image: Image.Image,
    *,
    task: str = "mazenav",
    temperature: float = 0.2,
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID,
    device: str | None = None,
) -> Sam2TaskResult:
    return build_sam2_result(
        generate_fn,
        image,
        task=task,
        temperature=temperature,
        sam2_model_id=sam2_model_id,
        device=device,
    )


def centroids_to_text(centroids: Dict[str, List[Tuple[int, int, float]]]) -> str:
    lines: List[str] = []
    for label, instances in centroids.items():
        for i, (cx, cy, score) in enumerate(instances):
            suffix = f"_{i}" if len(instances) > 1 else ""
            lines.append(f"  - {label}{suffix}: centroid = ({cx}, {cy}) px  score={score}")
    return "\n".join(lines) if lines else "  (no centroids found)"


def boxes_to_text(boxes: Dict[str, List[Tuple[int, int, int, int, float]]]) -> str:
    lines: List[str] = []
    for label, instances in boxes.items():
        for i, (x0, y0, x1, y1, score) in enumerate(instances):
            suffix = f"_{i}" if len(instances) > 1 else ""
            lines.append(f"  - {label}{suffix}: box = ({x0},{y0})→({x1},{y1})  score={score}")
    return "\n".join(lines) if lines else "  (no boxes found)"
