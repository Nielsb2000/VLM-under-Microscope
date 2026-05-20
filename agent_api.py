"""agent_api.py — FastAPI wrapper around the DeepAgent (main.py) loop.

Start automatically via docker-compose (root docker-compose.yml).
Manually: uv run uvicorn agent_api:app --host 0.0.0.0 --port 3001

Endpoints:
  GET  /status       → {"running": bool}
  POST /chat         → {"message": "..."} → {"reply": str, "trace": [...]}
  POST /reset        → wipes agent memory (new conversation)
  POST /stop         → cancels the currently running chat call

Trace entry schema:
  { "type": "thinking" | "tool_call" | "tool_result" | "reply",
    "content": str,
    "tool": str | None,    # tool name for tool_call / tool_result
    "input": str | None }  # tool input snippet for tool_call
"""

import asyncio
import base64
import io
import logging
import random
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm_client import get_default_llm

logger = logging.getLogger("agent_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_agent = None
_current_chat_future: Optional[asyncio.Future] = None


def _get_agent():
    global _agent
    if _agent is None:
        logger.info("Initialising agent…")
        _agent = get_default_llm()
        logger.info("Agent ready")
    return _agent


def _extract_trace(messages: list) -> list[dict]:
    """Convert LangChain message list → step objects for the UI.

    Each step corresponds to one AIMessage and contains:
      - thinking: the model's reasoning text (may be None)
      - calls: list of tool calls, each with its matched result
    ToolMessages are matched to their AIMessage tool_call via tool_call_id.
    """
    # Build lookup: tool_call_id → raw result text (first pass)
    results_by_id: dict[str, str] = {}
    for msg in messages:
        if type(msg).__name__ == "ToolMessage":
            tcid = getattr(msg, "tool_call_id", None)
            if tcid:
                raw = getattr(msg, "content", "") or ""
                if isinstance(raw, list):
                    raw = "\n".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in raw
                    )
                results_by_id[tcid] = raw.strip()

    steps = []
    step_n = 0
    for msg in messages:
        if type(msg).__name__ != "AIMessage":
            continue

        # Thinking text (the content of the AIMessage itself)
        text = getattr(msg, "content", "") or ""
        if isinstance(text, list):
            text = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in text
            )
        thinking = text.strip() or None

        tool_calls = getattr(msg, "tool_calls", []) or []
        if not thinking and not tool_calls:
            continue

        step_n += 1
        calls = []
        for tc in tool_calls:
            name   = tc.get("name", "unknown")
            args   = tc.get("args", {}) or {}
            tcid   = tc.get("id", "")
            action = args.get("action") if name == "paint_canvas" else None

            raw_result = results_by_id.get(tcid, "")
            result_is_json = False
            if raw_result:
                try:
                    import json as _j
                    _j.loads(raw_result)
                    result_is_json = True
                except Exception:
                    pass

            calls.append({
                "tool":           name,
                "action":         action,
                "category":       _classify_tool(name, action),
                "input_summary":  _summarise_args(name, args),
                "result":         (raw_result[:800] + "…") if len(raw_result) > 800 else raw_result,
                "result_is_json": result_is_json,
            })

        steps.append({
            "type":     "step",
            "step":     step_n,
            "thinking": thinking,
            "calls":    calls,
        })

    return steps


# ---- Tool category classification ----

_PAINT_DRAW   = {"rect", "ellipse", "arrow", "dot", "line", "text", "bulk"}
_PAINT_ADJUST = {
    "set_filters", "zoom", "pan", "zoom_to_region", "zoom_to_object",
    "reset_viewport", "cursor_move", "cursor_hide", "viewport",
}
_PAINT_NAV    = {
    "load_tile_grid", "camera_left", "camera_right", "camera_up", "camera_down",
    "camera_move", "camera_goto", "camera_state", "set_canvas_mode",
}
_PAINT_IMAGE  = {
    "load_image", "load_image_by_name", "get_canvas_image",
    "export_png", "export_json", "crop", "list_images",
}
_EXEC_TOOLS   = {"execute", "read_file", "write_file", "glob", "grep", "bash", "run_command"}


def _classify_tool(name: str, action: str | None) -> str:
    if name == "paint_canvas":
        if action in _PAINT_DRAW:   return "draw"
        if action in _PAINT_ADJUST: return "adjust"
        if action in _PAINT_NAV:    return "navigate"
        if action in _PAINT_IMAGE:  return "image"
        return "meta"
    if name in ("analyze_sandbox_image", "screenshot_and_ask", "move_and_verify", "segment_viewport"):
        return "vision"
    if name == "get_sem_status":
        return "meta"
    if name in _EXEC_TOOLS:
        return "exec"
    return "meta"


def _summarise_args(name: str, args: dict) -> str:
    """Return a short one-line summary of tool call arguments."""
    if not args:
        return ""
    # For paint_canvas, omit 'action' (shown as the call label) and unpack params
    if name == "paint_canvas":
        params = args.get("params", {}) or {}
        if params and isinstance(params, dict):
            parts = []
            for k, v in params.items():
                s = str(v)
                if len(s) > 80:
                    s = s[:80] + "…"
                parts.append(f"{k}={s}")
            return ", ".join(parts)
    parts = []
    for k, v in args.items():
        if k == "action" and name == "paint_canvas":
            continue
        s = str(v)
        if len(s) > 100:
            s = s[:100] + "…"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _get_agent()
    yield


app = FastAPI(title="DeepAgent API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/status")
def status():
    return {"running": _agent is not None}


@app.post("/chat")
async def chat(req: ChatRequest):
    global _current_chat_future
    if _current_chat_future is not None and not _current_chat_future.done():
        return JSONResponse(status_code=429, content={"error": "Agent is already processing a request. Use /stop to cancel it."})

    loop = asyncio.get_event_loop()
    _current_chat_future = loop.run_in_executor(None, _do_chat, req.message)
    try:
        result = await _current_chat_future
        return result
    except asyncio.CancelledError:
        logger.info("Chat cancelled by /stop")
        return {"reply": "[Agent run stopped by user]", "trace": []}
    finally:
        _current_chat_future = None


def _do_chat(message: str) -> dict:
    agent = _get_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": "default"}},
    )
    messages = result.get("messages", [])
    reply = ""
    if messages:
        last = messages[-1]
        content = getattr(last, "content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        reply = content
    trace = _extract_trace(messages)
    return {"reply": reply, "trace": trace}


@app.post("/stop")
async def stop():
    global _current_chat_future
    if _current_chat_future and not _current_chat_future.done():
        _current_chat_future.cancel()
        logger.info("Agent run cancelled via /stop")
        return {"ok": True, "cancelled": True}
    return {"ok": True, "cancelled": False}


@app.post("/reset")
def reset():
    global _agent
    _agent = None
    logger.info("Agent memory cleared — reinitialising…")
    _get_agent()  # immediately start a fresh session so the agent is ready for the next message
    logger.info("Agent ready (fresh session)")
    return {"ok": True}


# ---------------------------------------------------------------------------
# SAM2 automatic segmentation
# ---------------------------------------------------------------------------

_sam2_gen = None
_sam2_gen_lock = threading.Lock()


def _get_sam2_gen():
    """Lazy-load SAM2AutomaticMaskGenerator; uses CUDA if available."""
    global _sam2_gen
    with _sam2_gen_lock:
        if _sam2_gen is None:
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            device = "cuda" if torch.cuda.is_available() else "cpu"
            # cuDNN init fails inside python:3.11-slim; plain CUDA math works fine
            torch.backends.cudnn.enabled = False
            logger.info(f"Loading SAM2 on {device}…")
            _sam2_gen = SAM2AutomaticMaskGenerator.from_pretrained(
                "facebook/sam2.1-hiera-small",
                device=device,
                points_per_side=16,
                pred_iou_thresh=0.82,
                stability_score_thresh=0.90,
                min_mask_region_area=400,
            )
            logger.info(f"SAM2 ready on {device}")
    return _sam2_gen


class SegmentRequest(BaseModel):
    image_b64: str          # data URL: "data:<mime>;base64,<data>"
    centroids: bool = True
    bboxes: bool = True
    mask: bool = True
    max_dim: int = 640      # resize image to this max dimension before segmenting


@app.post("/segment")
async def segment_image(req: SegmentRequest):
    """Run SAM2 automatic segmentation on a base64-encoded image."""
    try:
        import numpy as np
        from PIL import Image

        # Decode base64 image
        _header, b64data = req.image_b64.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = img.size

        # Resize for speed while preserving aspect ratio
        scale = min(req.max_dim / orig_w, req.max_dim / orig_h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img_small = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            img_small = img
            scale = 1.0

        img_np = np.array(img_small)
        inv = 1.0 / scale

        # Run SAM2 in a thread to avoid blocking the event loop
        generator = _get_sam2_gen()
        loop = asyncio.get_event_loop()

        import torch
        def _run_gen():
            with torch.inference_mode():
                return generator.generate(img_np)

        masks = await loop.run_in_executor(None, _run_gen)
        # Sort largest → smallest so smaller masks render on top
        masks.sort(key=lambda m: m["area"], reverse=True)

        result: dict = {"ok": True, "count": len(masks), "width": orig_w, "height": orig_h}

        if req.centroids:
            result["centroids"] = [
                [int((m["bbox"][0] + m["bbox"][2] / 2) * inv),
                 int((m["bbox"][1] + m["bbox"][3] / 2) * inv)]
                for m in masks
            ]

        if req.bboxes:
            result["bboxes"] = [
                [int(m["bbox"][0] * inv), int(m["bbox"][1] * inv),
                 int(m["bbox"][2] * inv), int(m["bbox"][3] * inv)]
                for m in masks
            ]  # each entry: [x, y, w, h]

        if req.mask:
            # Build RGBA overlay at original resolution
            overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
            rng = random.Random(42)
            for m in masks:
                mask_small = (m["segmentation"].astype(np.uint8) * 255)
                mask_full = np.array(
                    Image.fromarray(mask_small).resize((orig_w, orig_h), Image.NEAREST)
                ).astype(bool)
                color = [rng.randint(60, 230) for _ in range(3)] + [160]
                overlay[mask_full] = color

            mask_img = Image.fromarray(overlay, mode="RGBA")
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            result["mask_png"] = (
                "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            )

        return result

    except Exception as exc:
        logger.exception("Segment error")
        return {"ok": False, "error": str(exc)}
