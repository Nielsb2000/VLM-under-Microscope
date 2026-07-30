"""agent_api.py - FastAPI wrapper around the DeepAgent (main.py) loop.

Start automatically via docker-compose (root docker-compose.yml).
Manually: uv run uvicorn agent_api:app --host 0.0.0.0 --port 3001

Endpoints:
  GET  /status       -> service/agent/busy status
  POST /chat         -> {"message": "...", "thread_id": "..."} -> {"reply": str, "trace": [...]}
  POST /reset        -> wipes agent memory (new conversation)
  POST /stop         -> requests cancellation of the currently running chat call
  POST /segment      -> SAM2 automatic segmentation
  POST /segment-sam3 -> SAM3 text-prompted segmentation

Important behavior:
  - /chat requests are serialized with an asyncio.Lock instead of rejected with local HTTP 429.
  - Local agent-busy state is no longer reported as 429, so true upstream/provider 429s are easier to diagnose.
  - Each /chat request can provide a thread_id. For benchmark/evaluation runs, pass a unique thread_id per sample/variant
    to avoid cross-sample conversation leakage.
  - CHAT_TIMEOUT_SECONDS can be configured through the environment. Default: 300 seconds.

Trace entry schema:
  { "type": "thinking" | "tool_call" | "tool_result" | "reply",
    "content": str,
    "tool": str | None,
    "input": str | None }
"""

import asyncio
import base64
import io
import logging
import os
import random
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llm_client import get_default_llm

logger = logging.getLogger("agent_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_agent = None
_current_chat_future: Optional[asyncio.Future] = None
_current_request_id: Optional[str] = None
_current_thread_id: Optional[str] = None
_chat_lock: Optional[asyncio.Lock] = None

CHAT_TIMEOUT_SECONDS = int(os.environ.get("CHAT_TIMEOUT_SECONDS", "300"))


def _get_chat_lock() -> asyncio.Lock:
    """Create the asyncio lock lazily inside the active event loop."""
    global _chat_lock
    if _chat_lock is None:
        _chat_lock = asyncio.Lock()
    return _chat_lock


def _get_agent():
    global _agent
    if _agent is None:
        logger.info("Initialising agent...")
        _agent = get_default_llm()
        logger.info("Agent ready")
    return _agent


def _extract_model_name(agent=None) -> str | None:
    """Best-effort model/deployment name extraction for run packaging.

    Prefer explicit environment variables, then shallow introspection of the
    constructed agent/LLM object. This must never raise because trace capture is
    non-critical for the API response.
    """
    for key in (
        "MODEL_NAME",
        "OPENAI_MODEL",
        "LLM_MODEL",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "OPENAI_API_MODEL",
    ):
        value = os.environ.get(key)
        if value:
            return value

    seen: set[int] = set()
    queue = [agent] if agent is not None else []
    attr_names = ("model_name", "model", "deployment_name", "azure_deployment", "model_id")
    while queue and len(seen) < 80:
        obj = queue.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        for attr in attr_names:
            try:
                value = getattr(obj, attr, None)
            except Exception:
                value = None
            if isinstance(value, str) and value.strip():
                return value.strip()
        try:
            data = getattr(obj, "__dict__", {})
        except Exception:
            data = {}
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, str) and any(
                    token in value.lower()
                    for token in ("gpt", "claude", "gemini", "o1", "o3", "o4")
                ):
                    return value
                if hasattr(value, "__dict__") and id(value) not in seen:
                    queue.append(value)
    return None


def _extract_trace(messages: list) -> list[dict]:
    """Convert LangChain message list -> step objects for the UI.

    Each step corresponds to one AIMessage and contains:
      - thinking: the model's reasoning text (may be None)
      - calls: list of tool calls, each with its matched result

    ToolMessages are matched to their AIMessage tool_call via tool_call_id.
    """
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
            name = tc.get("name", "unknown")
            args = tc.get("args", {}) or {}
            tcid = tc.get("id", "")
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

            calls.append(
                {
                    "tool": name,
                    "action": action,
                    "category": _classify_tool(name, action),
                    "input_summary": _summarise_args(name, args),
                    "result": (raw_result[:800] + "...") if len(raw_result) > 800 else raw_result,
                    "result_is_json": result_is_json,
                }
            )

        steps.append(
            {
                "type": "step",
                "step": step_n,
                "thinking": thinking,
                "calls": calls,
            }
        )

    return steps


# ---- Tool category classification ----

_PAINT_DRAW = {"rect", "ellipse", "arrow", "dot", "line", "text", "bulk"}
_PAINT_ADJUST = {
    "set_filters",
    "zoom",
    "pan",
    "zoom_to_region",
    "zoom_to_object",
    "reset_viewport",
    "cursor_move",
    "cursor_hide",
    "viewport",
}
_PAINT_NAV = {
    "load_tile_grid",
    "camera_left",
    "camera_right",
    "camera_up",
    "camera_down",
    "camera_move",
    "camera_goto",
    "camera_state",
    "set_canvas_mode",
}
_PAINT_IMAGE = {
    "load_image",
    "load_image_by_name",
    "get_canvas_image",
    "export_png",
    "export_json",
    "crop",
    "list_images",
}
_EXEC_TOOLS = {"execute", "read_file", "write_file", "glob", "grep", "bash", "run_command"}


def _classify_tool(name: str, action: str | None) -> str:
    if name == "paint_canvas":
        if action in _PAINT_DRAW:
            return "draw"
        if action in _PAINT_ADJUST:
            return "adjust"
        if action in _PAINT_NAV:
            return "navigate"
        if action in _PAINT_IMAGE:
            return "image"
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
    if name == "paint_canvas":
        params = args.get("params", {}) or {}
        if params and isinstance(params, dict):
            parts = []
            for k, v in params.items():
                s = str(v)
                if len(s) > 80:
                    s = s[:80] + "..."
                parts.append(f"{k}={s}")
            return ", ".join(parts)
    parts = []
    for k, v in args.items():
        if k == "action" and name == "paint_canvas":
            continue
        s = str(v)
        if len(s) > 100:
            s = s[:100] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _get_chat_lock()
    _get_agent()
    yield


app = FastAPI(title="DeepAgent API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = Field(
        default="default",
        description="Conversation thread id. Use a unique value per benchmark sample/variant to avoid context leakage.",
    )


@app.get("/status")
def status():
    busy = _current_chat_future is not None and not _current_chat_future.done()
    return {
        "running": _agent is not None,
        "busy": busy,
        "current_request_id": _current_request_id,
        "current_thread_id": _current_thread_id,
        "chat_timeout_seconds": CHAT_TIMEOUT_SECONDS,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """Run one agent chat call.

    Calls are serialized instead of rejected. This avoids local HTTP 429s during
    batch evaluation when the evaluator submits another request while the agent
    is still processing the previous one.
    """
    global _current_chat_future, _current_request_id, _current_thread_id

    request_id = uuid.uuid4().hex[:8]
    lock = _get_chat_lock()

    if lock.locked():
        logger.info(
            "Chat request %s queued behind active request=%s thread_id=%s",
            request_id,
            _current_request_id,
            _current_thread_id,
        )

    async with lock:
        _current_request_id = request_id
        _current_thread_id = req.thread_id
        started = time.time()
        logger.info("Chat request %s started thread_id=%s", request_id, req.thread_id)

        loop = asyncio.get_event_loop()
        _current_chat_future = loop.run_in_executor(
            None,
            _do_chat,
            req.message,
            req.thread_id,
            request_id,
        )

        try:
            result = await asyncio.wait_for(_current_chat_future, timeout=CHAT_TIMEOUT_SECONDS)
            elapsed = time.time() - started
            logger.info("Chat request %s completed in %.1fs", request_id, elapsed)
            return result
        except asyncio.TimeoutError:
            elapsed = time.time() - started
            logger.warning("Chat request %s timed out after %.1fs", request_id, elapsed)
            return JSONResponse(
                status_code=504,
                content={
                    "error": "agent_timeout",
                    "message": f"Agent call exceeded {CHAT_TIMEOUT_SECONDS}s timeout.",
                    "request_id": request_id,
                    "thread_id": req.thread_id,
                },
            )
        except asyncio.CancelledError:
            logger.info("Chat request %s cancelled by /stop", request_id)
            return {
                "reply": "[Agent run stopped by user]",
                "trace": [],
                "request_id": request_id,
                "thread_id": req.thread_id,
            }
        finally:
            _current_chat_future = None
            _current_request_id = None
            _current_thread_id = None


def _do_chat(message: str, thread_id: str = "default", request_id: str = "unknown") -> dict:
    import json as _json
    from datetime import datetime as _dt

    started_at = _dt.utcnow().isoformat() + "Z"
    agent = _get_agent()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    completed_at = _dt.utcnow().isoformat() + "Z"
    messages = result.get("messages", [])
    model_name = _extract_model_name(agent)
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

    try:
        traces_dir = Path("logs") / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_thread_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in thread_id)[:80]
        trace_path = traces_dir / f"trace_{ts}_{request_id}_{safe_thread_id}.json"
        trace_payload = {
            "request_id": request_id,
            "thread_id": thread_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "user_message": message,
            "model_name": model_name,
            "reply": reply,
            "steps": trace,
        }
        trace_path.write_text(_json.dumps(trace_payload, indent=2))
        logger.info("Trace saved -> %s", trace_path)
    except Exception as exc:
        logger.warning("Could not persist trace: %s", exc)

    return {
        "reply": reply,
        "trace": trace,
        "model_name": model_name,
        "request_id": request_id,
        "thread_id": thread_id,
    }


@app.post("/stop")
async def stop():
    global _current_chat_future
    if _current_chat_future and not _current_chat_future.done():
        _current_chat_future.cancel()
        logger.info("Agent run cancellation requested via /stop")
        return {
            "ok": True,
            "cancelled": True,
            "note": "Cancellation was requested. Blocking work running in a thread may continue until the underlying agent call returns.",
        }
    return {"ok": True, "cancelled": False}


@app.post("/reset")
def reset():
    global _agent
    _agent = None
    logger.info("Agent memory cleared - reinitialising...")
    _get_agent()
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
            torch.backends.cudnn.enabled = False
            logger.info(f"Loading SAM2 on {device}...")
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
    image_b64: str
    centroids: bool = True
    bboxes: bool = True
    mask: bool = True
    max_dim: int = 640


@app.post("/segment")
async def segment_image(req: SegmentRequest):
    """Run SAM2 automatic segmentation on a base64-encoded image."""
    try:
        import numpy as np
        from PIL import Image

        _header, b64data = req.image_b64.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = img.size

        scale = min(req.max_dim / orig_w, req.max_dim / orig_h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img_small = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            img_small = img
            scale = 1.0

        img_np = np.array(img_small)
        inv = 1.0 / scale

        generator = _get_sam2_gen()
        loop = asyncio.get_event_loop()

        import torch

        def _run_gen():
            with torch.inference_mode():
                return generator.generate(img_np)

        masks = await loop.run_in_executor(None, _run_gen)
        masks.sort(key=lambda m: m["area"], reverse=True)

        result: dict = {"ok": True, "count": len(masks), "width": orig_w, "height": orig_h}

        if req.centroids:
            result["centroids"] = [
                [
                    int((m["bbox"][0] + m["bbox"][2] / 2) * inv),
                    int((m["bbox"][1] + m["bbox"][3] / 2) * inv),
                ]
                for m in masks
            ]

        if req.bboxes:
            result["bboxes"] = [
                [
                    int(m["bbox"][0] * inv),
                    int(m["bbox"][1] * inv),
                    int(m["bbox"][2] * inv),
                    int(m["bbox"][3] * inv),
                ]
                for m in masks
            ]

        if req.mask:
            overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
            rng = random.Random(42)
            for m in masks:
                mask_small = m["segmentation"].astype(np.uint8) * 255
                mask_full = np.array(
                    Image.fromarray(mask_small).resize((orig_w, orig_h), Image.NEAREST)
                ).astype(bool)
                color = [rng.randint(60, 230) for _ in range(3)] + [160]
                overlay[mask_full] = color

            mask_img = Image.fromarray(overlay, mode="RGBA")
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            result["mask_png"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return result

    except Exception as exc:
        logger.exception("Segment error")
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# SAM3 text-prompted segmentation
# ---------------------------------------------------------------------------

_sam3_model = None
_sam3_proc = None
_sam3_lock = threading.Lock()


def _get_sam3():
    """Lazy-load Sam3Model + Sam3Processor; uses CUDA if available."""
    global _sam3_model, _sam3_proc
    with _sam3_lock:
        if _sam3_model is None or _sam3_proc is None:
            _m, _p = None, None
            try:
                import torch
                from transformers import Sam3Model, Sam3Processor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                torch.backends.cudnn.enabled = False
                dtype = torch.bfloat16 if device == "cuda" else torch.float32

                hf_cache = os.environ.get(
                    "HF_HUB_CACHE",
                    os.path.expanduser("~/.cache/huggingface/hub"),
                )
                snap_root = os.path.join(hf_cache, "models--facebook--sam3", "snapshots")
                model_path = "facebook/sam3"
                if os.path.isdir(snap_root):
                    snaps = sorted(os.listdir(snap_root))
                    if snaps:
                        model_path = os.path.join(snap_root, snaps[-1])

                logger.info(f"Loading SAM3 from {model_path} on {device}...")
                _p = Sam3Processor.from_pretrained(model_path)
                _m = Sam3Model.from_pretrained(model_path, torch_dtype=dtype).to(device)
                logger.info(f"SAM3 ready on {device}")
            except Exception:
                _sam3_model = None
                _sam3_proc = None
                raise
            _sam3_model = _m
            _sam3_proc = _p
    return _sam3_model, _sam3_proc


class SegmentSam3Request(BaseModel):
    image_b64: str
    text_prompt: str = "particle"
    threshold: float = 0.4
    mask_threshold: float = 0.4
    max_dim: int = 640
    return_overlay: bool = True
    return_instances: bool = True


@app.post("/segment-sam3")
async def segment_sam3(req: SegmentSam3Request):
    """Text-prompted instance segmentation via SAM3 (facebook/sam3)."""
    try:
        import numpy as np
        import torch
        from PIL import Image

        _header, b64data = req.image_b64.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = img.size

        scale = min(req.max_dim / orig_w, req.max_dim / orig_h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img_small = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            img_small = img
            scale = 1.0
        inv = 1.0 / scale

        model, processor = _get_sam3()

        def _run_sam3():
            inputs = processor(images=img_small, text=req.text_prompt, return_tensors="pt")
            dev = next(model.parameters()).device
            inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            return processor.post_process_instance_segmentation(
                outputs,
                threshold=req.threshold,
                mask_threshold=req.mask_threshold,
                target_sizes=[(img_small.height, img_small.width)],
            )[0]

        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, _run_sam3)

        masks = res["masks"]
        scores = res["scores"]
        boxes = res["boxes"]

        count = len(masks)
        result: dict = {
            "ok": True,
            "count": count,
            "text_prompt": req.text_prompt,
            "width": orig_w,
            "height": orig_h,
        }

        if req.return_instances:
            instances = []
            for mask_t, score, box in zip(masks, scores.tolist(), boxes.tolist()):
                m = mask_t.cpu().numpy().astype(bool)
                ys, xs = np.where(m)
                cx = int(xs.mean() * inv) if ys.size else 0
                cy = int(ys.mean() * inv) if ys.size else 0
                x0, y0, x1, y1 = [int(v * inv) for v in box]
                instances.append(
                    {
                        "score": round(score, 3),
                        "centroid": [cx, cy],
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                    }
                )
            result["instances"] = instances

        if req.return_overlay:
            overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
            rng = random.Random(42)
            for mask_t in masks:
                m_small = mask_t.cpu().numpy().astype(np.uint8) * 255
                m_full = np.array(
                    Image.fromarray(m_small).resize((orig_w, orig_h), Image.NEAREST)
                ).astype(bool)
                color = [rng.randint(60, 230) for _ in range(3)] + [160]
                overlay[m_full] = color
            overlay_img = Image.fromarray(overlay, mode="RGBA")
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            result["composite_overlay"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return result

    except Exception as exc:
        logger.exception("Segment-SAM3 error")
        return {"ok": False, "error": str(exc)}
