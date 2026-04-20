import os
import io
import time
from datetime import datetime
from typing import Dict, Any, List
from PIL import Image, ImageDraw
from agent_sandbox import Sandbox
from sandbox_browser_functions import (
    browser_navigate_gui,
    browser_click,
    browser_move_to,
    browser_type,
    browser_hotkey,
    browser_scroll,
    browser_drag_to,
    get_sandbox_client,
    write_binary_file
)

_CURSOR_RADIUS = 10
_CURSOR_COLOR = (220, 30, 30, 220)   # red, slightly transparent
_CURSOR_OUTLINE = (255, 255, 255, 255)  # white ring for contrast


def _overlay_cursor(png_bytes: bytes, x: int, y: int) -> bytes:
    """Draw a red dot at (x, y) on the screenshot PNG and return updated bytes."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r = _CURSOR_RADIUS
    # White outline ring
    draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=_CURSOR_OUTLINE)
    # Red dot
    draw.ellipse([x - r, y - r, x + r, y + r], fill=_CURSOR_COLOR)
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    composited.save(buf, format="PNG")
    return buf.getvalue()


def run_visible_browser_steps(
    steps: List[Dict[str, Any]],
    screenshot_dir: str = "/workspace/screenshots",
    settle_seconds: float = 0.8,
) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"/workspace/screenshots/steps_{timestamp}"
    results = []
    client = get_sandbox_client()

    # Ensure session dir exists and is writable by gem user
    client.shell.exec_command(command=f"mkdir -p '{session_dir}' && chmod 777 '/workspace/screenshots' '{session_dir}'")

    cursor_x, cursor_y = None, None  # track latest known cursor position

    for i, step in enumerate(steps, start=1):
        op = step.get("op")

        if op == "navigate":
            res = browser_navigate_gui(
                step["url"],
                settle_seconds=step.get("settle_seconds", 2.0),
            )
        elif op == "click":
            cursor_x, cursor_y = step.get("x", cursor_x), step.get("y", cursor_y)
            res = browser_click(
                step.get("x"), step.get("y"),
                num_clicks=step.get("num_clicks", 1),
                button=step.get("button"),
            )
        elif op == "move":
            cursor_x, cursor_y = step["x"], step["y"]
            res = browser_move_to(step["x"], step["y"])
        elif op == "type":
            # IMPORTANT: default clipboard off
            res = browser_type(step["text"], use_clipboard=step.get("use_clipboard", False))
        elif op == "hotkey":
            res = browser_hotkey(step["keys"])
        elif op == "scroll":
            res = browser_scroll(dy=step.get("dy", 400), dx=step.get("dx", 0))
        elif op == "drag":
            cursor_x, cursor_y = step["x"], step["y"]
            res = browser_drag_to(step["x"], step["y"])
        else:
            res = {"success": False, "error": f"Unknown op: {op}", "step": step}

        # Wait for browser to settle before screenshot
        time.sleep(step.get("settle_seconds", settle_seconds))

        if not res.get("success"):
            return {"success": False, "results": results, "error": res.get("error")}

        # Screenshot after each step (after sleep, only if step succeeded)
        shot_path = os.path.join(session_dir, f"screenshot{i:02d}_{op}.png")

        screenshot_data = client.browser.screenshot()
        png_bytes = b"".join(screenshot_data)
        # Overlay red cursor dot at last known position
        if cursor_x is not None and cursor_y is not None:
            png_bytes = _overlay_cursor(png_bytes, cursor_x, cursor_y)
        write_res = write_binary_file(client, shot_path, png_bytes)
        results.append({
            "step_index": i,
            "step": step,
            "op_result": res,
            "screenshot": {
                "success": write_res.get("success", False),
                "path": shot_path,
                "bytes": len(png_bytes),
                "write": write_res,
            },
        })

    return {"success": True, "results": results, "screenshot_dir": session_dir}