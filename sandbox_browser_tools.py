import os
import io
import time
from datetime import datetime
from typing import Dict, Any, List
from PIL import Image, ImageDraw
from agent_sandbox import Sandbox
from agent_tools_vision import _overlay_coordinate_grid, _overlay_cursor
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

# _overlay_cursor is imported from agent_tools_vision


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

    # Disable PyAutoGUI fail-safe by patching the source file in the sandbox.
    # The server process caches the module, so we patch it via a running python -c
    # that modifies the global state of the already-imported module through /proc.
    # The most reliable way: sed the default in the installed source file so any
    # fresh subprocess or server restart picks it up too.
    client.shell.exec_command(
        command=(
            "python3 -c \""
            "import pyautogui, inspect, pathlib; "
            "src = pathlib.Path(inspect.getfile(pyautogui)); "
            "txt = src.read_text(); "
            "patched = txt.replace('FAILSAFE = True', 'FAILSAFE = False'); "
            "src.write_text(patched) if 'FAILSAFE = True' in txt else None\""
        )
    )

    # Capture initial state: save a coordinate-grid overlay as screenshot00_initial_grid.png
    try:
        initial_data = client.browser.screenshot()
        initial_png = b"".join(initial_data)
        grid_png = _overlay_coordinate_grid(initial_png)
        grid_path = os.path.join(session_dir, "screenshot00_initial_grid.png")
        write_binary_file(client, grid_path, grid_png)
    except Exception as e:
        print(f"[initial_grid] failed to save grid screenshot: {e}")

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
        elif op in ("drag", "dragto"):
            # Support both {x, y} (dest only) and {x1, y1, x2, y2} schemas
            if "x1" in step:
                browser_move_to(step["x1"], step["y1"])
                time.sleep(0.1)
                cursor_x, cursor_y = step["x2"], step["y2"]
                res = browser_drag_to(step["x2"], step["y2"])
            else:
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
            try:
                before_size = len(png_bytes)
                overlaid = _overlay_cursor(png_bytes, cursor_x, cursor_y)
                png_bytes = overlaid
                print(f"[overlay_cursor] step {i} ({op}): {before_size} -> {len(png_bytes)} bytes at ({cursor_x},{cursor_y})")
            except Exception as overlay_err:
                print(f"[overlay_cursor] FAILED step {i} ({op}) at ({cursor_x},{cursor_y}): {overlay_err}")
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