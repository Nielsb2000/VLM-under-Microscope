import os
import time
import base64
from typing import Dict, Any, List
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


def run_visible_browser_steps(
    steps: List[Dict[str, Any]],
    screenshot_dir: str = "/workspace/screenshots",
    settle_seconds: float = 0.8,
) -> Dict[str, Any]:
    results = []
    client = get_sandbox_client()

    # Ensure dir exists *in the sandbox*
    client.shell.exec_command(command=f"mkdir -p '{screenshot_dir}'")

    for i, step in enumerate(steps, start=1):
        op = step.get("op")

        if op == "navigate":
            res = browser_navigate_gui(
                step["url"],
                settle_seconds=step.get("settle_seconds", 2.0),
            )
        elif op == "click":
            res = browser_click(
                step.get("x"), step.get("y"),
                num_clicks=step.get("num_clicks", 1),
                button=step.get("button"),
            )
        elif op == "move":
            res = browser_move_to(step["x"], step["y"])
        elif op == "type":
            # IMPORTANT: default clipboard off
            res = browser_type(step["text"], use_clipboard=step.get("use_clipboard", False))
        elif op == "hotkey":
            res = browser_hotkey(step["keys"])
        elif op == "scroll":
            res = browser_scroll(dy=step.get("dy", 400), dx=step.get("dx", 0))
        elif op == "drag":
            res = browser_drag_to(step["x"], step["y"])
        else:
            res = {"success": False, "error": f"Unknown op: {op}", "step": step}

        # Screenshot after each step
        shot_path = os.path.join(screenshot_dir, f"{i:02d}_{op}.png")

        png_bytes = b"".join(client.browser.screenshot())

        # optional sanity check
        if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return {"success": False, "results": results, "error": f"Not PNG, head={png_bytes[:16]!r}"}

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

        time.sleep(step.get("settle_seconds", settle_seconds))

        if not res.get("success"):
            return {"success": False, "results": results, "error": res.get("error")}

    return {"success": True, "results": results, "screenshot_dir": screenshot_dir}