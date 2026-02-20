"""
sandbox_browser_steps.py - Run visible browser steps in the sandbox browser.

This module provides the run_visible_browser_steps function for executing a sequence of browser actions and saving screenshots after each step.
"""

import os
import time
from typing import Dict, Any, List
from sandbox_browser_functions import (
    browser_navigate_gui,
    browser_click,
    browser_move_to,
    browser_type,
    browser_hotkey,
    browser_scroll,
    browser_drag_to,
    take_browser_screenshot_png,
)

def run_visible_browser_steps(
    steps: List[Dict[str, Any]],
    screenshot_dir: str = "/home/gem/screenshots/browser_steps",
    settle_seconds: float = 0.8,
) -> Dict[str, Any]:
    """
    steps: list of dicts like:
      {"op":"navigate","url":"https://..."}
      {"op":"click","x":100,"y":200}
      {"op":"type","text":"hello"}
      {"op":"hotkey","keys":["enter"]}
      {"op":"scroll","dy":600}
    Saves screenshots after every step so you can see what happened.
    """
    os.makedirs(screenshot_dir, exist_ok=True)
    results = []

    for i, step in enumerate(steps, start=1):
        op = step.get("op")
        if op == "navigate":
            res = browser_navigate_gui(
                step["url"],
                settle_seconds=step.get("settle_seconds", 2.0),
            )
        elif op == "click":
            res = browser_click(step.get("x"), step.get("y"),
                                num_clicks=step.get("num_clicks", 1),
                                button=step.get("button"))
        elif op == "move":
            res = browser_move_to(step["x"], step["y"])
        elif op == "type":
            res = browser_type(step["text"], use_clipboard=step.get("use_clipboard", True))
        elif op == "hotkey":
            res = browser_hotkey(step["keys"])
        elif op == "scroll":
            res = browser_scroll(dy=step.get("dy", 400), dx=step.get("dx", 0))
        elif op == "drag":
            res = browser_drag_to(step["x"], step["y"])
        else:
            res = {"success": False, "error": f"Unknown op: {op}", "step": step}

        # Always screenshot after each step for visibility
        shot_path = os.path.join(screenshot_dir, f"{i:02d}_{op}.png")
        shot = take_browser_screenshot_png(shot_path)

        results.append({
            "step_index": i,
            "step": step,
            "op_result": res,
            "screenshot": shot,
        })

        time.sleep(step.get("settle_seconds", settle_seconds))

        # Optional: stop early on failure
        if not res.get("success"):
            return {"success": False, "results": results, "error": res.get("error")}

    return {"success": True, "results": results, "screenshot_dir": screenshot_dir}
