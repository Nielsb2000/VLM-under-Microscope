"""
sandbox_browser_tools.py - Browser operation tools for the AIO Sandbox browser.

This module provides functions for interacting with the browser inside the sandbox, such as clicking, typing, scrolling, navigation, etc.
"""

import os
from typing import Dict, Any, Optional, List
from agent_sandbox import Sandbox
from agent_sandbox.browser import (
    Action_Click,
    Action_MoveTo,
    Action_Typing,
    Action_Scroll,
    Action_Hotkey,
    Action_DragTo,
)

def get_sandbox_client() -> Sandbox:
    sandbox_url = os.getenv("SANDBOX_BASE_URL", "http://localhost:8080")
    return Sandbox(base_url=sandbox_url)


def take_browser_screenshot_png(path: str = "screenshot.png") -> Dict[str, Any]:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # ✅ add this
        client = get_sandbox_client()
        stream = client.browser.take_screenshot()
        written = 0
        with open(path, "wb") as f:
            for chunk in stream:
                f.write(chunk)
                written += len(chunk)
        return {"success": True, "path": path, "bytes_written": written}
    except Exception as e:
        return {"success": False, "error": str(e), "path": path}


def browser_hotkey(keys: List[str]) -> Dict[str, Any]:
    try:
        client = get_sandbox_client()
        res = client.browser.execute_action(request=Action_Hotkey(keys=keys))
        return {"success": True, "result": res}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_type(text: str, use_clipboard: bool = True) -> Dict[str, Any]:
    try:
        client = get_sandbox_client()
        res = client.browser.execute_action(
            request=Action_Typing(text=text, use_clipboard=use_clipboard)
        )
        return {"success": True, "result": res}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_click(x: Optional[int] = None, y: Optional[int] = None,
                  num_clicks: int = 1, button: Optional[str] = None) -> Dict[str, Any]:
    try:
        client = get_sandbox_client()
        res = client.browser.execute_action(
            request=Action_Click(x=x, y=y, num_clicks=num_clicks, button=button)
        )
        return {"success": True, "result": res}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_move_to(x: int, y: int) -> Dict[str, Any]:
    try:
        client = get_sandbox_client()
        res = client.browser.execute_action(request=Action_MoveTo(x=x, y=y))
        return {"success": True, "result": res}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_scroll(dy: int = 400, dx: int = 0) -> Dict[str, Any]:
    try:
        client = get_sandbox_client()
        res = client.browser.execute_action(request=Action_Scroll(dx=dx, dy=dy))
        return {"success": True, "result": res}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_drag_to(x: int, y: int) -> Dict[str, Any]:
    try:
        client = get_sandbox_client()
        res = client.browser.execute_action(request=Action_DragTo(x=x, y=y))
        return {"success": True, "result": res}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_navigate_gui(url: str, settle_seconds: float = 2.0,
                         screenshot_path: Optional[str] = None) -> Dict[str, Any]:
    import time
    try:
        r1 = browser_hotkey(["ctrl", "l"])
        if not r1.get("success"):
            return r1

        r2 = browser_type(url, use_clipboard=True)
        if not r2.get("success"):
            return r2

        r3 = browser_hotkey(["enter"])
        if not r3.get("success"):
            return r3

        if settle_seconds:
            time.sleep(settle_seconds)

        out = {"success": True, "url": url, "result": r3.get("result")}
        if screenshot_path:
            out["screenshot"] = take_browser_screenshot_png(screenshot_path)
        return out
    except Exception as e:
        return {"success": False, "error": str(e), "url": url}
