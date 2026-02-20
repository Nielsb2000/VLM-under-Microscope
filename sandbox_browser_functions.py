"""
sandbox_browser_tools.py - Browser operation tools for the AIO Sandbox browser.

This module provides functions for interacting with the browser inside the sandbox, such as clicking, typing, scrolling, navigation, etc.
"""

import os
from typing import Dict, Any, Optional, List
from agent_sandbox import Sandbox
import pathlib
import datetime
import uuid
import requests
import traceback
import base64
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




def write_binary_file(client, path: str, data: bytes) -> Dict[str, Any]:
    """
    Write binary bytes to a file inside the sandbox.

    Tries sandbox File API first (best). If that fails (SDK expects str),
    falls back to a safe base64->python decode+write in the sandbox.
    """
    # 1) Try File API directly (some SDKs accept bytes)
    try:
        client.file.write_file(file=path, content=data)
        return {"success": True, "method": "file_api_bytes", "path": path, "bytes_written": len(data)}
    except Exception as e1:
        # 2) Fallback: base64 decode and write via sandbox shell
        try:
            b64 = base64.b64encode(data).decode("ascii")
            cmd = (
                "python3 - <<'PY'\n"
                "import base64, os\n"
                f"data = base64.b64decode({b64!r})\n"
                f"path = {path!r}\n"
                "os.makedirs(os.path.dirname(path), exist_ok=True)\n"
                "with open(path, 'wb') as f:\n"
                "    f.write(data)\n"
                "print('WROTE', len(data), 'BYTES')\n"
                "PY"
            )
            client.shell.exec_command(command=cmd)
            return {"success": True, "method": "shell_base64_fallback", "path": path, "bytes_written": len(data), "note": str(e1)}
        except Exception as e2:
            return {"success": False, "error": f"file_api_failed={e1}; fallback_failed={e2}", "path": path}


def take_browser_screenshot_png(
    path: str = "screenshot.png",
    vnc_url: str = "http://localhost:8080/screenshot",
    timeout: float = 10.0,
    retries: int = 2,
    chunk_size: int = 8192,
) -> Dict[str, Any]:
    """
    Take screenshot from the sandbox VNC endpoint and save as a unique PNG.

    Args:
        path: desired filename or path (may include directory). If directory missing,
              will create it. Extension will be .png regardless.
        vnc_url: URL to request screenshot from.
        timeout: seconds for the requests.get timeout.
        retries: number of attempts on transient failure (total tries = retries+1).
        chunk_size: bytes per write when streaming.

    Returns:
        dict with keys:
            - success (bool)
            - path (absolute path to saved file if success else attempted path)
            - bytes_written (int)
            - status_code (int) optional
            - error (str) optional
            - traceback (str) optional
            - url (str) the vnc_url used
    """
    try:
        # Normalize path, ensure PNG extension
        p = pathlib.Path(path)
        # if path is a directory (ends with slash), use default filename
        if str(path).endswith(os.sep) or p.name == "":
            p = (p / "screenshot.png")
        # force .png extension
        p = p.with_suffix(".png")

        # Create directory if needed
        p.parent.mkdir(parents=True, exist_ok=True)

        # Build unique filename: timestamp + short uuid
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")  # UTC
        short_id = uuid.uuid4().hex[:8]
        unique_name = f"{p.stem}_{timestamp}_{short_id}{p.suffix}"
        unique_path = p.parent / unique_name

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                # stream response to file (don't load whole body into memory)
                with requests.get(vnc_url, stream=True, timeout=timeout) as resp:
                    resp.raise_for_status()

                    status_code = getattr(resp, "status_code", None)
                    # Some VNC endpoints may not set Content-Type; be lenient but check if present
                    content_type = resp.headers.get("Content-Type", "").lower()
                    # Accept common image types; prefer png but allow jpeg if server returns that
                    if content_type and "image" not in content_type:
                        # Not necessarily fatal — but warn / raise
                        # We'll still attempt to write but mark as suspicious.
                        pass

                    bytes_written = 0
                    with open(unique_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                bytes_written += len(chunk)

                # Simple validation: first bytes match PNG signature
                with open(unique_path, "rb") as f:
                    header = f.read(8)
                png_sig = b"\x89PNG\r\n\x1a\n"
                if not header.startswith(png_sig):
                    # if not PNG, try renaming to what it actually is (jpeg) or fail
                    # check for JPEG signature
                    if header.startswith(b"\xff\xd8\xff"):
                        # rename to .jpg
                        alt_path = unique_path.with_suffix(".jpg")
                        unique_path.rename(alt_path)
                        unique_path = alt_path
                        # still return success, but with note
                        return {
                            "success": True,
                            "path": str(unique_path.resolve()),
                            "bytes_written": bytes_written,
                            "status_code": status_code,
                            "url": vnc_url,
                            "note": "content was JPEG, file saved with .jpg extension",
                        }
                    else:
                        # Unknown/invalid image
                        raise ValueError("Downloaded content is not PNG or JPEG image (invalid header)")

                # success
                return {
                    "success": True,
                    "path": str(unique_path.resolve()),
                    "bytes_written": bytes_written,
                    "status_code": status_code,
                    "url": vnc_url,
                }

            except Exception as exc:
                last_exc = exc
                # if last attempt, raise, otherwise retry
                if attempt < retries:
                    # simple backoff
                    import time
                    time.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    raise

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "success": False,
            "error": str(e),
            "traceback": tb,
            "path": str(path),
            "url": vnc_url,
        }

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

        r2 = browser_type(url, use_clipboard=False)
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
