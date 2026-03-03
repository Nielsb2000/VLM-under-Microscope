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
    try:
        base_dir = "/workspace/screenshots"
        rel_path = os.path.relpath(path, base_dir)
        safe_path = os.path.join(base_dir, rel_path)
        # Ensure screenshots directory and subfolders exist
        client.shell.exec_command(command=f"mkdir -p '{os.path.dirname(safe_path)}'")
        client.shell.exec_command(command=f"mkdir -p '{base_dir}'")
        b64 = base64.b64encode(data).decode("ascii")
        cmd = (
            "python3 - <<'PY'\n"
            "import base64, os\n"
            f"data = base64.b64decode({b64!r})\n"
            f"path = {safe_path!r}\n"
            "os.makedirs(os.path.dirname(path), exist_ok=True)\n"
            "with open(path, 'wb') as f:\n"
            "    f.write(data)\n"
            "print('WROTE', len(data), 'BYTES')\n"
            "PY"
        )
        res = client.shell.exec_command(command=cmd)
        shell_output = getattr(res, 'output', None)
        shell_error = getattr(res, 'error', None)
        # Print and log shell output and error
        print(f"[write_binary_file] Shell output: {shell_output}")
        print(f"[write_binary_file] Shell error: {shell_error}")
        # Ensure log directory exists before writing log
        client.shell.exec_command(command=f"mkdir -p '{base_dir}'")
        log_path = os.path.join(base_dir, "write_binary_file_shell.log")
        try:
            with open(log_path, "a") as logf:
                logf.write(f"PATH: {safe_path}\nOUTPUT: {shell_output}\nERROR: {shell_error}\n\n")
        except Exception as log_exc:
            print(f"[write_binary_file] Log write exception: {log_exc}")
        if res and getattr(res, 'success', True):
            return {"success": True, "method": "shell_base64", "path": safe_path, "bytes_written": len(data), "shell_output": shell_output}
        else:
            return {"success": False, "path": safe_path, "bytes_written": len(data), "shell_output": shell_output, "shell_error": shell_error}
    except Exception as e:
        print(f"[write_binary_file] Exception: {e}")
        try:
            client.shell.exec_command(command=f"mkdir -p '{base_dir}'")
            log_path = os.path.join(base_dir, "write_binary_file_shell.log")
            with open(log_path, "a") as logf:
                logf.write(f"PATH: {path}\nEXCEPTION: {e}\n\n")
        except Exception as log_exc:
            print(f"[write_binary_file] Log write exception: {log_exc}")
        return {"success": False, "path": path, "bytes_written": len(data), "error": str(e)}


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
            # Method 4: Call minimal working screenshot script via shell command
            try:
                temp_path = path + ".tmp"
                res_temp = client.file.write_file(file=temp_path, content=data)
                if res_temp and getattr(res_temp, 'success', True):
                    cmd4 = f"python3 test_basic_screenshot.py {temp_path} {path}"
                    res4 = client.shell.exec_command(command=cmd4)
                    if res4 and getattr(res4, 'success', True):
                        return {"success": True, "method": "minimal_script_shell", "path": path, "bytes_written": len(data)}
            except Exception:
                pass
            # Method 1: Sandbox file API
            try:
                res1 = client.file.write_file(file=path, content=data)
                if res1 and getattr(res1, 'success', True):
                    return {"success": True, "method": "file_api", "path": path, "bytes_written": len(data)}
            except Exception:
                pass
            # Method 2: Shell command (base64 decode)
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
                res2 = client.shell.exec_command(command=cmd)
                if res2 and getattr(res2, 'success', True):
                    return {"success": True, "method": "shell_base64", "path": path, "bytes_written": len(data)}
            except Exception:
                pass
            # Method 3: Python file write via exec_command
            try:
                hex_bytes = data.hex()
                cmd3 = (
                    "python3 - <<'PY'\n"
                    "import os\n"
                    f"path = {path!r}\n"
                    f"data = bytes.fromhex({hex_bytes!r})\n"
                    "os.makedirs(os.path.dirname(path), exist_ok=True)\n"
                    "with open(path, 'wb') as f:\n"
                    "    f.write(data)\n"
                    "print('WROTE', len(data), 'BYTES')\n"
                    "PY"
                )
                res3 = client.shell.exec_command(command=cmd3)
                if res3 and getattr(res3, 'success', True):
                    return {"success": True, "method": "shell_hex", "path": path, "bytes_written": len(data)}
            except Exception:
                pass
            return {"success": False, "path": path, "bytes_written": len(data)}

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
