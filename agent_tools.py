"""Tool definitions for the LLM agent to interact with the AIO Sandbox.

Note: Basic file operations (read, write, list, execute) are handled by built-in
deepagents tools through AIOSandboxBackend. These tools provide unique functionality.
"""

from langchain_core.tools import tool
from sandbox_browser_tools import run_visible_browser_steps
import subprocess
import json as _json
import os as _os
from urllib.request import urlopen, Request as _Request
from urllib.parse import urlencode as _urlencode
from MCP_functions import call_mcp_tool
from sandbox_core_functions import get_sandbox_context, create_skill_in_sandbox

_PAINT_BASE = _os.environ.get("PAINT_SERVICE_URL", "http://localhost:3000")


def _paint(method: str, path: str, body: dict | None = None, binary: bool = False):
    """Internal helper — call the paint-service REST API."""
    url = f"{_PAINT_BASE}{path}"
    data = _json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = _Request(url, data=data, headers=headers, method=method.upper())
    with urlopen(req, timeout=15) as resp:
        return resp.read() if binary else _json.loads(resp.read())


@tool
def paint_canvas(action: str, params: dict = None) -> dict:
    """Control the paint-service annotation canvas.

    Use this tool to programmatically annotate images and export results.
    The paint-service must be running at http://localhost:3000 (start it with
    `docker run -p 3000:3000 paint-service` or `docker-compose up -d` inside
    the paint-service/ folder).

    **action** — one of:

    Canvas lifecycle:
      - "new"          Reset canvas. params: {width, height}
      - "clear"        Remove all annotations (keep background).
      - "state"        Return full canvas + objects JSON.

    Image:
      - "load_image"   Load background image. params: {url} OR {path} (local file).

    Drawing (all support optional "label", "createdBy"):
      - "rect"         params: {x, y, width, height, stroke, fill, strokeWidth}
      - "ellipse"      params: {cx, cy, rx, ry, stroke, fill, strokeWidth}
      - "arrow"        params: {x1, y1, x2, y2, stroke, strokeWidth}
      - "dot"          params: {cx, cy, radius, fill}
      - "line"         params: {x1, y1, x2, y2, stroke, strokeWidth}
      - "text"         params: {x, y, text, fontSize, fill}
      - "bulk"         params: {operations: [list of shape dicts with "type" field]}

    Objects:
      - "list"         Return list of all objects.
      - "update"       params: {id, ...fields to update}
      - "delete"       params: {id}

    Export:
      - "export_png"   Download annotated PNG → returns {saved_to, size_bytes}.
                       params: {save_to} (optional local path, defaults to /tmp/annotated.png)
      - "export_json"  Return annotation JSON.
      - "get_canvas_image"  Fetch the current canvas (background + annotations) as a
                       base64-encoded PNG data URL. Also saves the PNG to a timestamped
                       folder under screenshots/paint_<timestamp>/ on the host, which is
                       volume-mounted into the sandbox at /workspace/screenshots/paint_<timestamp>/.
                       Paint-service exports always land in paint_* folders; browser step
                       screenshots land in steps_* folders — they never mix.
                       Returns {saved_to, width, height}.
                       params: {filename} (optional, default "canvas_export.png")
                       Use this to give a vision model a snapshot of what is currently
                       visible on the canvas — e.g. to inspect, count, or locate objects.
                       After calling this, pass saved_to to analyze_sandbox_image.

    Viewport (changes what the human sees in real-time — NO image change, just navigation):
      - "zoom"           Zoom in/out on the canvas view. Does NOT crop or alter the image.
                         Use this when the user just wants to get a closer look at something
                         without permanently changing the underlying image.
                         params: {zoom, centerX, centerY}
                         centerX/Y are canvas coords to keep centered.
      - "pan"            Pan (scroll) the view. Does NOT alter the image.
                         params: {x, y} (canvas point at viewport top-left),
                         or relative move: {relative: true, dx, dy}
      - "zoom_to_region" Fit a canvas bounding box into full view. Does NOT alter the image.
                         params: {x, y, width, height, padding(optional, default 60)}
      - "zoom_to_object" Fit a specific annotation into view by id. Does NOT alter the image.
                         params: {id, padding(optional)}
      - "reset_viewport" Return to 1:1 zoom at origin. Does NOT alter the image.
      - "viewport"       Return current viewport state.
      - "set_filters"    Adjust visual appearance of the canvas viewport (does NOT alter the
                         underlying image — purely a display filter).
                         params: {brightness, contrast, saturation} — all optional, values are
                         percentages (100 = normal, 0 = min, 200 = doubled, max 300).
                         Use to enhance visibility of faint features, improve contrast for
                         analysis, or boost colour saturation.

    Image crop (permanently replaces the background with a new cropped image):
      - "crop"           Slice out a sub-region of the background image and load it as the
                         new canvas background. This PERMANENTLY changes the displayed image —
                         the old image is replaced by the cropped sub-region. Use this when
                         the human has drawn a bounding box and wants to "zoom into" that
                         region as a new image (e.g. for closer inspection or annotation).
                         After cropping, the canvas size equals the crop dimensions and the
                         viewport zoom is set so the new image fills the window.
                         Annotations inside the crop region are translated to the new origin;
                         those outside are removed.

                         Rectangular crop (axis-aligned bounding box):
                           params: {x, y, width, height, keepAnnotations(default True)}
                           x/y/width/height are canvas coordinates of the region to cut out.

                         Shape-masked crop (irregular/circular cutout):
                           params: {shapeId, keepAnnotations(default True)}
                           shapeId is the id of a closed shape drawn on the canvas (ellipse,
                           freehand, etc.). The bounding box is derived from the shape and
                           pixels outside the shape boundary are transparent in the output.
                           Example: human draws an ellipse → pass its id as shapeId → the
                           output image is a circular/elliptical crop.

                         The agent MUST read canvas state first to get shape ids and
                         coordinates — do NOT guess them.

    Model cursor (shows a visible cyan cursor on the human's screen):
      - "cursor_move"    Move model cursor to canvas coordinates.
                         params: {x, y, label(optional)}   — cursor appears immediately.
      - "cursor_hide"    Hide the model cursor.

    Images:
      - "list_images"    List all uploaded images available on the server.
      - "load_image_by_name"  Load a previously uploaded image by filename.
                         params: {filename}

    Returns:
        dict with the API response or {saved_to, size_bytes} for PNG export.
    """
    if params is None:
        params = {}

    match action:
        case "new":
            return _paint("POST", "/api/canvas/new", params or {"width": 1200, "height": 800})
        case "clear":
            return _paint("POST", "/api/canvas/clear")
        case "state":
            return _paint("GET", "/api/canvas/state")
        case "load_image":
            if "path" in params:
                # Read local file and upload via multipart using subprocess to avoid
                # complexity of building multipart in stdlib
                import subprocess as sp
                path = params["path"]
                result = sp.run(
                    ["curl", "-sf", "-X", "POST", f"{_PAINT_BASE}/api/canvas/load-image",
                     "-F", f"image=@{path}"],
                    capture_output=True, text=True, timeout=30,
                )
                return _json.loads(result.stdout) if result.stdout else {"error": result.stderr}
            return _paint("POST", "/api/canvas/load-image", {"url": params["url"]})
        case "rect":
            return _paint("POST", "/api/draw/rect", {**params, "createdBy": params.get("createdBy", "model")})
        case "ellipse":
            return _paint("POST", "/api/draw/ellipse", {**params, "createdBy": params.get("createdBy", "model")})
        case "arrow":
            return _paint("POST", "/api/draw/arrow", {**params, "createdBy": params.get("createdBy", "model")})
        case "dot":
            return _paint("POST", "/api/draw/dot", {**params, "createdBy": params.get("createdBy", "model")})
        case "line":
            return _paint("POST", "/api/draw/line", {**params, "createdBy": params.get("createdBy", "model")})
        case "text":
            return _paint("POST", "/api/draw/text", {**params, "createdBy": params.get("createdBy", "model")})
        case "bulk":
            ops = params.get("operations", [])
            for op in ops:
                op.setdefault("createdBy", "model")
            return _paint("POST", "/api/canvas/ops", {"operations": ops})
        case "list":
            return {"objects": _paint("GET", "/api/objects")}
        case "update":
            obj_id = params.pop("id")
            return _paint("PATCH", f"/api/objects/{obj_id}", params)
        case "delete":
            return _paint("DELETE", f"/api/objects/{params['id']}")
        case "export_png":
            save_to = params.get("save_to", "/tmp/annotated.png")
            png_bytes = _paint("GET", "/api/export/png", binary=True)
            with open(save_to, "wb") as f:
                f.write(png_bytes)
            return {"saved_to": save_to, "size_bytes": len(png_bytes)}
        case "export_json":
            return _paint("GET", "/api/export/json")
        case "get_canvas_image":
            import base64 as _b64
            from datetime import datetime as _dt
            png_bytes = _paint("GET", "/api/export/png", binary=True)
            s = _paint("GET", "/api/canvas/state")
            data_url = "data:image/png;base64," + _b64.b64encode(png_bytes).decode("utf-8")
            # Save into screenshots/paint_<timestamp>/ so:
            #   - it is readable by the sandbox at /workspace/screenshots/paint_<timestamp>/
            #   - it stays separate from browser step screenshots (screenshots/steps_<timestamp>/)
            _project_root = _os.path.dirname(_os.path.abspath(__file__))
            _ts = _dt.now().strftime("%Y%m%d_%H%M%S_%f")
            _session_dir_host = _os.path.join(_project_root, "screenshots", f"paint_{_ts}")
            _os.makedirs(_session_dir_host, exist_ok=True)
            _filename = params.get("filename", "canvas_export.png")
            _host_path = _os.path.join(_session_dir_host, _filename)
            with open(_host_path, "wb") as f:
                f.write(png_bytes)
            # Sandbox-visible path (screenshots/ is volume-mounted at /workspace/screenshots/)
            _sandbox_path = f"/workspace/screenshots/paint_{_ts}/{_filename}"
            return {
                "saved_to": _sandbox_path,
                "width": s.get("canvas", {}).get("width"),
                "height": s.get("canvas", {}).get("height"),
            }
        # ---- Viewport ----
        case "zoom":
            return _paint("POST", "/api/viewport/zoom", params)
        case "pan":
            return _paint("POST", "/api/viewport/pan", params)
        case "zoom_to_region":
            for k in ("x", "y", "width", "height"):
                if k not in params:
                    return {"error": f"zoom_to_region requires '{k}' parameter"}
            return _paint("POST", "/api/viewport/zoom-to-region", params)
        case "zoom_to_object":
            if "id" not in params:
                return {"error": "zoom_to_object requires 'id' parameter"}
            return _paint("POST", "/api/viewport/zoom-to-object", params)
        case "reset_viewport":
            return _paint("POST", "/api/viewport/reset")
        case "viewport":
            return _paint("GET", "/api/viewport")
        case "set_filters":
            return _paint("POST", "/api/viewport/filters", params)
        # ---- Image crop ----
        case "crop":
            if "shapeId" not in params:
                for k in ("x", "y", "width", "height"):
                    if k not in params:
                        return {"error": f"crop requires '{k}' parameter (or provide 'shapeId' for shape-masked crop)"}
            body = {k: params[k] for k in params}  # pass all params through
            body.setdefault("keepAnnotations", True)
            return _paint("POST", "/api/canvas/crop", body)
        # ---- Cursor ----
        case "cursor_move":
            if "x" not in params or "y" not in params:
                return {"error": "cursor_move requires 'x' and 'y' parameters"}
            return _paint("POST", "/api/viewport/cursor",
                          {"x": params["x"], "y": params["y"],
                           "visible": True, "label": params.get("label", "")})
        case "cursor_hide":
            _paint("DELETE", "/api/viewport/cursor")
            return {"ok": True}
        # ---- Images ----
        case "list_images":
            return _paint("GET", "/api/images")
        case "load_image_by_name":
            return _paint("POST", "/api/images/load", {"filename": params["filename"]})
        case _:
            return {"error": f"Unknown action: {action}. See docstring for valid actions."}

@tool
def run_browser_steps(
    steps: list[dict],
    screenshot_dir: str = "/workspace/screenshots",
    settle_seconds: float = 0.8,
) -> dict:
    """Run a sequence of visible GUI browser steps in the sandbox and save a screenshot after each step.

    Args:
        steps: List of step dicts, e.g.
            {"op":"navigate","url":"https://example.com"}
            {"op":"click","x":100,"y":200}
            {"op":"type","text":"hello"}
            {"op":"hotkey","keys":["enter"]}
            {"op":"scroll","dy":600}
        screenshot_dir: Directory to save screenshots.
        settle_seconds: Default delay between steps.

    Returns:
        Dict with success flag, results per step, and screenshot_dir path.
    """
    # Ensure screenshot directory exists using bash (avoids some os.makedirs permission issues)
    return run_visible_browser_steps(steps, screenshot_dir, settle_seconds)

@tool
def create_skill(skill_name: str, description: str, content: str, parent_skill: str = None) -> dict:
    """Create a new skill in the AIO Sandbox skills directory.
    
    IMPORTANT: Sub-skills should be created under 'master-skill' parent and master-skill must be updated.
    
    Args:
        skill_name: Name of the skill (used for directory name, use kebab-case)
        description: Brief description
        content: Markdown content for the skill
        parent_skill: Parent skill directory name (e.g., 'master-skill' for sub-skills)
        
    Returns:
        Dictionary with success status
    """
    return create_skill_in_sandbox(skill_name, description, content, parent_skill)


@tool
def get_sandbox_info() -> dict:
    """Get information about the AIO Sandbox environment.
    
    Returns:
        Dictionary with sandbox context (home_dir, user, workspace)
    """
    return get_sandbox_context()


@tool
def call_mcp_tool_in_sandbox(tool_name: str, arguments: dict) -> dict:
    """Call an MCP (Model Context Protocol) tool in the AIO Sandbox.
    
    The sandbox provides built-in MCP tools for browser automation, file operations,
    terminal commands, and document conversion.
    
    IMPORTANT: Always consult /workspace/skills/master-skill/mcp-tools/SKILL.md first
    to learn available tools and their argument formats.
    
    Args:
        tool_name: MCP tool name (e.g., 'browser_navigate', 'file_read', 'terminal_execute')
        arguments: Dictionary of tool-specific arguments
        
    Returns:
        Dictionary with tool execution results
        
    Examples:
        # File operations
        call_mcp_tool_in_sandbox('file_read', {'path': '/tmp/data.txt'})
        
        # Terminal commands
        call_mcp_tool_in_sandbox('terminal_execute', {'command': 'ls -la'})
    """
    if tool_name.startswith("browser_"):
        return {
            "success": False,
            "error": (
                "Blocked: MCP browser_* tools are disabled for this agent. "
                "Use run_browser_steps (GUI runner) for browser actions."
            ),
        }
    return call_mcp_tool(tool_name, arguments)
