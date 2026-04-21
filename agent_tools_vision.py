# agent_tools_vision.py
import base64
import io
import json
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from sandbox_core_functions import execute_python_code
from agent_sandbox import Sandbox
from PIL import Image, ImageDraw, ImageFont
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def _sandbox_path_to_host(image_path: str) -> str | None:
    """
    Translates a sandbox path to a host path if the directory is volume-mounted.
    Returns None if no known mapping exists.
    Volume mounts (from docker-compose.yml):
      /workspace/screenshots/ -> {project_root}/screenshots/
      /workspace/skills/      -> {project_root}/skills/
    """
    mappings = {
        "/workspace/screenshots/": os.path.join(_PROJECT_ROOT, "screenshots") + "/",
        "/workspace/skills/": os.path.join(_PROJECT_ROOT, "skills") + "/",
    }
    for sandbox_prefix, host_prefix in mappings.items():
        if image_path.startswith(sandbox_prefix):
            return host_prefix + image_path[len(sandbox_prefix):]
    return None


def sandbox_image_to_data_url(image_path: str) -> dict:
    """
    Reads an image and returns a data: URL.

    For paths under volume-mounted directories (/workspace/screenshots/, etc.),
    reads directly on the host to avoid printing base64 to sandbox stdout
    (which would bloat the conversation history with ~1M tokens).
    Falls back to sandbox-side execution for other sandbox paths.
    """
    import mimetypes

    host_path = _sandbox_path_to_host(image_path)
    if host_path and os.path.isfile(host_path):
        try:
            with open(host_path, "rb") as f:
                data = f.read()
            mime = mimetypes.guess_type(host_path)[0] or "image/png"
            b64 = base64.b64encode(data).decode("utf-8")
            return {"success": True, "mime": mime, "data_url": f"data:{mime};base64,{b64}"}
        except Exception as e:
            return {"success": False, "error": f"Host read failed: {e}"}

    # Fallback: read via sandbox execution (avoid for large images — stdout goes to history)
    py = f"""
import base64, mimetypes, json
from pathlib import Path

p = Path({image_path!r})
data = p.read_bytes()
mime = mimetypes.guess_type(str(p))[0] or "image/png"
b64 = base64.b64encode(data).decode("utf-8")
print(json.dumps({{"mime": mime, "data_url": f"data:{{mime}};base64,{{b64}}"}}))
"""
    r = execute_python_code(py, timeout=30)
    if not r.get("success"):
        return {"success": False, "error": r.get("error", "Failed to encode image")}

    try:
        payload = json.loads(r["output"].strip().splitlines()[-1])
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": f"Failed to parse encoder output: {e}", "raw_output": r.get("output")}


def make_analyze_sandbox_image_tool(llm):
    """
    Returns a callable tool function that closes over (captures) the llm.
    DeepAgents can register this function as a tool.
    """
    @tool 
    def analyze_sandbox_image(image_path: str, question: str = "Describe this image.") -> dict:
        """Analyze an image stored inside the sandbox filesystem using a vision-capable model.

        Args:
            image_path: Path to an image file inside the sandbox (e.g. /workspace/pizza_not_pizza/001.png)
            question: What you want to know about the image

        Returns:
            Dict with keys: success, answer, (and image_path/mime on success)
        """
        # 1) Encode sandbox image to a data URL
        encoded = sandbox_image_to_data_url(image_path)
        if not encoded.get("success"):
            return encoded

        data_url = encoded["data_url"]

        # 2) Ask the vision-capable model with multimodal content
        msg = HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])

        try:
            resp = llm.invoke([msg])
            return {
                "success": True,
                "image_path": image_path,
                "mime": encoded.get("mime"),
                "answer": resp.content,
            }
        except Exception as e:
            return {"success": False, "error": f"Vision model call failed: {e}", "image_path": image_path}

    # Give the inner function a stable name for tool registries
    return analyze_sandbox_image


def _overlay_coordinate_grid(png_bytes: bytes, grid_spacing: int = 100) -> bytes:
    """
    Draw a labelled coordinate grid over a PNG image.

    Grid lines are drawn every `grid_spacing` pixels.  Each intersection is
    labelled with its (x, y) pixel coordinate so the vision model can read off
    exact positions directly from the image.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    line_color = (0, 200, 255, 120)       # cyan, semi-transparent
    label_bg   = (0, 0, 0, 160)           # dark background behind labels
    label_fg   = (0, 220, 255, 255)       # bright cyan text

    # Try to load a small bitmap font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    xs = list(range(0, w, grid_spacing))
    ys = list(range(0, h, grid_spacing))

    # Vertical lines
    for x in xs:
        draw.line([(x, 0), (x, h)], fill=line_color, width=1)

    # Horizontal lines
    for y in ys:
        draw.line([(0, y), (w, y)], fill=line_color, width=1)

    # Labels at every intersection
    for x in xs:
        for y in ys:
            label = f"({x},{y})"
            bbox = draw.textbbox((x + 2, y + 2), label, font=font)
            # Small dark background rect so text is readable over any content
            draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill=label_bg)
            draw.text((x + 2, y + 2), label, fill=label_fg, font=font)

    composited = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    composited.save(buf, format="PNG")
    return buf.getvalue()


_CURSOR_RADIUS = 10
_CURSOR_COLOR = (220, 30, 30, 220)   # red, slightly transparent
_CURSOR_OUTLINE = (255, 255, 255, 255)  # white ring for contrast


def _overlay_cursor(png_bytes: bytes, x: int, y: int) -> bytes:
    """Draw a red dot at (x, y) on the screenshot PNG and return updated bytes."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r = _CURSOR_RADIUS
    draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=_CURSOR_OUTLINE)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=_CURSOR_COLOR)
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    composited.save(buf, format="PNG")
    return buf.getvalue()


def _png_to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")


def make_screenshot_and_ask_tool(llm):
    """
    Returns a tool that captures a live browser screenshot from the sandbox,
    base64-encodes it, and sends it to the vision model with a question.

    Typical use-case:
        - Take a screenshot of what is currently visible in the sandbox browser
        - Ask the model "what are the (x, y) coordinates of the duck?"
        - Use the returned coordinates in a subsequent run_browser_steps move/click
    """
    @tool
    def screenshot_and_ask(
        question: str = "Describe what you see.",
        cursor_x: int = None,
        cursor_y: int = None,
    ) -> dict:
        """Take a live screenshot of the sandbox browser and ask the vision model a question about it.

        Two images are sent to the model:
          1. Raw screenshot with red cursor dot (if cursor position provided)
          2. Same screenshot with coordinate grid overlay and red cursor dot

        This is the primary way to give the agent visual awareness of the current browser state.
        Use it to locate UI elements, read text, or understand the current page before acting.

        Example questions:
          - "What are the pixel (x, y) coordinates of the duck in this image?"
          - "List all buttons visible and their approximate positions."
          - "What text is shown in the center of the screen?"

        Args:
            question: The question to ask the vision model about the screenshot.
            cursor_x: Optional current cursor X pixel position — draws a red dot on both images.
            cursor_y: Optional current cursor Y pixel position — draws a red dot on both images.

        Returns:
            Dict with keys: success, answer, width, height (and error on failure).
        """
        try:
            sandbox_url = os.getenv("SANDBOX_BASE_URL", "http://localhost:8080")
            client = Sandbox(base_url=sandbox_url)

            # Capture raw PNG bytes from the sandbox browser
            screenshot_data = client.browser.screenshot()
            png_bytes = b"".join(screenshot_data)

            # Get image dimensions
            img = Image.open(io.BytesIO(png_bytes))
            width, height = img.size

            # Apply red cursor dot to raw screenshot (if position known)
            raw_with_cursor = png_bytes
            if cursor_x is not None and cursor_y is not None:
                raw_with_cursor = _overlay_cursor(png_bytes, cursor_x, cursor_y)

            # Build coordinate-grid overlay version, then apply cursor dot on top
            grid_png_bytes = _overlay_coordinate_grid(raw_with_cursor)

            # Encode both images as data URLs
            raw_data_url  = _png_to_data_url(raw_with_cursor)
            grid_data_url = _png_to_data_url(grid_png_bytes)

            # Build prompt describing both images
            cursor_note = (
                f" The current cursor position is marked with a red dot at ({cursor_x},{cursor_y})."
                if cursor_x is not None and cursor_y is not None else ""
            )
            full_question = (
                f"You are given two images of the same browser screenshot ({width}x{height} pixels).{cursor_note} "
                f"Image 1 is the raw screenshot with the cursor dot. "
                f"Image 2 has a cyan coordinate grid overlaid with labels showing (x,y) pixel positions "
                f"every 100 pixels — use these labels to pinpoint exact coordinates. "
                f"Pixel (0,0) is top-left, ({width},{height}) is bottom-right. "
                + question
            )

            msg = HumanMessage(content=[
                {"type": "text", "text": full_question},
                {"type": "image_url", "image_url": {"url": raw_data_url}},
                {"type": "image_url", "image_url": {"url": grid_data_url}},
            ])

            resp = llm.invoke([msg])
            return {
                "success": True,
                "answer": resp.content,
                "width": width,
                "height": height,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    return screenshot_and_ask


def make_move_and_verify_tool(llm):
    """
    Returns a tool that moves the cursor to (x, y) and immediately self-verifies
    by feeding the resulting screenshot back to the vision model.

    The model checks whether the red cursor dot landed on the intended target
    (5px tolerance). No separate judge step needed — correction hints are included
    in the return value so the agent can retry in one loop.
    """
    @tool
    def move_and_verify(
        x: int,
        y: int,
        intent: str,
        settle_seconds: float = 0.5,
    ) -> dict:
        """Move the cursor to (x, y) and self-verify it landed on the correct target.

        After the move, a screenshot is taken automatically. The resulting image
        (with red cursor dot + coordinate grid) is sent to the vision model which
        checks whether the cursor is within 5px of the target described by `intent`.

        Use this instead of run_browser_steps for cursor moves that require accuracy.
        run_browser_steps is still preferred for navigate/type/click/scroll/hotkey.

        Args:
            x: Target X pixel coordinate.
            y: Target Y pixel coordinate.
            intent: Description of what the cursor should be on, e.g. "the Sign Up button".
            settle_seconds: Time to wait after the move before taking the screenshot.

        Returns:
            Dict with keys:
              success (bool)
              correct (bool)        — True if cursor is within 5px of the target
              judgement (str)       — Full model verdict + reason + hint
              suggested_x, suggested_y — Parsed corrected coordinates (if INCORRECT)
              width, height         — Screenshot dimensions
        """
        import time
        import re
        from datetime import datetime
        from sandbox_browser_functions import browser_move_to, write_binary_file

        try:
            # Execute the move
            res = browser_move_to(x, y)
            if not res.get("success"):
                return {"success": False, "error": res.get("error"), "correct": False}

            time.sleep(settle_seconds)

            # Capture post-move screenshot
            sandbox_url = os.getenv("SANDBOX_BASE_URL", "http://localhost:8080")
            client = Sandbox(base_url=sandbox_url)
            screenshot_data = client.browser.screenshot()
            png_bytes = b"".join(screenshot_data)

            img = Image.open(io.BytesIO(png_bytes))
            width, height = img.size

            # Apply cursor dot then grid overlay
            with_cursor = _overlay_cursor(png_bytes, x, y)
            with_grid   = _overlay_coordinate_grid(with_cursor)

            # Persist both images to the screenshots directory
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            session_dir = f"/workspace/screenshots/move_verify_{ts}"
            client.shell.exec_command(command=f"mkdir -p '{session_dir}' && chmod 777 '{session_dir}'")
            write_binary_file(client, f"{session_dir}/cursor.png", with_cursor)
            write_binary_file(client, f"{session_dir}/cursor_grid.png", with_grid)
            with_grid   = _overlay_coordinate_grid(with_cursor)

            raw_data_url  = _png_to_data_url(with_cursor)
            grid_data_url = _png_to_data_url(with_grid)

            verify_prompt = (
                f"You are verifying a cursor move. The task was: \"{intent}\". "
                f"The cursor is now at ({x},{y}) — shown as a red dot in both images. "
                f"Image 1 is the raw screenshot with the red cursor dot. "
                f"Image 2 has a cyan coordinate grid with labels every 100px for reference. "
                f"Screenshot is {width}x{height} px; (0,0) is top-left. "
                f"\n\nIs the red dot within 10 pixels of the correct target? "
                f"Reply in this exact format:\n"
                f"VERDICT: CORRECT or INCORRECT\n"
                f"REASON: <one sentence>\n"
                f"HINT: <if INCORRECT, give the exact corrected (x,y); if CORRECT, write 'None'>"
            )

            msg = HumanMessage(content=[
                {"type": "text", "text": verify_prompt},
                {"type": "image_url", "image_url": {"url": raw_data_url}},
                {"type": "image_url", "image_url": {"url": grid_data_url}},
            ])

            resp = llm.invoke([msg])
            text = resp.content
            correct = "VERDICT: CORRECT" in text.upper()

            # Try to parse a suggested coordinate from the HINT line
            suggested_x = suggested_y = None
            if not correct:
                m = re.search(r"\((\d+)\s*,\s*(\d+)\)", text)
                if m:
                    suggested_x, suggested_y = int(m.group(1)), int(m.group(2))

            return {
                "success": True,
                "correct": correct,
                "judgement": text,
                "moved_to_x": x,
                "moved_to_y": y,
                "suggested_x": suggested_x,
                "suggested_y": suggested_y,
                "screenshot": f"{session_dir}/cursor.png",
                "screenshot_grid": f"{session_dir}/cursor_grid.png",
                "width": width,
                "height": height,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "correct": False}

    return move_and_verify
