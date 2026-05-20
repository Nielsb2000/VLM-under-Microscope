---
name: sandbox-browser
description: Visible GUI browser automation for the AIO Sandbox. Uses run_browser_steps and saves screenshots to SANDBOX_SCREENSHOT_DIR (/workspace/screenshots).
---

# Sandbox Browser Skill

This skill governs visible, step-by-step GUI browser automation inside the AIO Sandbox.

Use this skill when the task requires:
- Moving the mouse
- Clicking elements
- Typing text
- Sending hotkeys
- Scrolling
- Drag operations
- Navigating through the browser UI
- Capturing screenshots after each step

---

# Tool Policy

## Visible Interaction (Authoritative Path)

When the task requires visible GUI interaction:

- Use: run_browser_steps
- Do NOT use MCP browser_* tools

The GUI runner:
- Executes real visible browser actions
- Captures a screenshot after every step
- Returns structured per-step results
- Saves screenshots to SANDBOX_SCREENSHOT_DIR

## Text Extraction Only

If the task only requires reading content and does not require visible interaction or screenshots, MCP browser tools may be used (if allowed by master policy).

---

# Screenshot Directory (Authoritative)

Screenshots MUST be saved to the sandbox screenshot directory defined by:

Environment variable:
SANDBOX_SCREENSHOT_DIR

In this system, docker-compose defines:

SANDBOX_SCREENSHOT_DIR=/workspace/screenshots

This directory is host-mounted:

./screenshots  →  /workspace/screenshots

Screenshots are expected to persist in that location.

---

# Screenshot Directory Preflight (Required)

Before performing browser steps, ensure the configured screenshot directory exists and is writable:

mkdir -p /workspace/screenshots && touch /workspace/screenshots/.write_test && rm /workspace/screenshots/.write_test

If this fails with Permission denied:

Fix host bind mount permissions:

mkdir -p ./screenshots
sudo chown -R 1000:1000 ./screenshots
sudo chmod -R 777 ./screenshots
docker compose down && docker compose up -d

No alternative screenshot directories are permitted.

---

# Screenshot Endpoint

Important:

/vnc/ is reserved for noVNC static assets.

Do NOT use:
http://localhost:8080/vnc/screenshot

Use:
http://localhost:8080/screenshot

Verification:

curl -v http://127.0.0.1:8080/screenshot -o /workspace/screenshots/test.png

Expected:
HTTP 200 with Content-Type: image/png

---

# Visual Awareness: Screenshot + Vision Query

To understand what is currently visible in the browser before acting, use the `screenshot_and_ask` tool.

This tool:
- Captures a live PNG screenshot of the sandbox browser
- Base64-encodes it and sends it directly to the vision model
- Returns the model's answer as a string

Use it to locate elements by visual appearance before issuing move/click steps.

## Tool Signature

screenshot_and_ask(question: str) -> dict

Returns: { success: bool, answer: str, width: int, height: int }

## Typical Workflow

1. Call screenshot_and_ask to get coordinates of a visual target:

answer = screenshot_and_ask(
    question="What are the pixel (x, y) coordinates of the duck in this image?"
)
# answer["answer"] → "The duck is at approximately (312, 245)"

2. Parse the coordinates from the answer.

3. Use run_browser_steps to move or click:

run_browser_steps([{"op": "move", "x": 312, "y": 245}])

## Important Notes

- The model is told the image dimensions and that (0,0) is top-left automatically.
- Prefer specific questions: "What are the pixel coordinates of X?" over vague ones.
- screenshot_and_ask is read-only — it does NOT interact with the browser.
- Always call screenshot_and_ask BEFORE acting when you need visual context.

---

# Supported Operations

The step dispatcher supports:

navigate  → Open URL via address bar (ctrl+l, type, enter)
click     → Click at x,y
move      → Move cursor to x,y
type      → Type text
hotkey    → Send key combination
scroll    → Scroll by dy/dx
drag      → Drag to x,y

Recommended aliases for robustness:

mousemove → move
dragto    → drag

Dispatcher example:

elif op in ("move", "mousemove"):
    res = browser_move_to(step["x"], step["y"])

elif op in ("drag", "dragto"):
    res = browser_drag_to(step["x"], step["y"])

---

# Example Usage

Move to top-right and capture screenshot:

steps = [
  {"op": "move", "x": 1279, "y": 0, "settle_seconds": 0.2}
]

run_browser_steps(steps, screenshot_dir="/workspace/screenshots")

Navigate and search:

steps = [
  {"op": "navigate", "url": "https://www.google.com", "settle_seconds": 2.0},
  {"op": "type", "text": "wikipedia", "use_clipboard": True},
  {"op": "hotkey", "keys": ["enter"]}
]

run_browser_steps(steps, screenshot_dir="/workspace/screenshots")

---

# Common Failure Modes

Unknown op: mousemove  
→ Use "move" or add alias support.

404 on /vnc/screenshot  
→ Use /screenshot instead.

Permission denied creating screenshot directory  
→ Fix host permissions for ./screenshots.

Clicks miss target  
→ Increase settle_seconds.
→ Adjust coordinates.
→ Inspect saved screenshots.
