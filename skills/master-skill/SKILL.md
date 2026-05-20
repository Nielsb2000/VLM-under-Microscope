---
name: master-skill
description: Main reference guide and contents page for all available skills. Start here to discover specialized skill areas.
---

# Master Skills Reference Guide

This is the central routing authority for all skills.

The agent MUST use this file to determine:
- Which skill to consult
- Which tool to use
- Which tool to avoid

---

# 🚨 GLOBAL TOOL POLICY

## Browser Automation Rule

When performing browser interactions:

Before the first browser interaction, always run a bash command to ensure a writable screenshot directory exists:

Persistent directory (only if mounted and writable):
  - `/workspace/screenshots`

If `/workspace/screenshots` is used, it must be verified writable before execution.

---

- If the task requires **visible step-by-step interaction**:
  - Use: `run_browser_steps`
  - Do NOT use MCP `browser_*` tools

- If the task requires only text extraction:
  - MCP browser tools may be used

The GUI runner:
- Executes visible browser operations
- Saves step screenshots
- Uses the `/screenshot` endpoint (NOT `/vnc/screenshot`)
- Is the authoritative browser execution tool

---

# Available Skills

---

## 🖱️ `./sandbox-browser/SKILL.md`
Visible GUI browser automation in the AIO Sandbox.

Use for:
- `run_browser_steps` step schemas
- Supported operations (move, click, type, hotkey, scroll, drag, navigate)
- Screenshot endpoint rules (`/workspace/screenshot`)
- Screenshot directory requirements
- Browser automation troubleshooting

---

## 🗂️ `./sandbox-filesystem/SKILL.md`
Sandbox directory layout and path rules.

Key directories:
- `/home/gem` → user files
- `/workspace/skills` → skill docs
- `/workspace/screenshots` → persistent screenshots (if mounted)

Use for:
- Path resolution
- File location questions

---

## 💻 `./bash-scripting/SKILL.md`
Shell commands and automation.

---

## 🐍 `./python-programming/SKILL.md`
Python logic and computation tasks.

---

## 🌐 `./web-network/SKILL.md`
HTTP requests and network operations.

---

## 🧠 `./mcp-tools/SKILL.md`
MCP tools for:
- File operations
- Code execution
- Document conversion
- Limited browser extraction (policy-restricted)

---

## 📁 `./file-management/SKILL.md`
File organization and manipulation.

---

## � `./sem-service/SKILL.md`
SEM Service annotation canvas — live browser app at http://localhost:3000.

Use for:
- Zooming / panning the view in the browser
- Loading an image into the paint app
- Drawing annotations (rect, ellipse, arrow, dot, text)
- Cropping to a bounding box (true pixel zoom)
- Moving the model cursor on the screen
- Reading what annotations are currently on the canvas
- Exporting PNG or JSON from the canvas
- Randomising image filters for a case-study run (`randomize_filters`)
- Adjusting brightness / contrast / saturation (`set_filters`)

**Trigger phrases:** "zoom in on", "draw a box around", "load this image in the paint app",
"what's on the canvas", "annotate", "crop to", "highlight", "point at",
"randomize", "adjust brightness", "refine image quality"

---

## 📊 `./sem-histogram-eval/SKILL.md`
SEM image quality evaluation via brightness histogram comparison.

Use for:
- Computing the objective SEM Histogram Error score after agent refinement
- Understanding the Wasserstein + clipping metric formula
- Running `sem_histogram_error.py` from inside the sandbox
- Saving result histograms to `/workspace/histograms/result/`

**When to use:** after the agent has finished iteratively adjusting
brightness/contrast/saturation and declared it is satisfied with the image
quality — call this as the final evaluation step.

**Script location:** `/workspace/skills/master-skill/sem-histogram-eval/sem_histogram_error.py`

**Trigger phrases:** "evaluate the result", "compute the histogram error",
"run the evaluation", "how good is the image quality", "score the result"

---

# How To Use

1. Identify the domain.
2. Open the corresponding skill.
3. Follow its tool rules.
4. Respect the browser automation policy.
5. Cite which skill was used.

---

# Notes

- This file governs routing decisions.
- Browser automation must follow the GUI runner rule.
- The `/vnc/` path is reserved for noVNC static assets.
- Screenshots must use the `/screenshot` endpoint.
- Skills define allowed and disallowed tool usage.
