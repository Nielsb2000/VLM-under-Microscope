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
