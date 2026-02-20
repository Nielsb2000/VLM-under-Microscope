---
name: mcp-tools
description: MCP (Model Context Protocol) tools for file operations, code execution, and document conversion via the AIO Sandbox MCP hub.
---

# MCP Tools Reference

The AIO Sandbox includes a built-in MCP hub at `/mcp`. Use the `call_mcp_tool_in_sandbox` tool to access these capabilities.

---

# 🚨 BROWSER AUTOMATION POLICY (CRITICAL)

When the agent must perform **visible step-by-step browser actions** (navigate, click, type, scroll, drag, hotkeys, etc.):

ALWAYS use:
    run_browser_steps(...)

NEVER use MCP browser_* tools for multi-step flows, including:

- browser_navigate
- browser_click
- browser_form_input_fill
- browser_scroll
- browser_press_key
- browser_evaluate
- browser_new_tab
- browser_switch_tab
- browser_go_back
- browser_go_forward
- browser_close
- any other interactive browser_* MCP action

Rationale:
The GUI runner is the authoritative tool for visible browser interaction and per-step screenshots.  
MCP browser tools must NOT be used for step-by-step automation in this agent.

---

# Screenshot Storage (GUI Runner)

The GUI runner saves screenshots after each step.

Default directory:
    /screenshots/browser_steps

This directory is mounted persistently via Docker:
    Host: ./sandbox_screenshots
    Container: /screenshots

Do NOT write screenshots to:
    /home/gem/screenshots

Use /screenshots for persistent artifacts.

---

# Filesystem Note

Before using file operations, consult:

    /workspace/skills/master-skill/sandbox-filesystem/SKILL.md

Quick Reference:

- User files → /home/gem
- Skills → /workspace/skills
- Temp → /tmp
- Persistent artifacts → /screenshots

Important:
Permissions on /home/gem may vary depending on container runtime.
Prefer /screenshots for generated files and artifacts.

---

# Allowed MCP Usage

MCP tools are allowed for:

- File operations
- Code execution
- Shell commands
- Document conversion
- Read-only browser extraction

---

# MCP Browser Tools (Extraction Only)

Allowed only for non-interactive extraction:

- browser_get_text
- browser_get_markdown
- browser_screenshot (single capture only)

Do NOT use MCP browser tools for interactive flows.

---

# File Operations

Tool: sandbox_file_operations

Actions:
- read
- write
- replace
- search
- find
- list

Example:

call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'read',
    'path': '/workspace/config.py'
})

---

# Code Execution

Tool: sandbox_execute_code

Example:

call_mcp_tool_in_sandbox('sandbox_execute_code', {
    'code': 'print("hello")',
    'language': 'python'
})

Tool: sandbox_execute_bash

Example:

call_mcp_tool_in_sandbox('sandbox_execute_bash', {
    'cmd': 'ls -la',
    'cwd': '/workspace'
})

---

# Advanced File Editing

Tool: sandbox_str_replace_editor

Example:

call_mcp_tool_in_sandbox('sandbox_str_replace_editor', {
    'command': 'str_replace',
    'path': '/workspace/config.py',
    'old_str': 'DEBUG = False',
    'new_str': 'DEBUG = True'
})

---

# Document Conversion

Tool: sandbox_convert_to_markdown

Example:

call_mcp_tool_in_sandbox('sandbox_convert_to_markdown', {
    'uri': 'file:///tmp/report.docx'
})

---

# Important Notes

1. Always use call_mcp_tool_in_sandbox wrapper.
2. Never call MCP browser_* tools for interactive flows.
3. Use run_browser_steps for visible browser automation.
4. Prefer /screenshots for persistent artifacts.
5. Always check the "success" field in tool responses.
