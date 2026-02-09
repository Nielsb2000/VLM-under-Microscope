---
name: mcp-tools
description: MCP (Model Context Protocol) tools for browser automation, file operations, code execution, and document conversion via the AIO Sandbox MCP hub.
---

# MCP Tools Reference

The AIO Sandbox includes a built-in MCP hub at `/mcp` with comprehensive tooling. Use the `call_mcp_tool_in_sandbox` tool to access these capabilities.

## Tool Categories

### 🌐 Browser Tools
Full web automation with navigation, interaction, form filling, and content extraction.

### 📁 File Operations
Unified file system operations (read, write, search, list, replace).

### 💻 Code & Shell Execution
Execute Python, JavaScript, and bash commands in the sandbox.

### ✏️ Advanced File Editing
Professional file editor with str_replace capabilities.

### 📄 Document Conversion
Convert documents to markdown via markitdown.

---

## 🗂️ Filesystem Note

**CRITICAL**: Before using file operations, consult `/workspace/skills/master-skill/sandbox-filesystem/SKILL.md` for complete filesystem structure.

**Quick Reference:**
- **User files** → `/home/gem` (NOT `/workspace`)
- **Skills** → `/workspace/skills`
- **Temp** → `/tmp`

Always search `/home/gem` when looking for user's project files.

---

## Browser Tools

### browser_navigate
Navigate to a URL.

**Arguments:**
```python
{"url": "https://example.com"}
```

**Example:**
```python
call_mcp_tool_in_sandbox('browser_navigate', {
    'url': 'https://github.com/trending'
})
```

### browser_get_clickable_elements
Get all clickable/hoverable/selectable elements with their indices.
**Call this ONCE before clicking to get element indices.**

**Arguments:**
```python
{}
```

**Example:**
```python
result = call_mcp_tool_in_sandbox('browser_get_clickable_elements', {})
# Returns elements with index numbers you can click
```

### browser_click
Click an element by its index (from `browser_get_clickable_elements`).

**Arguments:**
```python
{"index": 0}  # NOT selector - use index from browser_get_clickable_elements
```

**Example:**
```python
# First, get elements
elements = call_mcp_tool_in_sandbox('browser_get_clickable_elements', {})
# Then click by index
call_mcp_tool_in_sandbox('browser_click', {'index': 0})
```

### browser_form_input_fill
Fill an input field by index or selector.

**Arguments:**
```python
{
    "value": "text to enter",
    "index": 0,  # OR use selector
    "clear": True  # Optional: clear existing text
}
```

**Example:**
```python
call_mcp_tool_in_sandbox('browser_form_input_fill', {
    'index': 1,
    'value': 'search query',
    'clear': True
})
```

### browser_get_text
Get the text content of the current page.

**Arguments:**
```python
{}
```

**Example:**
```python
result = call_mcp_tool_in_sandbox('browser_get_text', {})
text = result['result']['content']
```

### browser_get_markdown
Get the markdown-formatted content of the current page.

**Arguments:**
```python
{}
```

**Example:**
```python
result = call_mcp_tool_in_sandbox('browser_get_markdown', {})
markdown = result['result']['content']
```

### browser_screenshot
Take a screenshot of the page or specific element.

**Arguments:**
```python
{
    "name": "screenshot",  # Optional
    "fullPage": False,     # Optional: capture entire page
    "selector": "div.main", # Optional: CSS selector
    "index": 0,            # Optional: element index
    "highlight": False     # Optional: highlight element
}
```

**Example:**
```python
call_mcp_tool_in_sandbox('browser_screenshot', {
    'name': 'homepage',
    'fullPage': True
})
```

### browser_scroll
Scroll the page.

**Arguments:**
```python
{"amount": 500}  # Positive = down, negative = up, omit = scroll to bottom
```

**Example:**
```python
call_mcp_tool_in_sandbox('browser_scroll', {'amount': -300})
```

### browser_read_links
Get all links on the current page.

**Arguments:**
```python
{}
```

**Returns:**
```python
{"links": [{"text": "Link Text", "href": "https://..."}]}
```

### browser_new_tab
Open a new tab.

**Arguments:**
```python
{"url": "https://example.com"}
```

### browser_tab_list
Get list of all tabs.

**Arguments:**
```python
{}
```

**Returns:**
```python
{"tabList": [{"index": 0, "active": True, "title": "...", "url": "..."}]}
```

### browser_switch_tab
Switch to a specific tab by index.

**Arguments:**
```python
{"index": 1}
```

### browser_press_key
Press a keyboard key.

**Arguments:**
```python
{"key": "Enter"}  # Enter, Tab, Escape, ArrowDown, etc.
```

### browser_evaluate
Execute JavaScript in the browser console.

**Arguments:**
```python
{"script": "() => { return document.title; }"}
```

### browser_go_back / browser_go_forward
Navigate browser history.

**Arguments:**
```python
{}
```

### browser_close
Close the browser when done.

**Arguments:**
```python
{}
```

---

## File Operations (Unified Tool)

### sandbox_file_operations
**Single unified tool for all file operations.** `/tmp` and `/home/gem` are fully accessible.

**Actions:** `"read"`, `"write"`, `"replace"`, `"search"`, `"find"`, `"list"`

#### Read a file
```python
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'read',
    'path': '/workspace/config.py',
    'start_line': 0,   # Optional: line range
    'end_line': 50     # Optional
})
```

#### Write to a file
```python
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'write',
    'path': '/tmp/output.txt',
    'content': 'File content here',
    'append': False  # Optional: append mode
})
```

#### Replace text in file
```python
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'replace',
    'path': '/workspace/config.py',
    'content': 'DEBUG = False',  # Text to find
    'target': 'DEBUG = True'     # Text to replace with
})
```

#### Search for text in files (regex)
```python
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'search',
    'path': '/workspace',
    'content': 'import.*pandas',  # Regex pattern
    'recursive': True
})
```

#### Find files by pattern (glob)
```python
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'find',
    'path': '/workspace',
    'pattern': '*.py',
    'recursive': True
})
```

#### List directory contents
```python
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'list',
    'path': '/workspace/skills',
    'show_hidden': False,
    'file_types': ['.md', '.py']  # Optional filter
})
```

---

## Code & Shell Execution

### sandbox_execute_code
Execute Python or JavaScript code.

**Arguments:**
```python
{
    "code": "print('hello')",
    "language": "python",  # or "javascript"
    "timeout": 30  # Optional
}
```

**Example:**
```python
call_mcp_tool_in_sandbox('sandbox_execute_code', {
    'code': 'import sys; print(sys.version)',
    'language': 'python'
})
```

### sandbox_execute_bash
Execute shell commands with session management.

**Arguments:**
```python
{
    "cmd": "ls -la /workspace",
    "cwd": "/tmp",  # Optional working directory
    "new_session": False,  # Optional: create new session
    "timeout": 30  # Optional
}
```

**Example:**
```python
call_mcp_tool_in_sandbox('sandbox_execute_bash', {
    'cmd': 'pip list | grep pandas',
    'cwd': '/workspace'
})
```

---

## Advanced File Editor

### sandbox_str_replace_editor
Professional file editor with powerful editing capabilities.

**Commands:** `"view"`, `"create"`, `"str_replace"`, `"insert"`, `"undo_edit"`

#### View file or range
```python
call_mcp_tool_in_sandbox('sandbox_str_replace_editor', {
    'command': 'view',
    'path': '/workspace/main.py',
    'view_range': [1, 50]  # Optional: line range
})
```

#### Create new file
```python
call_mcp_tool_in_sandbox('sandbox_str_replace_editor', {
    'command': 'create',
    'path': '/workspace/new_file.py',
    'file_text': 'def hello():\n    print("Hello")'
})
```

#### Replace string
```python
call_mcp_tool_in_sandbox('sandbox_str_replace_editor', {
    'command': 'str_replace',
    'path': '/workspace/config.py',
    'old_str': 'DEBUG = False',
    'new_str': 'DEBUG = True'
})
```

#### Insert at line
```python
call_mcp_tool_in_sandbox('sandbox_str_replace_editor', {
    'command': 'insert',
    'path': '/workspace/main.py',
    'insert_line': 10,
    'new_str': '    # New comment\n'
})
```

---

## Document Conversion

### sandbox_convert_to_markdown
Convert documents/web pages to markdown via markitdown.

**Arguments:**
```python
{"uri": "file:///workspace/document.pdf"}
# Or: "uri": "https://example.com"
```

**Example:**
```python
call_mcp_tool_in_sandbox('sandbox_convert_to_markdown', {
    'uri': 'file:///tmp/report.docx'
})
```

---

## Sandbox Info Tools

### sandbox_get_context
Get sandbox environment information.

**Arguments:**
```python
{}
```

**Returns:** Version, home directory, environment details.

### sandbox_get_packages
Get installed packages.

**Arguments:**
```python
{"language": "python"}  # or "nodejs", or null for both
```

---

## Common Patterns

### Web Scraping Workflow
```python
# 1. Navigate
call_mcp_tool_in_sandbox('browser_navigate', {
    'url': 'https://news.ycombinator.com'
})

# 2. Get text content
result = call_mcp_tool_in_sandbox('browser_get_markdown', {})
content = result['result']['content']

# 3. Save to file
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'write',
    'path': '/workspace/scraped.md',
    'content': content
})

# 4. Take screenshot for reference
call_mcp_tool_in_sandbox('browser_screenshot', {
    'name': 'hn_homepage',
    'fullPage': True
})
```

### Form Automation
```python
# Navigate
call_mcp_tool_in_sandbox('browser_navigate', {
    'url': 'https://example.com/login'
})

# Get clickable elements ONCE
elements = call_mcp_tool_in_sandbox('browser_get_clickable_elements', {})

# Fill form by index (from elements list)
call_mcp_tool_in_sandbox('browser_form_input_fill', {
    'index': 0,  # Email field
    'value': 'user@example.com',
    'clear': True
})

call_mcp_tool_in_sandbox('browser_form_input_fill', {
    'index': 1,  # Password field
    'value': 'secret123'
})

# Click submit button by index
call_mcp_tool_in_sandbox('browser_click', {'index': 2})

# Verify success
call_mcp_tool_in_sandbox('browser_get_text', {})
```

### File Processing Pipeline
```python
# List Python files
files = call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'find',
    'path': '/workspace',
    'pattern': '*.py',
    'recursive': True
})

# Search for specific imports
matches = call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'search',
    'path': '/workspace',
    'content': 'import pandas',
    'recursive': True
})

# Replace text across files
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'replace',
    'path': '/workspace/utils.py',
    'content': 'old_function_name',
    'target': 'new_function_name'
})
```

### Code Execution & Testing
```python
# Execute Python code
result = call_mcp_tool_in_sandbox('sandbox_execute_code', {
    'code': '''
import json
data = {"status": "ok", "count": 42}
print(json.dumps(data, indent=2))
''',
    'language': 'python'
})

# Run bash commands
call_mcp_tool_in_sandbox('sandbox_execute_bash', {
    'cmd': 'pytest tests/ -v',
    'cwd': '/workspace',
    'timeout': 60
})

# Check installed packages
call_mcp_tool_in_sandbox('sandbox_get_packages', {
    'language': 'python'
})
```

---

## Important Notes

1. **Always use `call_mcp_tool_in_sandbox` wrapper** - Never call MCP tools directly
2. **Browser clicking requires indices** - Use `browser_get_clickable_elements` ONCE, then click by index
3. **File operations are unified** - Use `sandbox_file_operations` with different actions
4. **File paths** - Use absolute paths; `/tmp` and `/home/gem` fully accessible
5. **Error handling** - Always check `success` field in returned dictionaries
6. **Browser state persists** - Navigation/actions affect persistent browser instance
7. **Screenshots** - Returned as base64-encoded images
8. **Code execution** - Runs in isolated sandbox environment
9. **Timeouts** - Default 30s for most operations

## Use Cases

- **Web automation**: Navigate sites, fill forms, extract data, take screenshots
- **Data collection**: Scrape multiple pages, extract structured content
- **File processing**: Batch operations, search/replace across files
- **Document conversion**: PDF/DOCX to markdown for analysis
- **Code execution**: Run tests, process data, validate scripts
- **System tasks**: Execute builds, manage files, run commands
