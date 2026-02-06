---
name: sandbox-filesystem
description: Complete guide to the AIO Sandbox filesystem structure, directory locations, permissions, and file access patterns.
---

# Sandbox Filesystem Reference

**CRITICAL**: Understanding the sandbox filesystem is essential for all file operations, whether using MCP tools, shell commands, or Python scripts.

---

## 🗂️ Directory Structure

### `/home/gem` - **PRIMARY USER DIRECTORY**
**This is where user's actual project files and code live.**

- **Purpose**: Main working directory for user files
- **Access**: Full read/write permissions
- **Contains**: 
  - User's Python scripts (e.g., `hello_world.py`, `config.py`)
  - Project code and data files
  - User-created content
- **Examples**:
  - `/home/gem/hello_world.py`
  - `/home/gem/data/dataset.csv`
  - `/home/gem/scripts/analysis.py`

**When searching for user files, ALWAYS check `/home/gem` first.**

### `/workspace/skills` - Skills Documentation
- **Purpose**: Skill files mounted from host
- **Access**: Read-only (from skill perspective)
- **Contains**: `.md` skill documentation files
- **Structure**:
  - `/workspace/skills/master-skill/SKILL.md`
  - `/workspace/skills/master-skill/mcp-tools/SKILL.md`
  - `/workspace/skills/master-skill/python-programming/SKILL.md`
  - etc.

### `/tmp` - Temporary Storage
- **Purpose**: Temporary files and scratch work
- **Access**: Full read/write permissions
- **Use for**:
  - Downloaded files
  - Intermediate processing results
  - Temporary data that doesn't need persistence
- **Note**: Contents may be cleared between sessions

### Other Directories
- `/usr`, `/bin`, `/lib` - System directories (read-only)
- `/var` - Variable data (limited access)
- `/opt` - Optional software installations

---

## 📋 Common Patterns

### Finding User's Python Files

**❌ WRONG - Searching only /workspace:**
```python
# This only finds skill documentation, NOT user files!
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'find',
    'path': '/workspace',
    'pattern': '*.py'
})
# Result: Only finds skill files (if any)
```

**✅ CORRECT - Search /home/gem:**
```python
# This finds the user's actual Python files
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'find',
    'path': '/home/gem',
    'pattern': '*.py',
    'recursive': True
})
# Result: Finds hello_world.py, config.py, etc.
```

**✅ CORRECT - Search multiple locations:**
```bash
run_shell_in_sandbox('find /home/gem /tmp -name "*.py" -type f')
# Searches both user directory and temp storage
```

### Listing All Accessible Files

```bash
# Get complete view of user-accessible files
run_shell_in_sandbox('tree -L 3 /home/gem')

# Or without tree:
run_shell_in_sandbox('find /home/gem -type f | head -50')
```

### Checking Current Working Directory

When using MCP bash execution:
```python
call_mcp_tool_in_sandbox('sandbox_execute_bash', {
    'cmd': 'pwd'
})
# Returns current directory (often /home/gem)
```

### Working with User Files

**Reading:**
```python
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'read',
    'path': '/home/gem/hello_world.py'
})
```

**Writing:**
```python
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'write',
    'path': '/home/gem/output.txt',
    'content': 'Results here'
})
```

**Searching in user files:**
```python
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'search',
    'path': '/home/gem',
    'content': 'import pandas',
    'recursive': True
})
```

---

## 🎯 Quick Decision Guide

**Looking for user's project files?** → `/home/gem`

**Need to read skill documentation?** → `/workspace/skills`

**Creating temporary files?** → `/tmp`

**Running shell commands?** → Specify absolute paths or ensure CWD is `/home/gem`

**Searching for "user's Python files"?** → Search `/home/gem`, NOT `/workspace`

---

## ⚠️ Common Mistakes

1. **Searching `/workspace` only**
   - Problem: Misses all user files in `/home/gem`
   - Solution: Always search `/home/gem` for user content

2. **Assuming relative paths work**
   - Problem: CWD may vary between commands
   - Solution: Always use absolute paths (`/home/gem/file.py`)

3. **Forgetting about `/home/gem`**
   - Problem: Agent thinks `/workspace` is the only directory
   - Solution: Remember `/home/gem` is the PRIMARY user directory

4. **Using wrong path for file operations**
   - Problem: Trying to write to read-only `/workspace/skills`
   - Solution: Write to `/home/gem` or `/tmp`

---

## 📊 Directory Permissions Summary

| Directory | Read | Write | Purpose |
|-----------|------|-------|---------|
| `/home/gem` | ✅ | ✅ | User's main workspace |
| `/workspace/skills` | ✅ | ❌ | Skill documentation |
| `/tmp` | ✅ | ✅ | Temporary storage |
| `/usr`, `/bin` | ✅ | ❌ | System binaries |
| `/var` | Limited | Limited | Variable data |

---

## 🔍 Exploration Commands

### Get Full Filesystem Overview
```bash
# See directory structure
run_shell_in_sandbox('ls -la /home/gem')

# Find all Python files in user directory
run_shell_in_sandbox('find /home/gem -name "*.py" -type f')

# Check disk usage
run_shell_in_sandbox('du -sh /home/gem /tmp /workspace')

# List all accessible directories
run_shell_in_sandbox('ls -ld /home/gem /tmp /workspace/skills')
```

### Check What's in Each Location
```bash
# User files
run_shell_in_sandbox('ls -lah /home/gem')

# Skills
run_shell_in_sandbox('ls -lah /workspace/skills')

# Temp files
run_shell_in_sandbox('ls -lah /tmp')
```

---

## 💡 Best Practices

1. **Always use absolute paths** - Don't rely on CWD
2. **Search `/home/gem` for user files** - Not `/workspace`
3. **Check multiple locations** - Use `find /home/gem /tmp` for comprehensive search
4. **Verify paths before operations** - Use `ls` or file existence checks
5. **Use `/tmp` for intermediates** - Keep `/home/gem` organized
6. **Remember the hierarchy**:
   - `/home/gem` = user's code
   - `/workspace/skills` = documentation
   - `/tmp` = temporary work

---

## 🚀 Usage with Different Tools

### With MCP Tools
```python
# File operations: specify full path
call_mcp_tool_in_sandbox('sandbox_file_operations', {
    'action': 'read',
    'path': '/home/gem/config.py'  # Full path
})

# Bash execution: set cwd if needed
call_mcp_tool_in_sandbox('sandbox_execute_bash', {
    'cmd': 'ls -la',
    'cwd': '/home/gem'  # Explicit working directory
})
```

### With Shell Tool
```python
# Always use absolute paths
run_shell_in_sandbox('cat /home/gem/hello_world.py')
run_shell_in_sandbox('python /home/gem/script.py')
```

### With Python Tool
```python
# Specify full paths in Python code
run_python_in_sandbox('''
import os
print(os.listdir("/home/gem"))

with open("/home/gem/data.txt", "r") as f:
    print(f.read())
''')
```

---

## 📝 Summary

**The most important thing to remember:**

User's actual files are in `/home/gem`, NOT `/workspace`.

When asked to find, read, or work with "user files", "project files", or "Python scripts", always look in `/home/gem` first.
