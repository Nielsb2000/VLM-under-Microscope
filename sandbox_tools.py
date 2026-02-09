"""
Sandbox Tools - Interface for executing code and operations in AIO Sandbox.

This module provides functions that the LLM agent can use to execute code,
run shell commands, and perform file operations inside the AIO Sandbox container.
"""

import os
from typing import Dict, Any, Optional
from agent_sandbox import Sandbox
import requests


def get_sandbox_client() -> Sandbox:
    """Get a configured Sandbox client instance."""
    sandbox_url = os.getenv("SANDBOX_BASE_URL", "http://localhost:8080")
    return Sandbox(base_url=sandbox_url)


def execute_python_code(code: str, timeout: Optional[int] = 30) -> Dict[str, Any]:
    """
    Execute Python code in the sandbox environment.
    
    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds (default: 30)
        
    Returns:
        Dictionary with execution results including output, errors, and exit code
    """
    try:
        client = get_sandbox_client()
        result = client.code.execute_code(
            language="python",
            code=code,
            timeout=timeout
        )
        
        return {
            "success": True,
            "output": result.data.output if hasattr(result.data, 'output') else str(result.data),
            "exit_code": getattr(result.data, 'exit_code', 0)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "exit_code": 1
        }


def execute_shell_command(command: str, timeout: Optional[int] = 30) -> Dict[str, Any]:
    """
    Execute a shell command in the sandbox environment.
    
    Args:
        command: Shell command to execute
        timeout: Execution timeout in seconds (default: 30)
        
    Returns:
        Dictionary with command results including output, errors, and exit code
    """
    try:
        client = get_sandbox_client()
        result = client.shell.exec_command(
            command=command,
            timeout=timeout
        )
        
        return {
            "success": True,
            "output": result.data.output if result.data else "",
            "exit_code": getattr(result.data, 'exit_code', 0)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "exit_code": 1
        }


def read_sandbox_file(file_path: str) -> Dict[str, Any]:
    """
    Read a file from the sandbox filesystem.
    
    Args:
        file_path: Path to the file in the sandbox
        
    Returns:
        Dictionary with file content or error message
    """
    try:
        client = get_sandbox_client()
        result = client.file.read_file(file=file_path)
        
        return {
            "success": True,
            "content": result.data.content if result.data else "",
            "path": file_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "path": file_path
        }


def write_sandbox_file(file_path: str, content: str, append: bool = False) -> Dict[str, Any]:
    """
    Write content to a file in the sandbox filesystem.
    
    Args:
        file_path: Path to the file in the sandbox
        content: Content to write
        append: If True, append to file; if False, overwrite (default: False)
        
    Returns:
        Dictionary with success status or error message
    """
    try:
        client = get_sandbox_client()
        result = client.file.write_file(
            file=file_path,
            content=content,
            append=append
        )
        
        return {
            "success": True,
            "path": file_path,
            "bytes_written": len(content)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "path": file_path
        }


def list_sandbox_directory(directory: str = ".") -> Dict[str, Any]:
    """
    List contents of a directory in the sandbox.
    
    Args:
        directory: Directory path to list (default: current directory)
        
    Returns:
        Dictionary with list of files/directories or error message
    """
    try:
        client = get_sandbox_client()
        result = client.file.list_files(path=directory)
        
        files = result.data.files if hasattr(result.data, 'files') else []
        
        return {
            "success": True,
            "files": files,
            "directory": directory
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "directory": directory
        }


def delete_sandbox_file(file_path: str) -> Dict[str, Any]:
    """
    Delete a file or directory from the sandbox filesystem.
    
    Args:
        file_path: Path to the file or directory in the sandbox
        
    Returns:
        Dictionary with success status or error message
    """
    try:
        client = get_sandbox_client()
        result = client.shell.exec_command(command=f"rm -rf {file_path}")
        
        return {
            "success": True,
            "path": file_path,
            "message": f"Successfully deleted {file_path}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "path": file_path
        }


def update_file_content(file_path: str, old_text: str, new_text: str) -> Dict[str, Any]:
    """
    Update specific content within an existing file using sed-like replacement.
    
    Args:
        file_path: Path to the file in the sandbox
        old_text: Text to find and replace
        new_text: Text to replace with
        
    Returns:
        Dictionary with success status or error message
    """
    try:
        # First, read the file
        read_result = read_sandbox_file(file_path)
        if not read_result.get("success"):
            return read_result
            
        content = read_result.get("content", "")
        
        # Replace the text
        if old_text not in content:
            return {
                "success": False,
                "error": f"Text to replace not found in {file_path}",
                "path": file_path
            }
        
        updated_content = content.replace(old_text, new_text)
        
        # Write back
        write_result = write_sandbox_file(file_path, updated_content, append=False)
        
        if write_result.get("success"):
            return {
                "success": True,
                "path": file_path,
                "message": f"Successfully updated {file_path}"
            }
        return write_result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "path": file_path
        }


def create_skill_in_sandbox(skill_name: str, description: str, content: str, parent_skill: str = None) -> Dict[str, Any]:
    """
    Create a new skill file in the sandbox skills directory.
    Can create either a top-level skill or a sub-skill under a parent.
    
    Args:
        skill_name: Name of the skill (will be used for directory name)
        description: Brief description of the skill
        content: Markdown content for the skill
        parent_skill: Optional parent skill name (e.g., "master-skill" to create a sub-skill)
        
    Returns:
        Dictionary with success status or error message
    """
    # Determine skill path based on whether it's a sub-skill
    if parent_skill:
        skill_dir = f"/workspace/skills/{parent_skill}/{skill_name}"
    else:
        skill_dir = f"/workspace/skills/{skill_name}"
    
    skill_file = f"{skill_dir}/SKILL.md"
    
    # Create the skill content with frontmatter
    skill_content = f"""---
name: {skill_name}
description: {description}
---

{content}
"""
    
    try:
        client = get_sandbox_client()
        
        # Create directory
        client.shell.exec_command(command=f"mkdir -p {skill_dir}")
        
        # Write skill file
        result = client.file.write_file(
            file=skill_file,
            content=skill_content
        )
        
        return {
            "success": True,
            "skill_name": skill_name,
            "skill_path": skill_file,
            "parent_skill": parent_skill,
            "message": f"Skill '{skill_name}' created successfully at {skill_file}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "skill_name": skill_name
        }


def get_sandbox_context() -> Dict[str, Any]:
    """
    Get information about the sandbox environment.
    
    Returns:
        Dictionary with sandbox context information
    """
    try:
        client = get_sandbox_client()
        context = client.sandbox.get_context()
        
        return {
            "success": True,
            "home_dir": context.data.home_dir if hasattr(context.data, 'home_dir') else "/home/gem",
            "user": getattr(context.data, 'user', 'gem'),
            "workspace": getattr(context.data, 'workspace', '/workspace')
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call an MCP tool through the AIO Sandbox's built-in MCP hub.
    
    The sandbox provides browser, file, terminal, and markitdown tools at /mcp endpoint.
    
    Args:
        tool_name: Name of the MCP tool (e.g., 'browser_navigate', 'file_read', 'terminal_execute')
        arguments: Dictionary of arguments to pass to the tool
        
    Returns:
        Dictionary with tool execution results or error message
    """
    try:
        sandbox_url = os.getenv("SANDBOX_BASE_URL", "http://localhost:8080")
        mcp_endpoint = f"{sandbox_url}/mcp"
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": 1
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }

        response = requests.post(
            mcp_endpoint,
            json=payload,
            headers=headers,
            timeout=30.0
        )
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"MCP call failed with status {response.status_code}",
                "tool_name": tool_name
            }
        
        result = response.json()
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "tool_name": tool_name
            }
        
        return {
            "success": True,
            "result": result,
            "tool_name": tool_name
        }
        
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "MCP call timed out",
            "tool_name": tool_name
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to MCP hub",
            "tool_name": tool_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool_name": tool_name
        }

def list_mcp_tools() -> Dict[str, Any]:
    """
    List all available MCP tools in the sandbox.
    
    Returns:
        Dictionary with list of available tools or error message
    """
    try:
        sandbox_url = os.getenv("SANDBOX_BASE_URL", "http://localhost:8080")
        mcp_endpoint = f"{sandbox_url}/mcp"
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 1
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }

        response = requests.post(
            mcp_endpoint,
            json=payload,
            headers=headers,
            timeout=30.0
        )
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"MCP list failed with status {response.status_code}"
            }
        
        result = response.json()
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"]
            }
        
        return {
            "success": True,
            "tools": result.get("result", {}).get("tools", [])
        }
        
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "MCP list timed out"
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to MCP hub"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }