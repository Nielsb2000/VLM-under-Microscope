"""
sandbox_core_tools.py - Core, shell, file, and mcp tools for the AIO Sandbox.

This module provides functions for executing code, shell commands, file operations, and other core utilities in the sandbox environment.
"""

import os
from typing import Dict, Any, Optional
from agent_sandbox import Sandbox

def get_sandbox_client() -> Sandbox:
    """Get a configured Sandbox client instance."""
    sandbox_url = os.getenv("SANDBOX_BASE_URL", "http://localhost:8080")
    return Sandbox(base_url=sandbox_url)


def get_vnc_url() -> Dict[str, Any]:
    """
    URL you can open to watch the sandbox browser live.
    """
    try:
        sandbox_url = os.getenv("SANDBOX_BASE_URL", "http://localhost:8080").rstrip("/")
        return {"success": True, "vnc_url": f"{sandbox_url}/vnc/index.html?autoconnect=true"}
    except Exception as e:
        return {"success": False, "error": str(e)}




def execute_python_code(code: str, timeout: Optional[int] = 30) -> Dict[str, Any]:
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

def update_file_content(file_path: str, old_text: str, new_text: str) -> Dict[str, Any]:
    try:
        read_result = read_sandbox_file(file_path)
        if not read_result.get("success"):
            return read_result
        content = read_result.get("content", "")
        if old_text not in content:
            return {
                "success": False,
                "error": f"Text to replace not found in {file_path}",
                "path": file_path
            }
        updated_content = content.replace(old_text, new_text)
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
