"""Tool definitions for the LLM agent to interact with the AIO Sandbox.

Note: Basic file operations (read, write, list, execute) are handled by built-in
deepagents tools through AIOSandboxBackend. These tools provide unique functionality.
"""

from langchain_core.tools import tool
from sandbox_tools import (
    create_skill_in_sandbox,
    get_sandbox_context,
    call_mcp_tool,
)


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
        # Browser navigation
        call_mcp_tool_in_sandbox('browser_navigate', {'url': 'https://example.com'})
        
        # File operations
        call_mcp_tool_in_sandbox('file_read', {'path': '/tmp/data.txt'})
        
        # Terminal commands
        call_mcp_tool_in_sandbox('terminal_execute', {'command': 'ls -la'})
    """
    return call_mcp_tool(tool_name, arguments)
