"""Tool definitions for the LLM agent to interact with the AIO Sandbox.

Note: Basic file operations (read, write, list, execute) are handled by built-in
deepagents tools through AIOSandboxBackend. These tools provide unique functionality.
"""

from langchain_core.tools import tool
from sandbox_browser_tools import run_visible_browser_steps
import subprocess
from MCP_functions import call_mcp_tool
from sandbox_core_functions import get_sandbox_context, create_skill_in_sandbox

@tool
def run_browser_steps(
    steps: list[dict],
    screenshot_dir: str = "/workspace/screenshots",
    settle_seconds: float = 0.8,
) -> dict:
    """Run a sequence of visible GUI browser steps in the sandbox and save a screenshot after each step.

    Args:
        steps: List of step dicts, e.g.
            {"op":"navigate","url":"https://example.com"}
            {"op":"click","x":100,"y":200}
            {"op":"type","text":"hello"}
            {"op":"hotkey","keys":["enter"]}
            {"op":"scroll","dy":600}
        screenshot_dir: Directory to save screenshots.
        settle_seconds: Default delay between steps.

    Returns:
        Dict with success flag, results per step, and screenshot_dir path.
    """
    # Ensure screenshot directory exists using bash (avoids some os.makedirs permission issues)
    return run_visible_browser_steps(steps, screenshot_dir, settle_seconds)

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
        # File operations
        call_mcp_tool_in_sandbox('file_read', {'path': '/tmp/data.txt'})
        
        # Terminal commands
        call_mcp_tool_in_sandbox('terminal_execute', {'command': 'ls -la'})
    """
    if tool_name.startswith("browser_"):
        return {
            "success": False,
            "error": (
                "Blocked: MCP browser_* tools are disabled for this agent. "
                "Use run_browser_steps (GUI runner) for browser actions."
            ),
        }
    return call_mcp_tool(tool_name, arguments)
