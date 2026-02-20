"""
Sandbox Tools - Interface for executing code and operations in AIO Sandbox.

This module provides functions that the LLM agent can use to execute code,
run shell commands, and perform file operations inside the AIO Sandbox container.
"""


import os
from typing import Dict, Any
import requests

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