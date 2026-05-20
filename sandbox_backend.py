"""AIO Sandbox backend for deepagents.

This backend implements BaseSandbox to execute commands in the AIO Sandbox container.
"""

import os
import base64
import json
from pathlib import Path
from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.protocol import ExecuteResponse, WriteResult, EditResult, FileInfo, GrepMatch
from sandbox_core_functions import execute_shell_command
from deepagents.backends.protocol import FileDownloadResponse, FileUploadResponse


# Allowed base paths for file operations
ALLOWED_PATHS = ["/workspace", "/home/gem"]


def _map_exception_to_error_code(exc_msg: str) -> str:
    """Heuristic mapping from exception message to FileOperationError literal."""
    m = (exc_msg or "").lower()
    if "not found" in m or "404" in m or "no such file" in m:
        return "file_not_found"
    if "permission" in m or "access denied" in m or "permission denied" in m:
        return "permission_denied"
    if "is a directory" in m or ("is directory" in m and "not found" not in m):
        return "is_directory"
    return "invalid_path"


def _is_path_allowed(path: str) -> bool:
    """Check if a path is within allowed directories."""
    try:
        abs_path = Path(path).resolve() if not path.startswith('/') else Path(path)
        return any(str(abs_path).startswith(allowed) for allowed in ALLOWED_PATHS)
    except Exception:
        return False


class AIOSandboxBackend(BaseSandbox):
    """Backend that executes commands via AIO Sandbox HTTP API.
    
    Inherits all file operations from BaseSandbox (glob, grep, read_file, 
    write_file, etc.) which use shell commands. Only needs to implement execute().
    """
    
    def __init__(self, base_url: str | None = None):
        """Initialize with AIO Sandbox URL."""
        self.base_url = base_url or os.getenv("SANDBOX_BASE_URL", "http://localhost:8080")
    
    @property
    def id(self) -> str:
        """Return a unique identifier for this sandbox instance."""
        return f"aio-sandbox-{self.base_url}"
    
    def execute(self, command: str) -> ExecuteResponse:
        """Execute a shell command in the AIO Sandbox.
        
        Args:
            command: Shell command to execute
            
        Returns:
            ExecuteResponse with output and exit code
        """
        result = execute_shell_command(command)
        
        if result["success"]:
            return ExecuteResponse(
                output=result.get("output") or "",
                exit_code=result.get("exit_code", 0),
                truncated=False,
            )
        else:
            return ExecuteResponse(
                output=result.get("error") or "Command failed",
                exit_code=result.get("exit_code", 1),
                truncated=False,
            )
    
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """
        Report upload results. In the AIO Sandbox case with volume mounts,
        files are already accessible, so we return success (error=None).
        """
        responses = []
        for path, _ in files:
            # Validate path before acknowledging
            if not _is_path_allowed(path):
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
                continue
            # If you want to actually write the bytes into the sandbox, do that here.
            # For mount-based setup assume files are already present and return success.
            responses.append(FileUploadResponse(path=path, error=None))
        return responses

    def download_files(self, paths: list[str]) -> list:
        """
        Download files from the AIO Sandbox via the Sandbox API.
        Returns FileDownloadResponse objects with bytes in `.content` on success
        or an error literal on failure.
        """
        # Local imports to avoid top-level import problems and keep signature stable
        from deepagents.backends.protocol import FileDownloadResponse
        from sandbox_core_functions import get_sandbox_client

        client = get_sandbox_client()
        # DEBUG: uncomment to see which base_url the client uses
        # print("DEBUG: sandbox client base_url =", getattr(client, "base_url", None))

        responses: list[FileDownloadResponse] = []

        for path in paths:
            # quick allowlist check
            if not _is_path_allowed(path):
                responses.append(FileDownloadResponse(path=path, content=None, error="permission_denied"))
                continue

            try:
                # Read from the sandbox container via the sandbox API client
                resp = client.file.read_file(file=path)

                # If the API returned data.content, convert it to bytes and return success
                if getattr(resp, "data", None) and getattr(resp.data, "content", None) is not None:
                    raw = resp.data.content
                    content = raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8")
                    responses.append(FileDownloadResponse(path=path, content=content, error=None))
                else:
                    # No content was returned; treat as not found
                    responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))

            except Exception as e:
                # Map common exception text to one of FileOperationError literals
                msg = str(e).lower()
                if "permission" in msg or "access denied" in msg:
                    code = "permission_denied"
                elif "not found" in msg or "no such file" in msg or "404" in msg:
                    code = "file_not_found"
                elif "directory" in msg and "is a directory" in msg:
                    code = "is_directory"
                else:
                    code = "invalid_path"
                responses.append(FileDownloadResponse(path=path, content=None, error=code))

        return responses
    
    def ls_info(self, path: str) -> list[FileInfo]:
        """List directory with path validation."""
        if not _is_path_allowed(path):
            return []
        return super().ls_info(path)
    
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file with path validation, falling back to shell cat on SDK failures."""
        if not _is_path_allowed(file_path):
            return f"Error: Access denied to '{file_path}'. Only /workspace and /home/gem are accessible."
        result = super().read(file_path, offset, limit)
        # BaseSandbox returns an error string (starts with 'Error:') when the SDK
        # download_files path fails (e.g. false 'file not found'). Fall back to
        # a direct shell cat which is known to work reliably.
        if isinstance(result, str) and result.startswith("Error:"):
            cmd = f"cat '{file_path}'"
            if offset:
                cmd = f"tail -n +{offset + 1} '{file_path}'"
            shell_result = execute_shell_command(cmd)
            if shell_result.get("success"):
                output = shell_result.get("output", "")
                lines = output.splitlines(keepends=True)
                return "".join(lines[:limit]) if limit else output
            return result  # return original error if shell also fails
        return result
    
    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        """Search with path validation."""
        search_path = path or "/workspace"
        if not _is_path_allowed(search_path):
            return f"Error: Access denied to '{search_path}'. Only /workspace and /home/gem are accessible."
        return super().grep_raw(pattern, search_path, glob)
    
    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Glob with path validation - searches allowed paths only using find command."""
        # If searching from root or disallowed path, search all allowed paths
        search_paths = ALLOWED_PATHS if not _is_path_allowed(path) else [path]
        
        # Convert glob pattern to find pattern
        # Simple conversion: **/*.py -> -name "*.py"
        if '**/' in pattern:
            pattern = pattern.split('**/')[-1]
        pattern = pattern.lstrip('/')
        
        all_results = []
        for search_path in search_paths:
            # Use find command to search
            cmd = f"find {search_path} -type f -name '{pattern}' 2>/dev/null"
            result = self.execute(cmd)
            
            if result.exit_code == 0 and result.output:
                for file_path in result.output.strip().split('\n'):
                    if file_path and _is_path_allowed(file_path):
                        # Get file stats
                        stat_result = self.execute(f"stat -c '%s %Y' {file_path} 2>/dev/null")
                        if stat_result.exit_code == 0:
                            try:
                                size, mtime = stat_result.output.strip().split()
                                all_results.append({
                                    'path': file_path,
                                    'size': int(size),
                                    'modified': int(mtime),
                                })
                            except:
                                # Fallback without stats
                                all_results.append({'path': file_path})
        
        return all_results
    
    def _glob_info_old(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Old implementation - kept for reference."""
        # Path is allowed, search it directly and filter results
        results = super().glob_info(pattern, path)
        filtered = []
        for fi in results:
            file_path = fi['path']
            if not file_path.startswith('/'):
                file_path = f"{path.rstrip('/')}/{file_path}"
            if _is_path_allowed(file_path):
                fi_copy = fi.copy()
                fi_copy['path'] = file_path
                filtered.append(fi_copy)
        return filtered
    
    def write(self, file_path: str, content: str) -> WriteResult:
        """Override write to avoid library bug with {e} in f-strings and add path validation."""
        if not _is_path_allowed(file_path):
            return WriteResult(error=f"Access denied to '{file_path}'. Only /workspace and /home/gem are accessible.")
        
        # Encode content and create JSON payload
        content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        payload = json.dumps({"path": file_path, "content": content_b64})
        payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        
        # Fixed template with {{e}} properly escaped
        cmd = f"""python3 -c "
import os
import sys
import base64
import json

payload_b64 = sys.stdin.read().strip()
if not payload_b64:
    print('Error: No payload received for write operation', file=sys.stderr)
    sys.exit(1)

try:
    payload = base64.b64decode(payload_b64).decode('utf-8')
    data = json.loads(payload)
    file_path = data['path']
    content = base64.b64decode(data['content']).decode('utf-8')
except Exception as e:
    print(f'Error: Failed to decode write payload: {{e}}', file=sys.stderr)
    sys.exit(1)

if os.path.exists(file_path):
    print(f'Error: File \\\\'{file_path}\\\\' already exists', file=sys.stderr)
    sys.exit(1)

parent_dir = os.path.dirname(file_path) or '.'
os.makedirs(parent_dir, exist_ok=True)

with open(file_path, 'w') as f:
    f.write(content)
" <<'__DEEPAGENTS_EOF__'
{payload_b64}
__DEEPAGENTS_EOF__"""
        
        result = self.execute(cmd)
        
        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)
        
        return WriteResult(path=file_path, files_update=None)
    
    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Override edit to avoid library bug with {e} in f-strings and add path validation."""
        if not _is_path_allowed(file_path):
            return EditResult(error=f"Access denied to '{file_path}'. Only /workspace and /home/gem are accessible.")
        
        payload = json.dumps({"path": file_path, "old": old_string, "new": new_string})
        payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        
        # Fixed template with {{e}} properly escaped
        cmd = f"""python3 -c "
import sys
import base64
import json
import os

payload_b64 = sys.stdin.read().strip()
if not payload_b64:
    print('Error: No payload received for edit operation', file=sys.stderr)
    sys.exit(4)

try:
    payload = base64.b64decode(payload_b64).decode('utf-8')
    data = json.loads(payload)
    file_path = data['path']
    old = data['old']
    new = data['new']
except Exception as e:
    print(f'Error: Failed to decode edit payload: {{e}}', file=sys.stderr)
    sys.exit(4)

if not os.path.isfile(file_path):
    sys.exit(3)

with open(file_path, 'r') as f:
    text = f.read()

count = text.count(old)

if count == 0:
    sys.exit(1)
elif count > 1 and not {replace_all}:
    sys.exit(2)

if {replace_all}:
    result = text.replace(old, new)
else:
    result = text.replace(old, new, 1)

with open(file_path, 'w') as f:
    f.write(result)

print(count)
" <<'__DEEPAGENTS_EOF__'
{payload_b64}
__DEEPAGENTS_EOF__"""
        
        result = self.execute(cmd)
        exit_code = result.exit_code
        output = result.output.strip()
        
        error_messages = {
            1: f"Error: String not found in file: '{old_string}'",
            2: f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences.",
            3: f"Error: File '{file_path}' not found",
            4: f"Error: Failed to decode edit payload: {output}",
        }
        if exit_code in error_messages:
            return EditResult(error=error_messages[exit_code])
        if exit_code != 0:
            return EditResult(error=f"Error editing file (exit code {exit_code}): {output or 'Unknown error'}")
        
        count = int(output)
        return EditResult(path=file_path, files_update=None, occurrences=count)


def get_aio_sandbox_backend() -> AIOSandboxBackend:
    """Get an AIOSandboxBackend configured for the local AIO Sandbox container.
    
    Returns:
        AIOSandboxBackend instance pointing to localhost:8080
    """
    return AIOSandboxBackend()
