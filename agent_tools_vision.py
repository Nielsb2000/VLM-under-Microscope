# agent_tools_vision.py
import json
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from sandbox_core_functions import execute_python_code

def sandbox_image_to_data_url(image_path: str) -> dict:
    """
    Reads an image inside the sandbox filesystem and returns a data: URL.
    Uses sandbox-side Python to base64 encode the file.
    """
    py = f"""
import base64, mimetypes, json
from pathlib import Path

p = Path({image_path!r})
data = p.read_bytes()
mime = mimetypes.guess_type(str(p))[0] or "image/png"
b64 = base64.b64encode(data).decode("utf-8")
print(json.dumps({{"mime": mime, "data_url": f"data:{{mime}};base64,{{b64}}"}}))
"""
    r = execute_python_code(py, timeout=30)
    if not r.get("success"):
        return {"success": False, "error": r.get("error", "Failed to encode image")}

    # grab last line to be safe if sandbox prints extra lines
    try:
        payload = json.loads(r["output"].strip().splitlines()[-1])
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": f"Failed to parse encoder output: {e}", "raw_output": r.get("output")}


def make_analyze_sandbox_image_tool(llm):
    """
    Returns a callable tool function that closes over (captures) the llm.
    DeepAgents can register this function as a tool.
    """
    @tool 
    def analyze_sandbox_image(image_path: str, question: str = "Describe this image.") -> dict:
        """Analyze an image stored inside the sandbox filesystem using a vision-capable model.

        Args:
            image_path: Path to an image file inside the sandbox (e.g. /workspace/pizza_not_pizza/001.png)
            question: What you want to know about the image

        Returns:
            Dict with keys: success, answer, (and image_path/mime on success)
        """
        # 1) Encode sandbox image to a data URL
        encoded = sandbox_image_to_data_url(image_path)
        if not encoded.get("success"):
            return encoded

        data_url = encoded["data_url"]

        # 2) Ask the vision-capable model with multimodal content
        msg = HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])

        try:
            resp = llm.invoke([msg])
            return {
                "success": True,
                "image_path": image_path,
                "mime": encoded.get("mime"),
                "answer": resp.content,
            }
        except Exception as e:
            return {"success": False, "error": f"Vision model call failed: {e}", "image_path": image_path}

    # Give the inner function a stable name for tool registries
    return analyze_sandbox_image
