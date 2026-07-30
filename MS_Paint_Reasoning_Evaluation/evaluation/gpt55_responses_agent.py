"""GPT-5.5 Responses API helpers for MS Paint Reasoning Evaluation.

This module exists because GPT-5.5 does not support the combination
`/v1/chat/completions + function tools + reasoning_effort`.  For GPT-5.5,
use the Responses API with `reasoning={"effort": ...}` and custom tools.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


_RESPONSES_SEED_SUPPORTED: bool | None = None

READ_FILE_TOOL = {
    "type": "function",
    "name": "read_file",
    "description": (
        "Read a project file needed for the MS Paint reasoning evaluation, "
        "especially files under skills/. The path must be relative to the project root."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Project-root-relative path, for example skills/master-skill/SKILL.md.",
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}


def _to_plain_dict(obj: Any) -> Any:
    """Convert SDK response objects into plain Python structures where possible."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, list):
        return [_to_plain_dict(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain_dict(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return obj


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_project_path(root_dir: str, requested_path: str) -> str:
    requested_path = requested_path.strip().lstrip("/\\")
    abs_root = os.path.abspath(root_dir)
    abs_path = os.path.abspath(os.path.join(abs_root, requested_path))
    if not (abs_path == abs_root or abs_path.startswith(abs_root + os.sep)):
        raise ValueError(f"Refusing to read outside project root: {requested_path}")
    return abs_path


def read_project_file(root_dir: str, path: str) -> str:
    """Read a text file relative to root_dir, with path traversal protection."""
    abs_path = _safe_project_path(root_dir, path)
    if not os.path.exists(abs_path):
        return f"[ERROR] File not found: {path}"
    if os.path.isdir(abs_path):
        try:
            entries = sorted(os.listdir(abs_path))
        except Exception as exc:
            return f"[ERROR] Could not list directory {path}: {exc}"
        return "[DIRECTORY] " + path + "\n" + "\n".join(entries)

    # Skills are Markdown/text. For unexpected binary files, return a useful error.
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        return f"[ERROR] File is not UTF-8 text and cannot be read by read_file: {path}"
    except Exception as exc:
        return f"[ERROR] Could not read {path}: {exc}"


def extract_function_calls(response: Any) -> list[Any]:
    """Return function_call output items from a Responses API response."""
    output = _get_attr_or_key(response, "output", []) or []
    calls = []
    for item in output:
        item_type = _get_attr_or_key(item, "type")
        if item_type == "function_call":
            calls.append(item)
    return calls


def response_usage_dict(response: Any) -> dict[str, int | None] | None:
    usage = _get_attr_or_key(response, "usage")
    if usage is None:
        return None
    return {
        "input_tokens": _get_attr_or_key(usage, "input_tokens"),
        "output_tokens": _get_attr_or_key(usage, "output_tokens"),
        "total_tokens": _get_attr_or_key(usage, "total_tokens"),
    }


def responses_create_with_seed_fallback(client: Any, *, seed: int | None, **kwargs: Any) -> Any:
    """Call OpenAI Responses API with seed, retrying without it if unsupported.

    The Responses API may reject `seed` for some reasoning model paths. Cache that
    result per Python process so a multi-round tool call does not print the same
    warning three or four times for one question.
    """
    global _RESPONSES_SEED_SUPPORTED

    if seed is None or _RESPONSES_SEED_SUPPORTED is False:
        return client.responses.create(**kwargs)

    try:
        response = client.responses.create(seed=seed, **kwargs)
        _RESPONSES_SEED_SUPPORTED = True
        return response
    except Exception as exc:
        msg = str(exc).lower()
        unsupported_seed = "seed" in msg and (
            "unsupported" in msg
            or "unrecognized" in msg
            or "unknown" in msg
            or "extra" in msg
            or "unexpected" in msg
        )
        if not unsupported_seed:
            raise
        _RESPONSES_SEED_SUPPORTED = False
        print("[WARN] This Responses API path did not accept seed; retrying without seed.", file=sys.stderr)
        return client.responses.create(**kwargs)


def run_gpt55_no_skills(
    *,
    client: Any,
    model_name: str,
    system_prompt: str,
    question: str,
    image_base64: str,
    reasoning_effort: str,
    seed: int | None,
) -> tuple[str | None, dict[str, int | None] | None, Any]:
    """Single-shot GPT-5.5 Responses API call without project skills/tools."""
    response = responses_create_with_seed_fallback(
        client,
        seed=seed,
        model=model_name,
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"},
                ],
            }
        ],
        reasoning={"effort": reasoning_effort, "summary": "auto"},
    )
    return _get_attr_or_key(response, "output_text"), response_usage_dict(response), response


def run_gpt55_with_skills(
    *,
    client: Any,
    model_name: str,
    root_dir: str,
    system_prompt: str,
    question: str,
    image_base64: str,
    reasoning_effort: str,
    seed: int | None,
    max_tool_rounds: int = 12,
) -> tuple[str | None, dict[str, int | None] | None, Any]:
    """Run a small Responses API tool loop exposing read_file for skill files."""
    response = responses_create_with_seed_fallback(
        client,
        seed=seed,
        model=model_name,
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"},
                ],
            }
        ],
        tools=[READ_FILE_TOOL],
        reasoning={"effort": reasoning_effort, "summary": "auto"},
    )

    for _round in range(max_tool_rounds):
        calls = extract_function_calls(response)
        if not calls:
            return _get_attr_or_key(response, "output_text"), response_usage_dict(response), response

        tool_outputs = []
        for call in calls:
            name = _get_attr_or_key(call, "name")
            call_id = _get_attr_or_key(call, "call_id")
            raw_args = _get_attr_or_key(call, "arguments", "{}") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else _to_plain_dict(raw_args)
            except Exception as exc:
                args = {}
                output = f"[ERROR] Could not parse tool arguments: {exc}; raw={raw_args!r}"
            else:
                if name == "read_file":
                    output = read_project_file(root_dir, str(args.get("path", "")))
                else:
                    output = f"[ERROR] Unknown tool: {name}"

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            )

        response = responses_create_with_seed_fallback(
            client,
            seed=seed,
            model=model_name,
            previous_response_id=_get_attr_or_key(response, "id"),
            input=tool_outputs,
            tools=[READ_FILE_TOOL],
            reasoning={"effort": reasoning_effort, "summary": "auto"},
        )

    raise RuntimeError(f"GPT-5.5 tool loop exceeded max_tool_rounds={max_tool_rounds}")
