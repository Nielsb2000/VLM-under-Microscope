"""cs4_api.py — FastAPI endpoint for Case Study 4 pattern-search VLM inference.

Start:
    uv run uvicorn cs4_api:app --port 3002

Endpoints:
    GET  /health          → {"ok": true}
    POST /find-pattern    → prediction result JSON

Request body (POST /find-pattern):
    {
      "patternBase64": "<raw base64 or data URI>",
      "searchBase64":  "<raw base64 or data URI>",
      "searchMode":    "atlas_global_search" | "grid_scan_search"
    }

Response:
    {
      "ok":           bool,
      "found":        bool | null,
      "tile":         str | null,
      "bbox":         [x, y, w, h] | null,
      "confidence":   float | null,
      "reason":       str | null,
      "parsing_error": str | null,
      "raw_response": str | null,
      "model":        str | null,
      "usage":        {...} | null,
      "error":        str | null   # only when ok=false
    }
"""

import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import VLM call logic + config from the CS4 runner
from run_case_study_4 import _call_vlm, _load_openai_config
from outputs.case_study_4.response_parser import (
    build_prompt,
    get_system_prompt,
    parse_vlm_response,
)

app = FastAPI(title="CS4 Pattern Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class FindPatternRequest(BaseModel):
    patternBase64: str  # raw base64 or data URI (data:image/...;base64,<data>)
    searchBase64: str
    searchMode: str = "atlas_global_search"


def _strip_data_uri(b64: str) -> bytes:
    """Strip optional data URI prefix and decode to bytes."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/find-pattern")
def find_pattern(req: FindPatternRequest):
    api_key, base_url, _model, reasoning_effort = _load_openai_config()
    model = "gpt-5.5"

    with tempfile.TemporaryDirectory() as tmpdir:
        pattern_path = Path(tmpdir) / "pattern.png"
        search_path = Path(tmpdir) / "search.png"

        try:
            pattern_path.write_bytes(_strip_data_uri(req.patternBase64))
            search_path.write_bytes(_strip_data_uri(req.searchBase64))
        except Exception as exc:
            return {"ok": False, "error": f"Failed to decode image: {exc}"}

        system_prompt = get_system_prompt()
        user_prompt = build_prompt(req.searchMode)

        vlm_result = _call_vlm(
            pattern_path,
            search_path,
            system_prompt,
            user_prompt,
            api_key,
            base_url,
            model,
            reasoning_effort,
        )

    if not vlm_result["ok"]:
        return {"ok": False, "error": vlm_result.get("error", "VLM call failed")}

    raw = vlm_result["reply"]
    parsed = parse_vlm_response(raw)

    return {
        "ok": True,
        "found": parsed.get("found"),
        "tile": parsed.get("tile"),
        "bbox": parsed.get("bbox"),
        "confidence": parsed.get("confidence"),
        "reason": parsed.get("reason"),
        "parsing_error": parsed.get("parsing_error"),
        "raw_response": raw,
        "model": vlm_result.get("model"),
        "usage": vlm_result.get("usage"),
    }
