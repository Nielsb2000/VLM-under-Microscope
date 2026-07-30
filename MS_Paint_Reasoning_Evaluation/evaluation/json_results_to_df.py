"""
Parse all results JSONs in Results/dashboard_data into a hierarchical pandas DataFrame.

Usage:
    from json_results_to_df import load_results_df
    df = load_results_df()

or run directly:
    uv run python MS_Paint_Reasoning_Evaluation/evaluation/json_results_to_df.py
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "Results", "dashboard_data")

# Keep this empty by default. Add models here only if you explicitly want to hide
# them from every downstream visualization that calls load_results_df().
EXCLUDED_MODELS: set[str] = set()

VALID_BLUR_LEVELS = {"no_blur", "med_blur", "heavy_blur"}
VALID_REASONING_MODES = {"low", "medium", "high", "none"}


def _parse_filename(fname: str) -> tuple[str, str, str, str]:
    """
    Parse filenames like:
        llm_results_color_no_blur_gpt-4o-gpt-5.1-gpt-5.2-gpt-5.5_low_no_skills.json
        llm_results_inverted_greyscale_heavy_blur_gpt-5.5_high_skills.json

    Returns:
        image_type, blur_level, reasoning_mode, skills_mode

    The model string in the filename is intentionally ignored for row creation,
    because the canonical model value is embedded in each JSON key.
    """
    base = fname[:-5]  # strip .json
    if not base.startswith("llm_results_"):
        return "unknown", "unknown", "none", "unknown"

    base = base[len("llm_results_"):]
    parts = base.split("_")

    if len(parts) >= 2 and parts[-2:] == ["no", "skills"]:
        skills_mode = "no_skills"
        parts = parts[:-2]
    elif parts and parts[-1] == "skills":
        skills_mode = "skills"
        parts = parts[:-1]
    else:
        skills_mode = "unknown"

    if parts and parts[-1] in VALID_REASONING_MODES:
        reasoning_mode = parts[-1]
        parts = parts[:-1]
    else:
        reasoning_mode = "none"

    image_type = "unknown"
    blur_level = "unknown"
    for i in range(len(parts) - 1):
        candidate_blur = f"{parts[i]}_{parts[i + 1]}"
        if candidate_blur in VALID_BLUR_LEVELS:
            image_type = "_".join(parts[:i]) if i > 0 else "unknown"
            blur_level = candidate_blur
            break

    return image_type, blur_level, reasoning_mode, skills_mode


def _parse_result_value(value: Any) -> tuple[Any, Any, Any, Any]:
    """Return correct, input_tokens, output_tokens, elapsed_time from a result value."""
    correct = None
    input_tokens = None
    output_tokens = None
    elapsed_time = None

    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            correct = value[0]
        if len(value) > 1 and isinstance(value[1], (list, tuple)):
            if len(value[1]) > 0:
                input_tokens = value[1][0]
            if len(value[1]) > 1:
                output_tokens = value[1][1]
        if len(value) > 2:
            elapsed_time = value[2]

    return correct, input_tokens, output_tokens, elapsed_time


def load_results_df(results_dir: str | None = None) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    results_dir = results_dir or RESULTS_DIR

    if not os.path.exists(results_dir):
        return pd.DataFrame()

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json") or not fname.startswith("llm_results_"):
            continue

        image_type, blur_level, reasoning_mode, skills_mode = _parse_filename(fname)
        path = os.path.join(results_dir, fname)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Skipping unreadable results file {path}: {exc}")
            continue

        if not isinstance(data, dict):
            print(f"[WARN] Skipping non-dict results file {path}")
            continue

        for key, value in data.items():
            # Capture any model id after imgN_QM_, including dots, dashes, underscores,
            # slashes, and future model naming variants.
            m = re.match(r"^img(\d+)_Q(\d+)_(.+)$", key)
            if not m:
                continue

            img = int(m.group(1))
            q = int(m.group(2))
            model_in_key = m.group(3)

            if model_in_key in EXCLUDED_MODELS:
                continue

            correct, input_tokens, output_tokens, elapsed_time = _parse_result_value(value)

            records.append(
                {
                    "image_type": image_type,
                    "blur_level": blur_level,
                    "Model": model_in_key,
                    "Correct": correct,
                    "image_num": img,
                    "question_num": q,
                    "image_filename": f"img{img}.png",
                    "reasoning_mode": None if model_in_key == "gpt-4o" else reasoning_mode,
                    "skills_mode": skills_mode,
                    "Input Tokens": input_tokens,
                    "Output Tokens": output_tokens,
                    "Elapsed Time": elapsed_time,
                    "source_file": fname,
                }
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    df["reasoning_mode"] = df["reasoning_mode"].replace({"none": "low"})

    # Keep a stable, readable column order while preserving any future extra columns.
    preferred_cols = [
        "image_type",
        "blur_level",
        "Model",
        "Correct",
        "image_num",
        "question_num",
        "image_filename",
        "reasoning_mode",
        "skills_mode",
        "source_file",
        "Input Tokens",
        "Output Tokens",
        "Elapsed Time",
    ]
    existing_preferred = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_preferred]
    return df[existing_preferred + other_cols]


if __name__ == "__main__":
    df = load_results_df()
    print(df)
    if not df.empty and "Model" in df.columns:
        print("\nRows by model:")
        print(df["Model"].value_counts(dropna=False).sort_index())
