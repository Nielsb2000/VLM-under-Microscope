"""package_run.py - Case Study 1 run packager.

Collects all artifacts produced by a single Case Study 1 agent run and writes
them into a self-contained per-run folder:

    outputs/case_study_1/runs/<run_id>/
        run_manifest.json
        metrics/
            result_hist.json
        images/
            randomized_start.png
            final_result.png
            reference_hidden.png
            step_000.png  step_001.png  ...
        histograms/
            randomized_hist.png
            result_hist.png
            ref_hist.png
            comparison_hist.png
        full_trace/
            agent_trace.json
            model_messages.jsonl
            tool_calls.jsonl
            agent_timeline.jsonl
        prompts/
            sem_service_skill.md
            sem_histogram_eval_skill.md
        actions/
            filter_trajectory.csv
            vlm_snapshots.csv
        summary/
            run_summary.md

Sources
-------
- sem-service/histograms/result/result_hist_<ts>.json  (latest by default)
- GET /api/session/stats                                (filter/VLM logs - live API)
- logs/traces/trace_<ts>.json                          (latest agent trace)
- screenshots/paint_<ts>/                              (VLM step images)
- skills/master-skill/sem-service/SKILL.md             (prompt reference)
- skills/master-skill/sem-histogram-eval/SKILL.md

Usage
-----
    python package_run.py [--run-id RUN_ID] [--paint-url URL]
                          [--result-json PATH] [--traces-dir PATH]
                          [--screenshots-dir PATH] [--output-dir PATH]

All flags are optional - sensible defaults are derived from the project root.
Run this once immediately after the agent has finished and
sem_histogram_error.py has been called.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent


def _fetch_json(url: str, timeout: int = 10) -> dict | None:
    try:
        with urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        print(f"  [warn] Could not fetch {url}: {exc}", file=sys.stderr)
        return None


def _resolve_result_artifact(path_value: str | None, result_dir: Path) -> Path | None:
    """Resolve artifact paths recorded in result_hist.json.

    Supports:
    - new preferred format: filename only, e.g. result_img_20260602_154455.png
    - relative paths
    - old sandbox absolute paths, e.g. /workspace/histograms/result/result_img_*.png
    - host absolute paths
    """
    if not path_value:
        return None

    p = Path(path_value)

    # Preferred new format: filename or relative path.
    if not p.is_absolute():
        return result_dir / p

    # Old sandbox/container path recorded by the agent-side eval script.
    if str(p).startswith("/workspace/histograms/result/"):
        return result_dir / p.name

    # Old host absolute path.
    return p


def _latest_file(directory: Path, glob: str) -> Path | None:
    candidates = sorted(directory.glob(glob))
    return candidates[-1] if candidates else None


def _iso_to_dt(iso: str) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.rstrip("Z")).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _paint_ts_to_dt(dir_name: str) -> datetime | None:
    """Convert 'paint_20260528_143022_123456' -> datetime."""
    try:
        stem = dir_name.removeprefix("paint_")
        # format: YYYYMMDD_HHMMSS_ffffff
        return datetime.strptime(stem, "%Y%m%d_%H%M%S_%f").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _extract_model_name(trace_data: dict) -> str | None:
    """Best-effort model name extraction from agent trace payloads.

    Current agent_api.py trace files do not always include a model name, so this
    helper must never raise. It supports future trace shapes as well as the
    present {started_at, completed_at, user_message, steps} shape.
    """
    for key in ("model_name", "model", "llm_model"):
        value = trace_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = trace_data.get("metadata")
    if isinstance(metadata, dict):
        for key in ("model_name", "model", "llm_model"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _normalise_trace_steps(trace_data: dict) -> list[dict]:
    """Return trace steps in the structured format emitted by agent_api.py.

    The current agent API persists {steps: [{step, thinking, calls}...]}. Older
    traces may use a flat {trace: [...]} shape; those are converted into a small
    compatible step list so package_run.py can still create JSONL files.
    """
    steps = trace_data.get("steps")
    if isinstance(steps, list):
        return [step for step in steps if isinstance(step, dict)]

    flat = trace_data.get("trace")
    if not isinstance(flat, list):
        return []

    converted: list[dict] = []
    for i, entry in enumerate(flat, 1):
        if not isinstance(entry, dict):
            continue
        typ = entry.get("type")
        if typ == "tool_call":
            converted.append({
                "type": "step",
                "step": i,
                "thinking": None,
                "calls": [{
                    "tool": entry.get("tool"),
                    "action": None,
                    "category": "unknown",
                    "input_summary": entry.get("input"),
                    "result": None,
                }],
            })
        else:
            converted.append({
                "type": "step",
                "step": i,
                "thinking": entry.get("content"),
                "calls": [],
            })
    return converted


def _select_trace_file(traces_dir: Path, completed_dt: datetime) -> Path | None:
    """Select the trace whose completion timestamp is closest to the result.

    Falling back to the latest file keeps the behaviour useful if timestamps are
    missing or malformed.
    """
    candidates = sorted(traces_dir.glob("trace_*.json"))
    if not candidates:
        return None

    best: tuple[float, Path] | None = None
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        trace_dt = _iso_to_dt(data.get("completed_at", ""))
        if trace_dt is None:
            continue
        delta = abs((trace_dt - completed_dt).total_seconds())
        if best is None or delta < best[0]:
            best = (delta, path)

    return best[1] if best else candidates[-1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def package_run(
    *,
    run_id: str | None,
    paint_url: str,
    result_json_path: Path | None,
    traces_dir: Path,
    screenshots_dir: Path,
    output_base: Path,
    exploratory: bool | None,
    seed: int | None = None,
    sample_seed: int | None = None,
    randomization_seed: int | None = None,
    sample_index: int | None = None,
    model_name_override: str | None = None,
    console_log_path: Path | None = None,

) -> Path:
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Locate the result JSON
    # ------------------------------------------------------------------
    result_dir = _PROJECT_ROOT / "sem-service" / "histograms" / "result"
    if result_json_path is None:
        result_json_path = _latest_file(result_dir, "result_hist_*.json")
        if result_json_path is None:
            # Fall back to the "latest" copy
            fallback = result_dir / "result_hist.json"
            if fallback.exists():
                result_json_path = fallback
            else:
                print("ERROR: No result_hist*.json found. Run sem_histogram_error.py first.", file=sys.stderr)
                sys.exit(1)
    result_data: dict = json.loads(result_json_path.read_text())
    print(f"  Result JSON  : {result_json_path}")
    result_stem = result_json_path.stem
    result_suffix = result_stem.removeprefix("result_hist_")

    # ------------------------------------------------------------------
    # 2. Derive run_id and timing from the result JSON
    # ------------------------------------------------------------------
    completed_at_iso: str = result_data.get("capturedAt", "")
    completed_dt = _iso_to_dt(completed_at_iso) or datetime.now(timezone.utc)
    ref_captured_iso: str = result_data.get("referenceCapturedAt", "")
    started_dt = _iso_to_dt(ref_captured_iso) or completed_dt

    if not run_id:
        run_id = "case1_" + completed_dt.strftime("%Y%m%d_%H%M%S")

    run_dir = output_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Run dir      : {run_dir}")

    # ------------------------------------------------------------------
    # 3. Fetch live session stats (filter log, VLM log)
    # ------------------------------------------------------------------
    base = paint_url.rstrip("/")
    session_stats = _fetch_json(f"{base}/api/session/stats")
    if session_stats is None:
        warnings.append("session_stats_unavailable: /api/session/stats did not respond")

    filter_log: list[dict] = (session_stats or {}).get("filterLog", [])
    vlm_log: list[dict]    = (session_stats or {}).get("vlmSnapshotLog", [])

    # Fall back to summary counts embedded in the result JSON
    filter_adjustments = (
        session_stats.get("filterAdjustments") if session_stats else None
    ) or result_data.get("filterAdjustments", "n/a")
    vlm_snapshots = (
        session_stats.get("vlmSnapshots") if session_stats else None
    ) or result_data.get("vlmSnapshots", "n/a")

    # ------------------------------------------------------------------
    # 4. Create subdirectories
    # ------------------------------------------------------------------
    for sub in ("metrics", "images", "histograms", "full_trace", "prompts", "actions", "summary"):
        (run_dir / sub).mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 5. Copy metric JSON
    # ------------------------------------------------------------------
    shutil.copy2(result_json_path, run_dir / "metrics" / "result_hist.json")

    # ------------------------------------------------------------------
    # 6. Copy images
    # ------------------------------------------------------------------
    def _copy_optional(src_path_str: str | None, dst: Path, label: str) -> bool:
        src = _resolve_result_artifact(src_path_str, result_dir)

        if src is None:
            warnings.append(f"{label}_missing: path not recorded in result JSON")
            return False

        if not src.exists():
            warnings.append(f"{label}_missing: {src_path_str} resolved to {src}, but file was not found on disk")
            return False

        shutil.copy2(src, dst)
        return True

    _copy_optional(result_data.get("randomizedImageFile"), run_dir / "images" / "randomized_start.png", "randomized_start")
    _copy_optional(result_data.get("resultImage"),         run_dir / "images" / "final_result.png",     "final_result")
    _copy_optional(result_data.get("referenceImageFile"),  run_dir / "images" / "reference_hidden.png", "reference_hidden")

    # ------------------------------------------------------------------
    # 7. Copy histogram PNGs  (look for timestamped copies first)
    # ------------------------------------------------------------------
    ts_tag = completed_dt.strftime("%Y%m") if completed_at_iso else ""

    def _copy_hist(timestamped_name: str, latest_name: str, dst_name: str, label: str) -> None:
        src = result_dir / timestamped_name

        if not src.exists():
            src = result_dir / latest_name

        if src.exists():
            shutil.copy2(src, run_dir / "histograms" / dst_name)
        else:
            warnings.append(f"{label}_histogram_missing")

    _copy_hist(f"result_hist_{result_suffix}.png",     "result_hist.png",     "result_hist.png",     "result")
    _copy_hist(f"ref_hist_{result_suffix}.png",        "ref_hist.png",        "ref_hist.png",        "reference")
    _copy_hist(f"comparison_hist_{result_suffix}.png", "comparison_hist.png", "comparison_hist.png", "comparison")

    rand_hist = result_dir / f"rand_hist_{result_suffix}.png"
    if not rand_hist.exists():
        rand_hist = result_dir / "rand_hist.png"
    if not rand_hist.exists():
        rand_hist_dir = _PROJECT_ROOT / "sem-service" / "histograms" / "randomized"
        rand_hist = rand_hist_dir / "rand_hist.png"

    if rand_hist.exists():
        shutil.copy2(rand_hist, run_dir / "histograms" / "randomized_hist.png")
    else:
        warnings.append("randomized_histogram_missing")

    # ------------------------------------------------------------------
    # 8. Copy step images (VLM snapshots from screenshots/paint_*/)
    # ------------------------------------------------------------------
    step_paths: list[Path] = []
    if screenshots_dir.exists():
        paint_dirs = sorted(d for d in screenshots_dir.iterdir() if d.is_dir() and d.name.startswith("paint_"))
        for d in paint_dirs:
            dt = _paint_ts_to_dt(d.name)
            if dt is None:
                continue
            # Accept screenshots taken between randomize time and 5 minutes after eval
            if started_dt <= dt <= completed_dt + timedelta(minutes=5):
                for img in sorted(d.glob("*.png")):
                    step_paths.append(img)

    for i, src in enumerate(step_paths):
        shutil.copy2(src, run_dir / "images" / f"step_{i:03d}.png")

    if not step_paths:
        warnings.append("step_images_missing: no paint_* screenshot dirs matched the run window")

    # ------------------------------------------------------------------
    # 9. Copy prompts
    # ------------------------------------------------------------------
    skills_root = _PROJECT_ROOT / "skills" / "master-skill"
    prompt_sources = {
        "sem_service_skill.md":         skills_root / "sem-service"        / "SKILL.md",
        "sem_histogram_eval_skill.md":  skills_root / "sem-histogram-eval" / "SKILL.md",
        "master_skill.md":              skills_root / "SKILL.md",
    }
    for dst_name, src in prompt_sources.items():
        if src.exists():
            shutil.copy2(src, run_dir / "prompts" / dst_name)
        else:
            warnings.append(f"prompt_missing: {src}")

    # ------------------------------------------------------------------
    # 10. Write agent trace artifacts directly from logs/traces/trace_*.json
    # ------------------------------------------------------------------
    trace_data: dict | None = None
    model_name: str | None = None
    raw_trace_dst = run_dir / "full_trace" / "agent_trace.json"
    timeline_dst = run_dir / "full_trace" / "agent_timeline.jsonl"

    if traces_dir.exists():
        trace_file = _select_trace_file(traces_dir, completed_dt)
        if trace_file:
            try:
                trace_data = json.loads(trace_file.read_text())
                model_name = _extract_model_name(trace_data)
                steps = _normalise_trace_steps(trace_data)

                # Check the trace is plausibly from this run (completed_at within 10 min).
                trace_completed = _iso_to_dt(trace_data.get("completed_at", ""))
                if trace_completed and abs((trace_completed - completed_dt).total_seconds()) > 600:
                    warnings.append(f"trace_timestamp_mismatch: {trace_file.name} may not belong to this run")

                # Preserve the raw agent API trace as the canonical log artifact.
                shutil.copy2(trace_file, raw_trace_dst)

                # agent_reply.txt - final natural-language response, when available.
                if isinstance(trace_data.get("reply"), str):
                    agent_reply_dst.write_text(trace_data.get("reply", ""), encoding="utf-8")

                # model_messages.jsonl - one JSON object per model/thinking step.
                msgs_path = run_dir / "full_trace" / "model_messages.jsonl"
                model_rows = []
                for step in steps:
                    thinking = step.get("thinking")
                    if thinking:
                        model_rows.append({
                            "step": step.get("step"),
                            "type": "thinking",
                            "content": thinking,
                        })
                _write_jsonl(msgs_path, model_rows)

                # tool_calls.jsonl - one JSON object per tool call.
                calls_path = run_dir / "full_trace" / "tool_calls.jsonl"
                call_rows = []
                for step in steps:
                    for call in step.get("calls", []) or []:
                        call_rows.append({
                            "step":          step.get("step"),
                            "tool":          call.get("tool"),
                            "action":        call.get("action"),
                            "category":      call.get("category"),
                            "input_summary": call.get("input_summary"),
                            "result":        call.get("result"),
                            "result_is_json": call.get("result_is_json"),
                        })
                _write_jsonl(calls_path, call_rows)

                # agent_timeline.jsonl - compact chronological mix of model and tool events.
                timeline_rows = []
                for step in steps:
                    step_no = step.get("step")
                    if step.get("thinking"):
                        timeline_rows.append({
                            "step": step_no,
                            "event": "model_message",
                            "content": step.get("thinking"),
                        })
                    for call in step.get("calls", []) or []:
                        timeline_rows.append({
                            "step": step_no,
                            "event": "tool_call",
                            "tool": call.get("tool"),
                            "action": call.get("action"),
                            "category": call.get("category"),
                            "input_summary": call.get("input_summary"),
                            "result": call.get("result"),
                        })
                _write_jsonl(timeline_dst, timeline_rows)

                if not model_rows:
                    warnings.append("model_messages_empty: trace contained no thinking/model text")
                if not call_rows:
                    warnings.append("tool_calls_empty: trace contained no tool calls")

                print(f"  Trace        : {trace_file.name}")
            except Exception as exc:
                warnings.append(f"trace_parse_error: {exc}")
        else:
            warnings.append("trace_missing: no trace_*.json found in logs/traces/")
    else:
        warnings.append("trace_missing: logs/traces/ does not exist")

    # ------------------------------------------------------------------
    # 11. Write action CSVs
    # ------------------------------------------------------------------
    if filter_log:
        traj_path = run_dir / "actions" / "filter_trajectory.csv"
        with traj_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "t", "brightness", "contrast"])
            writer.writeheader()
            for i, entry in enumerate(filter_log, 1):
                writer.writerow({"step": i, **entry})
    else:
        warnings.append("filter_trajectory_missing: no filterLog in session stats")

    if vlm_log:
        vlm_path = run_dir / "actions" / "vlm_snapshots.csv"
        with vlm_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["snapshot", "t"])
            writer.writeheader()
            for i, entry in enumerate(vlm_log, 1):
                writer.writerow({"snapshot": i, **entry})
    else:
        warnings.append("vlm_snapshots_missing: no vlmSnapshotLog in session stats")

    # ------------------------------------------------------------------
    # 12. Compute summary metrics
    # ------------------------------------------------------------------
    final_score      = result_data.get("score")
    randomized_score = result_data.get("randomizedScore")
    abs_improvement  = (
        round(randomized_score - final_score, 6)
        if isinstance(randomized_score, (int, float)) and isinstance(final_score, (int, float))
        else None
    )
    rel_improvement  = (
        round(abs_improvement / randomized_score * 100, 2)
        if abs_improvement is not None and randomized_score and randomized_score > 0
        else None
    )

    # Prefer an explicit CLI model name, then the agent trace metadata, then
    # anything copied into result_hist.json by the evaluator.
    model_name = model_name_override or model_name or result_data.get("model_name") or result_data.get("modelName")

    # The randomization seed may be passed by the runner or recorded by the
    # randomizer/evaluator in the metric JSON.
    randomization_seed = (
        randomization_seed
        if randomization_seed is not None
        else result_data.get("randomizationSeed")
    )

    # Optional console log: package it if the caller provides a file.
    console_dst: Path | None = None
    if console_log_path is not None and console_log_path.exists():
        console_dst = run_dir / "full_trace" / "runner_console.txt"
        shutil.copy2(console_log_path, console_dst)

    # ------------------------------------------------------------------
    # 13. Write run_manifest.json
    # ------------------------------------------------------------------
    def _rel(p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            return str(p.relative_to(run_dir))
        except Exception:
            return str(p)

    metrics_json_dst  = run_dir / "metrics"   / "result_hist.json"
    rand_hist_dst     = run_dir / "histograms" / "randomized_hist.png"
    result_hist_dst   = run_dir / "histograms" / "result_hist.png"
    ref_hist_dst      = run_dir / "histograms" / "ref_hist.png"
    comp_hist_dst     = run_dir / "histograms" / "comparison_hist.png"
    raw_trace_dst     = run_dir / "full_trace" / "agent_trace.json"
    model_msgs_dst    = run_dir / "full_trace" / "model_messages.jsonl"
    tool_calls_dst    = run_dir / "full_trace" / "tool_calls.jsonl"
    timeline_dst      = run_dir / "full_trace" / "agent_timeline.jsonl"
    agent_reply_dst   = run_dir / "full_trace" / "agent_reply.txt"
    filter_traj_dst   = run_dir / "actions"    / "filter_trajectory.csv"
    vlm_snap_dst      = run_dir / "actions"    / "vlm_snapshots.csv"
    summary_dst       = run_dir / "summary"    / "run_summary.md"

    manifest = {
        "run_id":               run_id,
        "case_study":           "case_study_1",
        "sample_id":            result_data.get("referenceImage"),
        "started_at":           started_dt.isoformat(),
        "completed_at":         completed_dt.isoformat(),
        "status":               "completed",
        "exploratory":          exploratory,
        "model_name":           model_name,
        "dataset_image":        result_data.get("referenceImage"),
        "seed":                 seed,
        "sample_seed":          sample_seed,
        "sample_index":         sample_index,
        "randomization_seed":   randomization_seed,
        "final_filters":        result_data.get("finalFilters"),
        "random_filters":       result_data.get("randomFilters"),
        "metric_json":          _rel(metrics_json_dst),
        "final_score":          final_score,
        "randomized_score":     randomized_score,
        "absolute_improvement": abs_improvement,
        "relative_improvement": rel_improvement,
        "filter_adjustments":   filter_adjustments,
        "vlm_snapshots":        vlm_snapshots,
        "images": {
            "randomized_start": "images/randomized_start.png" if (run_dir / "images" / "randomized_start.png").exists() else None,
            "final_result":     "images/final_result.png"     if (run_dir / "images" / "final_result.png").exists()     else None,
            "reference_hidden": "images/reference_hidden.png" if (run_dir / "images" / "reference_hidden.png").exists() else None,
            "steps":            [f"images/step_{i:03d}.png" for i in range(len(step_paths))],
        },
        "histograms": {
            "randomized": _rel(rand_hist_dst)   if rand_hist_dst.exists()   else None,
            "result":     _rel(result_hist_dst) if result_hist_dst.exists() else None,
            "reference":  _rel(ref_hist_dst)    if ref_hist_dst.exists()    else None,
            "comparison": _rel(comp_hist_dst)   if comp_hist_dst.exists()   else None,
        },
        "logs": {
            "raw_agent_trace": _rel(raw_trace_dst)  if raw_trace_dst.exists()  else None,
            "model_messages":  _rel(model_msgs_dst) if model_msgs_dst.exists() else None,
            "tool_calls":      _rel(tool_calls_dst)  if tool_calls_dst.exists()  else None,
            "agent_reply":     _rel(agent_reply_dst) if agent_reply_dst.exists() else None,
            "service_events":  _rel(timeline_dst)    if timeline_dst.exists()    else None,
            "console":         _rel(console_dst)     if console_dst and console_dst.exists() else None,
        },
        "actions": {
            "filter_trajectory": _rel(filter_traj_dst) if filter_traj_dst.exists() else None,
            "vlm_snapshots":     _rel(vlm_snap_dst)    if vlm_snap_dst.exists()     else None,
        },
        "warnings": warnings,
    }

    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ------------------------------------------------------------------
    # 14. Write run_summary.md
    # ------------------------------------------------------------------
    quality = (
        "excellent" if isinstance(final_score, float) and final_score < 0.05 else
        "good"      if isinstance(final_score, float) and final_score < 0.15 else
        "fair"      if isinstance(final_score, float) and final_score < 0.40 else
        "poor"
    )

    summary_lines = [
        f"# Run Summary - {run_id}",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Case study | Case Study 1 - Image Quality Optimization |",
        f"| Image | `{result_data.get('referenceImage', 'unknown')}` |",
        f"| Started | {started_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} |",
        f"| Completed | {completed_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} |",
        f"| Model | {model_name or 'unknown'} |",
        f"| Base seed | {seed if seed is not None else 'unseeded'} |",
        f"| Sample seed | {sample_seed if sample_seed is not None else 'unseeded'} |",
        f"| Randomization seed | {randomization_seed if randomization_seed is not None else 'unseeded'} |",
        f"| Final score | **{final_score}** ({quality}) |",
        f"| Randomized score (challenge) | {randomized_score} |",
        f"| Absolute improvement | {abs_improvement} |",
        f"| Relative improvement | {rel_improvement}% |",
        f"| Filter adjustments | {filter_adjustments} |",
        f"| VLM snapshots | {vlm_snapshots} |",
        f"| Random filters | brightness={result_data.get('randomFilters', {}).get('brightness', '?')}  contrast={result_data.get('randomFilters', {}).get('contrast', '?')} |",
        f"| Final filters | brightness={result_data.get('finalFilters', {}).get('brightness', '?')}  contrast={result_data.get('finalFilters', {}).get('contrast', '?')} |",
        "",
        "## Score legend",
        "",
        "| Range | Quality |",
        "|-------|---------|",
        "| < 0.05 | excellent |",
        "| 0.05 – 0.15 | good |",
        "| 0.15 – 0.40 | fair |",
        "| > 0.40 | poor |",
        "",
    ]
    if warnings:
        summary_lines += ["## Warnings", ""]
        for w in warnings:
            summary_lines.append(f"- `{w}`")
        summary_lines.append("")

    summary_dst.write_text("\n".join(summary_lines))

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n  ✓ Run packaged -> {run_dir.resolve()}")
    if warnings:
        print(f"  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    · {w}")

    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Package a completed Case Study 1 agent run into a self-contained folder."
    )
    parser.add_argument("--run-id",        default=None,  help="Override run ID (default: case1_<timestamp>)")
    parser.add_argument("--paint-url",     default="http://localhost:3000", help="sem-service base URL")
    parser.add_argument("--result-json",   default=None,  help="Path to result_hist_*.json (default: latest in sem-service/histograms/result/)")
    parser.add_argument("--traces-dir",    default=str(_PROJECT_ROOT / "logs" / "traces"), help="Directory containing trace_*.json files")
    parser.add_argument("--screenshots-dir", default=str(_PROJECT_ROOT / "screenshots"),   help="Directory containing paint_* screenshot folders")
    parser.add_argument("--output-dir",    default=str(_PROJECT_ROOT / "outputs" / "case_study_1" / "runs"), help="Parent directory for run output folders")
    parser.add_argument("--exploratory",   choices=["true", "false", "unknown"], default="unknown", help="Whether this run used the exploratory protocol.")
    parser.add_argument("--seed", type=int, default=None, help="Base experiment seed to record in run_manifest.json")
    parser.add_argument("--sample-seed", type=int, default=None, help="Image sampling seed to record in run_manifest.json")
    parser.add_argument("--randomization-seed", type=int, default=None, help="Filter randomization seed to record in run_manifest.json")
    parser.add_argument("--sample-index", type=int, default=None, help="Selected dataset candidate index to record in run_manifest.json")
    parser.add_argument("--model-name", default=None, help="Override model name in run_manifest.json")
    parser.add_argument("--console-log", default=None, help="Optional runner console log file to copy into full_trace/")
    args = parser.parse_args(argv)
    exploratory = (
    True if args.exploratory == "true"
    else False if args.exploratory == "false"
    else None
)
    package_run(
        run_id=args.run_id,
        paint_url=args.paint_url,
        result_json_path=Path(args.result_json) if args.result_json else None,
        traces_dir=Path(args.traces_dir),
        screenshots_dir=Path(args.screenshots_dir),
        output_base=Path(args.output_dir),
        exploratory=exploratory,
        seed=args.seed,
        sample_seed=args.sample_seed,
        randomization_seed=args.randomization_seed,
        sample_index=args.sample_index,
        model_name_override=args.model_name,
        console_log_path=Path(args.console_log) if args.console_log else None,

    )


if __name__ == "__main__":
    main()
