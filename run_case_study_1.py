"""run_case_study_1.py - Automated Case Study 1 runner.

Orchestrates one or more complete Case Study 1 runs end-to-end:

  1. Sample a random image from the unlabeled dataset (skippable)
  2. Randomize filters and capture reference histogram
  3. Send the task prompt to the agent and wait for completion
  4. Run sem_histogram_error.py on the host to save canonical evaluation output
  5. Package all artifacts with package_run.py

Usage
-----
    # single run (defaults)
    python run_case_study_1.py

    # 5 sequential runs
    python run_case_study_1.py --runs 5

    # use labeled dataset instead
    python run_case_study_1.py --source labeled

    # skip the image-sampling step (use whatever is already loaded)
    python run_case_study_1.py --no-sample

    # 3 runs, 90-minute agent timeout each
    python run_case_study_1.py --runs 3 --timeout 5400

    # dry-run: print the prompt and exit without calling the agent
    python run_case_study_1.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Task prompts (sent to the agent for every run)
# ---------------------------------------------------------------------------

_EVAL_BLOCK = """
import subprocess
res = subprocess.run(
    ["python3",
     "/workspace/skills/master-skill/sem-histogram-eval/sem_histogram_error.py",
     "--paint-url", "http://host.docker.internal:3000"],
    capture_output=True, text=True, timeout=60
)
print(res.stdout)
if res.returncode != 0:
    print("STDERR:", res.stderr)
"""

_RULES = """\
IMPORTANT RULES:
- Do NOT read evaluation output files or histogram JSONs at any point.
- Do NOT call the evaluation script more than once.
- Use only paint_canvas, get_sem_status, and get_canvas_image to guide your decisions.
- If host.docker.internal does not resolve, retry with http://172.17.0.1:3000.
"""

# Full protocol: 4 exploratory settings + minimum 3 refinement iterations
TASK_PROMPT_EXPLORE = (
    "You are performing Case Study 1 - SEM Image Quality Optimisation.\n\n"
    "An SEM image has been loaded onto the canvas and its brightness/contrast filters\n"
    "have been randomised to an unknown combination. Your goal is to restore the image\n"
    "to its natural appearance using only your visual (VLM) perception - do NOT read\n"
    "any histogram or evaluation files.\n\n"
    "Follow this exact protocol:\n\n"
    "**Phase 1 - confirm state**\n"
    "1. Call get_sem_status() to verify the image is loaded and filters are randomised.\n\n"
    "**Phase 2 - exploration (4 settings)**\n"
    "2. Try 4 different filter combinations to map the image space. For each:\n"
    "   a. Call paint_canvas(\"set_filters\", {\"brightness\": <B>, \"contrast\": <C>})\n"
    "   b. Call get_sem_status() to confirm.\n"
    "   c. Call paint_canvas(\"get_canvas_image\") to capture the current view.\n"
    "   d. Visually assess: is the image too dark, too bright, too flat, clipped?\n\n"
    "**Phase 3 - refinement (minimum 3 iterations)**\n"
    "3. Based on what you learned, iteratively refine the filters toward natural appearance.\n"
    "   Each iteration:\n"
    "   a. paint_canvas(\"set_filters\", {...})\n"
    "   b. get_sem_status()\n"
    "   c. paint_canvas(\"get_canvas_image\") - assess visually\n"
    "   Repeat until you are satisfied that the image looks natural (good contrast,\n"
    "   no black/white clipping, detail visible throughout).\n\n"
    "**Phase 4 - evaluation (once, at the very end)**\n"
    "4. When satisfied, declare you are done and call the evaluation script ONCE:\n"
    + _EVAL_BLOCK + "\n" + _RULES
)

# Minimal protocol: skip exploration, go straight to refinement
TASK_PROMPT_NO_EXPLORE = (
    "You are performing Case Study 1 - SEM Image Quality Optimisation.\n\n"
    "An SEM image has been loaded onto the canvas and its brightness/contrast filters\n"
    "have been randomised to an unknown combination. Your goal is to restore the image\n"
    "to its natural appearance using only your visual (VLM) perception - do NOT read\n"
    "any histogram or evaluation files.\n\n"
    "Follow this exact protocol:\n\n"
    "**Phase 1 - confirm state**\n"
    "1. Call get_sem_status() to verify the image is loaded and filters are randomised.\n\n"
    "**Phase 2 - refinement (minimum 3 iterations)**\n"
    "2. Directly adjust the filters toward natural appearance. For each iteration:\n"
    "   a. Call paint_canvas(\"set_filters\", {\"brightness\": <B>, \"contrast\": <C>})\n"
    "   b. Call get_sem_status() to confirm.\n"
    "   c. Call paint_canvas(\"get_canvas_image\") - assess visually.\n"
    "   Repeat until you are satisfied that the image looks natural (good contrast,\n"
    "   no black/white clipping, detail visible throughout). Do at least 3 iterations.\n\n"
    "**Phase 3 - evaluation (once, at the very end)**\n"
    "3. When satisfied, declare you are done and call the evaluation script ONCE:\n"
    + _EVAL_BLOCK + "\n" + _RULES
)

# Default alias (explore on)
TASK_PROMPT = TASK_PROMPT_EXPLORE


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent


def _post(url: str, body: dict, timeout: int = 30) -> dict:
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, timeout: int = 15) -> dict:
    with urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _check_service(url: str, label: str) -> bool:
    try:
        _get(f"{url.rstrip('/')}/api/session/stats" if "3000" in url else f"{url.rstrip('/')}/status")
        return True
    except Exception as e:
        print(f"  [warn] {label} not reachable at {url}: {e}", file=sys.stderr)
        return False


def _derive_seed(base_seed: int | None, run_index: int, stream: str) -> int | None:
    """Derive stable per-run/per-stream uint32 seeds from one user seed."""
    if base_seed is None:
        return None
    key = f"case_study_1:{base_seed}:{run_index}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


# ---------------------------------------------------------------------------
# Single-run logic
# ---------------------------------------------------------------------------

def run_once(
    *,
    run_index: int,
    paint_url: str,
    agent_url: str,
    source: str,
    skip_sample: bool,
    skip_package: bool,
    agent_timeout: int,
    explore: bool,
    dry_run: bool,
    base_seed: int | None,
) -> dict:
    """Execute one full Case Study 1 run. Returns a summary dict."""
    base_paint = paint_url.rstrip("/")
    base_agent = agent_url.rstrip("/")

    ts_start = datetime.now(timezone.utc)
    run_label = f"run_{run_index:02d}_{ts_start.strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"  Case Study 1 - {run_label}")
    print(f"{'='*60}")

    sample_seed = _derive_seed(base_seed, run_index, "sample")
    randomization_seed = _derive_seed(base_seed, run_index, "randomize")

    summary: dict = {
        "run_index":  run_index,
        "run_label":  run_label,
        "started_at": ts_start.isoformat(),
        "status":     "failed",
        "score":      None,
        "image":      None,
        "error":      None,
        "run_dir":    None,
        "exploratory": explore,
        "seed": base_seed,
        "sample_seed": sample_seed,
        "randomization_seed": randomization_seed,

    }

    try:
        # ------------------------------------------------------------------
        # Step 1: Sample image
        # ------------------------------------------------------------------
        if skip_sample:
            print("  [1/5] Skipping image sample - using currently loaded image.")
            state = _get(f"{base_paint}/api/canvas/state")
            img_name = (state.get("canvas", {}).get("backgroundImage") or "?").split("/")[-1]
            print(f"        Currently loaded: {img_name}")
            summary["image"] = img_name
        else:
            print(f"  [1/5] Sampling random image from '{source}' dataset…")
            if dry_run:
                print("        [dry-run] would POST /api/dataset/sample")
                summary["image"] = "dry-run"
            else:
                resp = _post(f"{base_paint}/api/dataset/sample", {"source": source, "seed": sample_seed}, timeout=60)
                if not resp.get("ok"):
                    raise RuntimeError(f"Sample failed: {resp.get('error', resp)}")
                img_name = resp.get("filename", "?")
                category = resp.get("category", "?")
                print(f"        Loaded: {img_name}  (category: {category},  pool: {resp.get('total_candidates', '?')} images)")
                summary["image"] = img_name
                summary["image_category"] = category
                summary["sample_index"] = resp.get("pick_index")
                time.sleep(1.5)  # give the canvas time to render

        # ------------------------------------------------------------------
        # Step 2: Randomize
        # ------------------------------------------------------------------
        print("  [2/5] Randomizing filters and capturing reference histogram…")
        if dry_run:
            print("        [dry-run] would POST /api/randomize")
        else:
            resp = _post(f"{base_paint}/api/randomize", {"seed": randomization_seed}, timeout=30)
            if not resp.get("ok"):
                raise RuntimeError(f"Randomize failed: {resp.get('error', resp)}")
            rand_filters = resp.get("filters", {})
            summary["random_filters"] = rand_filters
            print(f"        Random filters: brightness={rand_filters.get('brightness','?')}  contrast={rand_filters.get('contrast','?')}  seed={randomization_seed}")
            time.sleep(1.0)

        # ------------------------------------------------------------------
        # Step 3: Send task prompt → wait for agent
        # ------------------------------------------------------------------
        prompt = TASK_PROMPT_EXPLORE if explore else TASK_PROMPT_NO_EXPLORE
        prompt_label = "with exploration" if explore else "no exploration (direct refinement)"
        timeout_label = "none" if agent_timeout is None else f"{agent_timeout}s"
        print(f"  [3/5] Sending task prompt to agent ({prompt_label}, timeout: {timeout_label})…")        
        print( "        This will take several minutes - watch the canvas for activity.")
        if dry_run:
            print("        [dry-run] would POST /chat with task prompt")
            print(f"\n--- TASK PROMPT PREVIEW ({prompt_label}) ---")
            print(prompt[:400] + "…")
            print("--- END PREVIEW ---\n")
            summary["status"] = "dry-run"
            return summary

        agent_resp = _post(f"{base_agent}/chat", {"message": prompt}, timeout=agent_timeout)
        print(f"        Agent finished. Reply length: {len(agent_resp.get('reply',''))} chars")
        model_name = agent_resp.get("model_name")
        if model_name:
            summary["model_name"] = model_name
            print(f"        Model: {model_name}")

        # ------------------------------------------------------------------
        # Step 4: Run host-side evaluation (canonical result file)
        # ------------------------------------------------------------------
        print("  [4/5] Running host-side evaluation (sem_histogram_error.py)…")
        eval_result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "sem_histogram_error.py"),
             "--paint-url", base_paint],
            capture_output=True, text=True, timeout=120, cwd=_PROJECT_ROOT,
        )
        console_chunks = [
            f"run_label={run_label}",
            f"seed={base_seed}",
            f"sample_seed={sample_seed}",
            f"randomization_seed={randomization_seed}",
            "\n--- agent reply ---\n" + agent_resp.get("reply", ""),
            "\n--- evaluation stdout ---\n" + eval_result.stdout,
            "\n--- evaluation stderr ---\n" + eval_result.stderr,
        ]
        if eval_result.returncode != 0:
            print(f"        [warn] Evaluation script exited with code {eval_result.returncode}", file=sys.stderr)
            print(f"        STDERR: {eval_result.stderr[:2000]}", file=sys.stderr)
        else:
            print("        Evaluation complete.")

        # ------------------------------------------------------------------
        # Step 5: Package artifacts
        # ------------------------------------------------------------------
        if skip_package:
            print("  [5/5] Skipping packaging (--skip-package).")
            summary["status"] = "completed_no_package"
        else:
            print("  [5/5] Packaging run artifacts…")
            pkg_cmd = [
                sys.executable,
                str(_PROJECT_ROOT / "package_run.py"),
                "--paint-url", base_paint,
                "--exploratory", "true" if explore else "false",
            ]
            if base_seed is not None:
                pkg_cmd += ["--seed", str(base_seed)]
            if sample_seed is not None:
                pkg_cmd += ["--sample-seed", str(sample_seed)]
            if randomization_seed is not None:
                pkg_cmd += ["--randomization-seed", str(randomization_seed)]
            if summary.get("sample_index") is not None:
                pkg_cmd += ["--sample-index", str(summary["sample_index"])]
            if summary.get("model_name"):
                pkg_cmd += ["--model-name", str(summary["model_name"])]

            pkg_result = subprocess.run(
                pkg_cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=_PROJECT_ROOT,
            )
            console_chunks.extend([
                "\n--- package stdout ---\n" + pkg_result.stdout,
                "\n--- package stderr ---\n" + pkg_result.stderr,
            ])
            if pkg_result.returncode != 0:
                print(f"        [warn] package_run.py exited with code {pkg_result.returncode}", file=sys.stderr)
                print(f"        STDERR: {pkg_result.stderr[:2000]}", file=sys.stderr)
            else:
                # Extract the run dir from package_run.py stdout
                for line in pkg_result.stdout.splitlines():
                    if "Run packaged" in line or "run_dir" in line.lower():
                        print(f"        {line.strip()}")
                        if "→" in line:
                            summary["run_dir"] = line.split("→")[-1].strip()

                if summary.get("run_dir"):
                    run_dir = Path(summary["run_dir"])
                    if not run_dir.is_absolute():
                        run_dir = (_PROJECT_ROOT / run_dir).resolve()
                    print(f"        Logs copied and manifest patched: {run_dir / 'run_manifest.json'}")

            summary["status"] = "completed"

        # ------------------------------------------------------------------
        # Read final score from latest result JSON
        # ------------------------------------------------------------------
        result_dir = _PROJECT_ROOT / "sem-service" / "histograms" / "result"
        candidates = sorted(result_dir.glob("result_hist_*.json"))
        if candidates:
            data = json.loads(candidates[-1].read_text())
            summary["score"] = data.get("score")
            summary["randomized_score"] = data.get("randomizedScore")
            summary["filter_adjustments"] = data.get("filterAdjustments")
            summary["vlm_snapshots"] = data.get("vlmSnapshots")
            print(f"\n  Score: {summary['score']}  (randomized: {summary.get('randomized_score')})")
            print(f"  Filter adjustments: {summary.get('filter_adjustments')}  |  VLM snapshots: {summary.get('vlm_snapshots')}")

    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        summary["error"] = "KeyboardInterrupt"
        print("\n  Interrupted.", file=sys.stderr)
        raise
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        print(f"\n  [ERROR] {exc}", file=sys.stderr)

    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Automated Case Study 1 runner - sample → randomize → agent → eval → package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs",        type=int,   default=1,                         help="Number of sequential runs (default: 1)")
    parser.add_argument("--source",      default="unlabeled", choices=["unlabeled", "labeled"], help="Dataset to sample from (default: unlabeled)")
    parser.add_argument("--paint-url",   default="http://localhost:3000",               help="sem-service base URL (default: localhost:3000)")
    parser.add_argument("--agent-url",   default="http://localhost:3001",               help="agent-api base URL (default: localhost:3001)")
    parser.add_argument("--timeout",     type=int,   default=0,                      help="Agent chat timeout in seconds (default: 3600 = 1 h)")
    parser.add_argument("--no-sample",   action="store_true",                           help="Skip image sampling; use whatever is already loaded")
    parser.add_argument("--no-explore",  action="store_true",                           help="Skip the 4 exploratory settings; go straight to refinement")
    parser.add_argument("--skip-package",action="store_true",                           help="Skip the package_run.py step")
    parser.add_argument("--delay",       type=int,   default=5,                         help="Seconds to wait between runs (default: 5)")
    parser.add_argument("--dry-run",     action="store_true",                           help="Print the task prompt and exit without calling the agent")
    parser.add_argument("--seed",        type=int,   default=None,                      help="Base seed for reproducible sample/filter randomization")
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    print("\nCase Study 1 - Automated Runner")
    print(f"  Runs        : {args.runs}")
    print(f"  Source      : {args.source}")
    print(f"  Paint URL   : {args.paint_url}")
    print(f"  Agent URL   : {args.agent_url}")
    print(f"  Timeout     : {args.timeout}s per run")
    print(f"  Sample image: {'no' if args.no_sample else 'yes'}")
    print(f"  Exploration : {'no (direct refinement)' if args.no_explore else 'yes (4 exploratory + 3 refinement)'}")
    print(f"  Seed        : {args.seed if args.seed is not None else 'unseeded'}")

    if not args.dry_run:
        paint_ok = _check_service(args.paint_url, "sem-service")
        agent_ok = _check_service(args.agent_url, "agent-api")
        if not paint_ok or not agent_ok:
            print("\nERROR: Required services are not reachable. Aborting.", file=sys.stderr)
            sys.exit(1)
        print("  Services    : OK\n")

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    all_summaries: list[dict] = []
    failed = 0

    for i in range(1, args.runs + 1):
        try:
            summary = run_once(
                run_index=i,
                paint_url=args.paint_url,
                agent_url=args.agent_url,
                source=args.source,
                skip_sample=args.no_sample,
                skip_package=args.skip_package,
                agent_timeout = None if args.timeout == 0 else args.timeout,
                explore=not args.no_explore,
                dry_run=args.dry_run,
                base_seed=args.seed,
            )
        except KeyboardInterrupt:
            print("\nAborted by user.")
            break

        all_summaries.append(summary)
        if summary["status"] not in ("completed", "dry-run"):
            failed += 1

        if i < args.runs:
            print(f"\n  Waiting {args.delay}s before next run…")
            time.sleep(args.delay)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    if args.dry_run or len(all_summaries) == 0:
        return

    print(f"\n{'='*60}")
    print(f"  SUMMARY - {len(all_summaries)} run(s), {failed} failed")
    print(f"{'='*60}")
    scores = [s["score"] for s in all_summaries if isinstance(s.get("score"), (int, float))]
    if scores:
        print(f"  Scores: {[round(s, 4) for s in scores]}")
        print(f"  Mean  : {round(sum(scores)/len(scores), 4)}")
        print(f"  Best  : {round(min(scores), 4)}")

    # Save multi-run summary JSON
    if len(all_summaries) > 1:
        out_dir = _PROJECT_ROOT / "outputs" / "case_study_1"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        summary_path = out_dir / f"multi_run_summary_{ts}.json"
        summary_path.write_text(json.dumps(all_summaries, indent=2))
        print(f"\n  Multi-run summary → {summary_path}")


if __name__ == "__main__":
    main()
