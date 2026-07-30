# Batch evaluation script for MS Paint Reasoning using tiny_test_eval.py and llm_check_answer.py

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
import json
import os
import re
import subprocess
import sys
from typing import Optional

IMG_DIR = "MS_paint_images"
QUESTIONS_DIR = os.path.join(IMG_DIR, "MS paint questions")
RESULTS_DIR = "Results"

IMAGE_TYPE_FOLDERS = {
    ("color", "none"): "original_images",
    ("greyscale", "none"): "greyscale_images",
    ("inverted_greyscale", "none"): "inverted_greyscale_images",
    ("color", "no_blur"): "original_images",
    ("greyscale", "no_blur"): "greyscale_images",
    ("inverted_greyscale", "no_blur"): "inverted_greyscale_images",
    ("color", "med_blur"): "original_med_blur_images",
    ("greyscale", "med_blur"): "med_blur_greyscale_images",
    ("inverted_greyscale", "med_blur"): "med_blur_inverted_greyscale_images",
    ("color", "heavy_blur"): "original_heavy_blur_images",
    ("greyscale", "heavy_blur"): "heavy_blur_greyscale_images",
    ("inverted_greyscale", "heavy_blur"): "heavy_blur_inverted_greyscale_images",
}


def get_image_type_folders():
    return IMAGE_TYPE_FOLDERS


def parse_token_usage(output: str) -> Optional[dict]:
    """Extract token usage from tiny_test_eval.py stdout."""
    m_start = re.search(r"Token Usage:\s*(\{)", output)
    if m_start:
        start_idx = m_start.start(1)
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(output)):
            if output[i] == "{":
                brace_count += 1
            elif output[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        dict_str = output[start_idx:end_idx]
        try:
            return json.loads(dict_str)
        except Exception:
            try:
                return ast.literal_eval(dict_str)
            except Exception:
                pass

    token_usage = {}
    m_input = re.search(r"['\"]?input_tokens['\"]?\s*:\s*([0-9]+)", output)
    m_output = re.search(r"['\"]?output_tokens['\"]?\s*:\s*([0-9]+)", output)
    m_total = re.search(r"['\"]?total_tokens['\"]?\s*:\s*([0-9]+)", output)
    if m_input:
        token_usage["input_tokens"] = int(m_input.group(1))
    if m_output:
        token_usage["output_tokens"] = int(m_output.group(1))
    if m_total:
        token_usage["total_tokens"] = int(m_total.group(1))
    return token_usage or None


def parse_elapsed_time(output: str) -> Optional[float]:
    m = re.search(r"Elapsed Time \(s\):\s*([0-9.]+)", output)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


@dataclass(frozen=True)
class EvalJob:
    image_type: str
    blur_level: str
    reasoning_effort: str
    skills: str
    img_index: str
    q_idx: int
    img_name: str
    model_name: str


def run_one_job(
    job: EvalJob,
    base_dir: str,
    results_dir_root: str,
    seed: Optional[int],
    judge_model: str,
    answer_timeout: int,
    judge_timeout: int,
) -> tuple[str, tuple]:
    """Run one model answer generation and its LLM judge check."""
    label = (
        f"{job.image_type}/{job.blur_level}/{job.reasoning_effort}/"
        f"skills={job.skills}/{job.img_name}/Q{job.q_idx}/{job.model_name}"
    )
    print(f"\n--- Evaluating {label} ---", flush=True)

    tiny_test_path = os.path.join(base_dir, "evaluation", "tiny_test_eval.py")
    cmd = [
        sys.executable,
        tiny_test_path,
        job.image_type,
        job.blur_level,
        job.img_index,
        str(job.q_idx),
        job.model_name,
        job.reasoning_effort,
        "--skills",
        job.skills,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=answer_timeout)
        output = result.stdout
        err = result.stderr
        if err:
            print(f"[tiny_test_eval.py stderr] {label}\n{err}", flush=True)
        if result.returncode != 0:
            print(f"[ERROR] tiny_test_eval.py failed for {label} with exit {result.returncode}", flush=True)
    except Exception as exc:
        print(f"[ERROR] Exception running tiny_test_eval.py for {label}: {exc}", flush=True)
        output = ""

    answer_dir = os.path.join(
        results_dir_root,
        f"{job.image_type}_{job.blur_level}_{job.reasoning_effort}",
        f"img{job.img_index}",
        f"q{job.q_idx}",
    )
    os.makedirs(answer_dir, exist_ok=True)
    skills_tag = "skills" if job.skills == "yes" else "noskills"
    answer_file = os.path.join(answer_dir, f"answer_{job.model_name}_{skills_tag}.txt")
    with open(answer_file, "w", encoding="utf-8") as out_f:
        out_f.write(output)
    print(f"[INFO] Answer written to {answer_file}", flush=True)

    token_usage = parse_token_usage(output)
    elapsed_time = parse_elapsed_time(output)

    llm_check_path = os.path.join(base_dir, "evaluation", "llm_check_answer.py")
    cmd_check = [
        sys.executable,
        llm_check_path,
        job.img_index,
        str(job.q_idx),
        job.model_name,
        job.image_type,
        job.blur_level,
        "--reasoning-effort",
        job.reasoning_effort,
        "--skills",
        job.skills,
        "--judge-model",
        judge_model,
    ]
    if seed is not None:
        cmd_check.extend(["--seed", str(seed)])

    print(f"[DEBUG] Running llm_check_answer.py for {label}", flush=True)
    try:
        result_check = subprocess.run(cmd_check, capture_output=True, text=True, timeout=judge_timeout)
        if result_check.stderr:
            print(f"[llm_check_answer.py stderr] {label}\n{result_check.stderr}", flush=True)
        llm_result = result_check.stdout.strip()
        if llm_result and llm_result[0] in {"0", "1"}:
            llm_result = llm_result[0]
        else:
            llm_result = "0"
    except Exception as exc:
        print(f"[ERROR] LLM check failed for {label}: {exc}", flush=True)
        llm_result = "0"

    input_tokens = None
    output_tokens = None
    if isinstance(token_usage, dict):
        input_tokens = token_usage.get("input_tokens") or token_usage.get("prompt_tokens")
        output_tokens = token_usage.get("output_tokens") or token_usage.get("completion_tokens")

    result_key = f"{job.img_name}_Q{job.q_idx}_{job.model_name}"
    print(
        f"[DONE] {label} | correct={llm_result} | input_tokens={input_tokens} | "
        f"output_tokens={output_tokens} | elapsed_time={elapsed_time}",
        flush=True,
    )
    return result_key, (int(llm_result), (input_tokens, output_tokens), elapsed_time)


def collect_jobs_for_configuration(
    blur_level: str,
    image_type: str,
    reasoning_effort: str,
    skills: str,
    models: list[str],
    img_dir_root: str,
    questions_dir: str,
    smoke: bool,
) -> list[EvalJob]:
    folder = IMAGE_TYPE_FOLDERS.get((image_type, blur_level))
    if not folder:
        print(f"[ERROR] Unsupported combination: {image_type}, {blur_level}")
        return []

    img_dir = os.path.join(img_dir_root, folder)
    if not os.path.exists(img_dir):
        print(f"[ERROR] Image directory not found: {img_dir}")
        return []

    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
    if smoke:
        img_files = [f for f in img_files if f == "img1.png"]
        if not img_files:
            print(f"[WARN] Smoke test mode enabled but img1.png not found in {img_dir}")
            return []

    jobs = []
    for img_file in img_files:
        img_name = os.path.splitext(img_file)[0]
        img_index = img_name[3:]
        q_path = os.path.join(questions_dir, f"Questions{img_index}.txt")
        if not os.path.exists(q_path):
            print(f"[WARN] No questions file for {img_file}, skipping.")
            continue
        with open(q_path, "r", encoding="utf-8") as f:
            questions = [q.strip() for q in f if q.strip()]
        for q_idx, _question in enumerate(questions, 1):
            for model_name in models:
                if model_name == "gpt-4o" and reasoning_effort != "low":
                    print(
                        f"[SKIP] gpt-4o ignores reasoning_effort; skipping effort={reasoning_effort} "
                        f"for {image_type}/{blur_level}/skills={skills}",
                        flush=True,
                    )
                    continue
                jobs.append(
                    EvalJob(
                        image_type=image_type,
                        blur_level=blur_level,
                        reasoning_effort=reasoning_effort,
                        skills=skills,
                        img_index=img_index,
                        q_idx=q_idx,
                        img_name=img_name,
                        model_name=model_name,
                    )
                )
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate MS Paint Reasoning using tiny_test_eval.py and llm_check_answer.py."
    )
    parser.add_argument("--blur-levels", nargs="+", default=["no_blur"], choices=["no_blur", "med_blur", "heavy_blur"])
    parser.add_argument("--image-types", nargs="+", default=["color"], choices=["color", "greyscale", "inverted_greyscale"])
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-5.1", "gpt-5.2", "gpt-5.5"], choices=["gpt-4o", "gpt-5.1", "gpt-5.2", "gpt-5.5"])
    parser.add_argument("--reasoning-effort", nargs="+", default=["low"], choices=["low", "medium", "high"])
    parser.add_argument("--skills", nargs="+", choices=["yes", "no"], default=["yes"])
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode: run only on img1 for quick testing.")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of concurrent eval+judge jobs.")
    parser.add_argument("--judge-model", default="gpt-4o", help="Model used only for answer judging. The positional model stays the answer model.")
    parser.add_argument("--answer-timeout", type=int, default=300, help="Timeout in seconds for each tiny_test_eval.py subprocess.")
    parser.add_argument("--judge-timeout", type=int, default=60, help="Timeout in seconds for each llm_check_answer.py subprocess.")
    parser.add_argument("--seed", type=int, default=12345, help="Best-effort seed passed to model calls and local RNGs.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir_root = os.path.join(base_dir, "MS_paint_images")
    questions_dir = os.path.join(img_dir_root, "MS paint questions")
    results_dir_root = os.path.join(base_dir, "Results")
    dashboard_data_dir = os.path.join(results_dir_root, "dashboard_data")
    os.makedirs(dashboard_data_dir, exist_ok=True)

    blur_levels = ["no_blur" if b == "none" else b for b in args.blur_levels]
    max_workers = max(1, args.max_workers)
    if "gpt-5.5" in args.models and max_workers > 1:
        print(
            f"[INFO] gpt-5.5 Responses API runs can be slow. If you need easier Ctrl-C handling, rerun with --max-workers 1.",
            flush=True,
        )

    for blur_level, image_type, reasoning_effort, skills in product(
        blur_levels, args.image_types, args.reasoning_effort, args.skills
    ):
        jobs = collect_jobs_for_configuration(
            blur_level=blur_level,
            image_type=image_type,
            reasoning_effort=reasoning_effort,
            skills=skills,
            models=args.models,
            img_dir_root=img_dir_root,
            questions_dir=questions_dir,
            smoke=args.smoke,
        )
        if not jobs:
            print(f"[WARN] No jobs for {image_type}/{blur_level}/{reasoning_effort}/skills={skills}")
            continue

        print(
            f"\n[INFO] Running {len(jobs)} eval+judge jobs with max_workers={max_workers} "
            f"for {image_type}/{blur_level}/{reasoning_effort}/skills={skills}",
            flush=True,
        )

        results_dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(run_one_job, job, base_dir, results_dir_root, args.seed, args.judge_model, args.answer_timeout, args.judge_timeout): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    key, value = future.result()
                    results_dict[key] = value
                except Exception as exc:
                    print(f"[ERROR] Job crashed: {job} | {exc}", flush=True)

        results_dict = dict(sorted(results_dict.items()))
        models_str = "-".join(args.models)
        skills_tag = "skills" if skills == "yes" else "no_skills"
        result_file_name = f"llm_results_{image_type}_{blur_level}_{models_str}_{reasoning_effort}_{skills_tag}.json"
        result_file_path = os.path.join(dashboard_data_dir, result_file_name)
        with open(result_file_path, "w", encoding="utf-8") as rf:
            json.dump(results_dict, rf, indent=2)
        print(f"\n[INFO] LLM results written to {result_file_path}", flush=True)


if __name__ == "__main__":
    main()
