# LLM answer checker for MS Paint Reasoning Evaluation
#
# Positional `model_name` is the answer-producing model, used to locate
# Results/.../answer_<model_name>_<skills>.txt. Use --judge-model to choose the
# separate grading model. This avoids accidentally using a slow experimental
# answer model, such as gpt-5.5, as the judge too.

import argparse
import json
import os
import re
import sys
from typing import Optional


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_answer_from_output(output: str) -> str:
    """Extract the final answer from tiny_test_eval.py stdout."""
    if not output:
        return ""

    # Preferred format produced by tiny_test_eval.py.
    match = re.search(r"Final Answer:\s*\n\s*(.*?)(?:\n\s*Token Usage:|\n\s*Elapsed Time|\nAnswer saved to:|\Z)", output, re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        if candidate and candidate not in {"[Missing final_answer]", "None"}:
            return candidate

    # JSON fallback when the model returned a raw JSON object.
    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            value = parsed.get("final_answer") or parsed.get("answer")
            if value:
                return str(value).strip()
    except Exception:
        pass

    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                value = parsed.get("final_answer") or parsed.get("answer")
                if value:
                    return str(value).strip()
        except Exception:
            pass

    # Last resort: preserve old behavior.
    return output.strip()


def load_nonempty_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_question_and_gt(root_dir: str, img_index: str, q_index: int) -> tuple[str, str, str, str]:
    questions_file = os.path.join(root_dir, "MS_paint_images", "MS paint questions", f"Questions{img_index}.txt")
    answers_file = os.path.join(root_dir, "MS_paint_images", "MS paint answers", f"Answers{img_index}.txt")
    questions = load_nonempty_lines(questions_file)
    answers = load_nonempty_lines(answers_file)
    question = questions[q_index - 1]
    gt_answer = answers[q_index - 1]
    return question, gt_answer, questions_file, answers_file


def answer_file_path(root_dir: str, image_type: str, blur_level: str, reasoning_effort: str, img_index: str, q_index: int, model_name: str, skills: str) -> str:
    skills_tag = "skills" if skills == "yes" else "noskills"
    return os.path.join(
        root_dir,
        "Results",
        f"{image_type}_{blur_level}_{reasoning_effort}",
        f"img{img_index}",
        f"q{q_index}",
        f"answer_{model_name}_{skills_tag}.txt",
    )


def quick_heuristic_match(model_answer: str, gt_answer: str) -> Optional[int]:
    """Cheap exact-ish match before spending a judge call."""
    a = normalize_text(model_answer)
    g = normalize_text(re.sub(r"^answer\s*\d+\s*:\s*", "", gt_answer, flags=re.IGNORECASE))
    if not a or not g:
        return None
    if g in a:
        return 1
    return None


def judge_with_chat_completions(client, *, judge_model: str, question: str, gt_answer: str, model_answer: str, use_seed: bool, seed: Optional[int]) -> str:
    system = (
        "You are a strict but fair grader for a visual question-answering benchmark. "
        "Return only 1 if the model answer is semantically correct according to the ground truth, otherwise return only 0. "
        "Ignore harmless wording differences. Do not explain."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Ground truth answer:\n{gt_answer}\n\n"
        f"Model answer:\n{model_answer}\n\n"
        "Is the model answer correct? Return exactly 1 or 0."
    )
    kwargs = dict(
        model=judge_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=5,
    )
    if use_seed and seed is not None:
        try:
            response = client.chat.completions.create(seed=seed, **kwargs)
        except Exception as exc:
            print(f"[WARN] Judge model did not accept seed; retrying without seed. {exc}", file=sys.stderr)
            response = client.chat.completions.create(**kwargs)
    else:
        response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content.strip() if response.choices else "0"
    return "1" if text.startswith("1") else "0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check one MS Paint model answer against ground truth.")
    parser.add_argument("img_index")
    parser.add_argument("q_index", type=int)
    parser.add_argument("model_name", help="Answer-producing model name used to locate the answer file.")
    parser.add_argument("image_type")
    parser.add_argument("blur_level")
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--skills", choices=["yes", "no"], default="yes")
    parser.add_argument("--judge-model", default="gpt-4o", help="Separate model used as the grader.")
    parser.add_argument("--seed", type=int, default=None, help="Accepted for CLI compatibility. Not used unless --use-judge-seed is set.")
    parser.add_argument("--use-judge-seed", action="store_true", help="Try passing --seed to the judge call.")
    parser.add_argument("--no-heuristic", action="store_true", help="Disable cheap exact-ish match before LLM grading.")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    answer_path = answer_file_path(
        root_dir,
        args.image_type,
        args.blur_level,
        args.reasoning_effort,
        args.img_index,
        args.q_index,
        args.model_name,
        args.skills,
    )

    question, gt_answer, questions_file, answers_file = get_question_and_gt(root_dir, args.img_index, args.q_index)

    print(f"[DEBUG] Questions file: {questions_file}", file=sys.stderr)
    print(f"[DEBUG] GT answers file: {answers_file}", file=sys.stderr)
    print(f"[DEBUG] Model answer file: {answer_path}", file=sys.stderr)
    print(f"[DEBUG] Judge model: {args.judge_model}", file=sys.stderr)
    print(f"[DEBUG] Loaded question: {question}", file=sys.stderr)

    if not os.path.exists(answer_path):
        print(f"[ERROR] Model answer file not found: {answer_path}", file=sys.stderr)
        print("0")
        return

    with open(answer_path, "r", encoding="utf-8") as f:
        raw_output = f.read()
    model_answer = extract_answer_from_output(raw_output)
    print(f"[DEBUG] Extracted model answer: {model_answer}", file=sys.stderr)
    print(f"[DEBUG] Loaded GT answer line: {gt_answer}", file=sys.stderr)

    if not model_answer:
        print("[ERROR] No model answer found.", file=sys.stderr)
        print("0")
        return

    if not args.no_heuristic:
        heuristic = quick_heuristic_match(model_answer, gt_answer)
        if heuristic is not None:
            print(f"[DEBUG] Heuristic result: {heuristic}", file=sys.stderr)
            print(str(heuristic))
            return

    import openai

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    result = judge_with_chat_completions(
        client,
        judge_model=args.judge_model,
        question=question,
        gt_answer=gt_answer,
        model_answer=model_answer,
        use_seed=args.use_judge_seed,
        seed=args.seed,
    )
    print(f"[DEBUG] LLM raw result: {result}", file=sys.stderr)
    print(result)


if __name__ == "__main__":
    main()
