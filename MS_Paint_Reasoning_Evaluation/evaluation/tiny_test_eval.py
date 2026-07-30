# Tiny Test Eval Script for MS Paint Reasoning Evaluation
import argparse
import base64
from io import BytesIO
import json
import os
import random
import re
import sys
import time

from PIL import Image

from gpt55_responses_agent import (
    run_gpt55_no_skills,
    run_gpt55_with_skills,
)


RESPONSES_API_MODELS = {"gpt-5.5"}
CHAT_REASONING_MODELS = {"gpt-5.1", "gpt-5.2", "gpt-5-pro"}


def apply_seed(seed: int | None) -> None:
    """Best-effort local reproducibility for Python-side randomness."""
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def create_chat_openai(model_name: str, reasoning_effort: str, api_key: str | None, seed: int | None):
    """Create LangChain ChatOpenAI for models that are safe on Chat Completions.

    Import LangChain lazily so GPT-5.5 Responses API runs do not pay the
    DeepAgents/LangChain import cost and cannot get interrupted during it.
    """
    from langchain_openai import ChatOpenAI

    if model_name in RESPONSES_API_MODELS:
        raise ValueError(
            f"{model_name} must use the Responses API for reasoning/tool use; "
            "do not construct it through ChatOpenAI for this eval path."
        )

    kwargs = {"model": model_name, "api_key": api_key}
    if model_name != "gpt-4o":
        kwargs["reasoning_effort"] = reasoning_effort
    if seed is not None:
        # LangChain may warn that seed should be explicit, but this keeps behavior aligned with old runs.
        kwargs["model_kwargs"] = {"seed": seed}
    return ChatOpenAI(**kwargs)


def completion_with_seed_fallback(client, *, seed: int | None, **kwargs):
    """Call OpenAI Chat Completions with seed, retrying without it if unsupported."""
    if seed is None:
        return client.chat.completions.create(**kwargs)
    try:
        return client.chat.completions.create(seed=seed, **kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        unsupported_seed = "seed" in msg and (
            "unsupported" in msg or "unrecognized" in msg or "unknown" in msg or "extra" in msg
        )
        if not unsupported_seed:
            raise
        print("[WARN] This model/API path did not accept seed; retrying without seed.", file=sys.stderr)
        return client.chat.completions.create(**kwargs)


def parse_json_object(answer: str | None):
    if not answer:
        return None
    try:
        return json.loads(answer)
    except Exception:
        match = re.search(r"\{.*\}", answer, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def normalize_parsed_answer(answer: str | None, token_usage, elapsed_time: float) -> dict:
    parsed = parse_json_object(answer)
    if isinstance(parsed, dict):
        parsed["token_usage"] = token_usage if token_usage is not None else parsed.get("token_usage")
        parsed["elapsed_time"] = float(elapsed_time)
        return parsed
    return {
        "reasoning": str(answer),
        "final_answer": str(answer),
        "token_usage": token_usage,
        "elapsed_time": float(elapsed_time),
    }


def print_eval_result(parsed: dict, elapsed_time: float, model_used: str | None = None) -> None:
    print("\nReasoning:\n", parsed.get("reasoning", "[Missing reasoning]"))
    print("\nFinal Answer:\n", parsed.get("final_answer", "[Missing final_answer]"))
    print("\nToken Usage:\n", json.dumps(parsed.get("token_usage", "n/a")))
    print("\nElapsed Time (s):\n", round(parsed.get("elapsed_time", elapsed_time), 3))
    if model_used:
        print("\n[INFO] Model used (from response metadata):", model_used)


def main():
    parser = argparse.ArgumentParser(description="Tiny Test Eval Script for MS Paint Reasoning Evaluation")
    parser.add_argument("image_type")
    parser.add_argument("blur_type")
    parser.add_argument("img_index")
    parser.add_argument("q_index", type=int)
    parser.add_argument("model_name")
    parser.add_argument("reasoning_effort", nargs="?", default="medium")
    parser.add_argument("--skills", choices=["yes", "no"], default="yes")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    apply_seed(args.seed)

    image_type = args.image_type
    blur_type = "none" if args.blur_type == "no_blur" else args.blur_type
    img_index = args.img_index
    q_index = args.q_index
    model_name = args.model_name
    reasoning_effort = args.reasoning_effort
    skills = args.skills == "yes"

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_root = os.path.join(root_dir, "MS_paint_images")
    question_root = os.path.join(img_root, "MS paint questions")

    image_type_folders = {
        ("color", "none"): "original_images",
        ("greyscale", "none"): "greyscale_images",
        ("inverted_greyscale", "none"): "inverted_greyscale_images",
        ("color", "med_blur"): "original_med_blur_images",
        ("greyscale", "med_blur"): "med_blur_greyscale_images",
        ("inverted_greyscale", "med_blur"): "med_blur_inverted_greyscale_images",
        ("color", "heavy_blur"): "original_heavy_blur_images",
        ("greyscale", "heavy_blur"): "heavy_blur_greyscale_images",
        ("inverted_greyscale", "heavy_blur"): "heavy_blur_inverted_greyscale_images",
    }
    folder = image_type_folders.get((image_type, blur_type))
    if folder is None:
        raise ValueError(f"Unsupported image/blur combination: {image_type}, {blur_type}")

    image_path = os.path.join(img_root, folder, f"img{img_index}.png")
    q_path = os.path.join(question_root, f"Questions{img_index}.txt")
    with open(q_path, "r", encoding="utf-8") as f:
        questions = [q.strip() for q in f if q.strip()]
    question = questions[q_index - 1]

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if model_name == "gpt-4o":
        print("[INFO] reasoning_effort parameter ignored for gpt-4o.")

    img = Image.open(image_path)
    skills_dir_repr = os.path.join(root_dir, "skills")

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    content = [
        {"type": "text", "text": question},
        {"type": "image", "base64": image_base64, "mime_type": "image/png"},
    ]

    deepagent_system_prompt = (
        "You are an MS Paint reasoning assistant with access to specialized skills.\n"
        "\n"
        "CRITICAL REQUIREMENT: Follow this EXACT workflow BEFORE answering:\n"
        "\n"
        "STEP 1: READ MASTER SKILL\n"
        "- MANDATORY: Call read_file to read skills/master-skill/SKILL.md\n"
        "- LIST ALL 4 SKILLS you find in the routing table\n"
        "\n"
        "STEP 2: ANALYZE INPUT TO DETERMINE SKILL ROUTING\n"
        "- Look at the PROVIDED IMAGE to determine its type:\n"
        "  * Does it have color? -> Use colored-images skill\n"
        "  * Only grayscale (no color)? -> Use grayscale-images skill\n"
        "  * Inverted/negative grayscale? -> Use inverted-grayscale-images skill\n"
        "- Read the QUESTION to check if it asks about shape identification:\n"
        "  * Keywords: 'shape', 'identify', 'what shape', 'circle', 'square', 'triangle', 'star', etc.\n"
        "  * If YES -> ALSO use recognizing-shapes skill\n"
        "\n"
        "STEP 3: READ RELEVANT SKILL FILE(S)\n"
        "- Call read_file for the domain skill (colored-images, grayscale-images, or inverted-grayscale-images)\n"
        "- If question is about shapes, ALSO call read_file for recognizing-shapes/SKILL.md\n"
        "- Read any example images referenced in the skill files if needed\n"
        "\n"
        "STEP 4: ANSWER THE QUESTION\n"
        "- Use the guidance from the skill files you read\n"
        "- Analyze the provided input image\n"
        "- Apply the protocols and reasoning methods from the skills\n"
        "\n"
        "IMPORTANT NOTES:\n"
        "- The input image is PROVIDED INLINE in the message. Use read_file ONLY for skill files and example images.\n"
        "- You WILL BE PENALIZED if you skip reading skills or don't follow the routing protocol.\n"
        "- In your reasoning, EXPLICITLY state which skills you accessed and why.\n"
        "- When answering a question always describe the shape you see without color information, then give the final answer. This ensures you are following the skills' guidance to identify shapes based on structure, not color.\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "Return a single valid JSON object with these exact keys: 'reasoning', 'final_answer', 'token_usage', 'elapsed_time'\n"
        "Example: {\"reasoning\": \"Step 1: Read master-skill/SKILL.md and found 4 skills. Step 2: Image has color, question asks about shapes. Step 3: Read colored-images and recognizing-shapes skills...\", \"final_answer\": \"...\", \"token_usage\": 123, \"elapsed_time\": 2.34}"
    )

    chatcompletion_system_prompt = (
        "You are an MS Paint reasoning assistant.\n"
        "Use concise reasoning before giving the final answer.\n"
        "If the question asks for an updated image, describe the changes and return the modified image if possible.\n"
        "OUTPUT: Return a single valid JSON object with exactly these keys: 'reasoning', 'final_answer', 'token_usage', 'elapsed_time'. No extra text.\n"
        "Example: {\"reasoning\": \"...\", \"final_answer\": \"...\", \"token_usage\": 123, \"elapsed_time\": 2.34}"
    )

    if skills:
        if model_name in RESPONSES_API_MODELS:
            import openai

            print(f"[INFO] Using Responses API path for {model_name} + skills + reasoning_effort.")
            print(f"[INFO] Skills root_dir: {root_dir}")
            print(f"[INFO] Best-effort seed: {args.seed}")
            openai_client = openai.OpenAI(api_key=openai_api_key)
            start_time = time.time()
            answer, token_usage, response = run_gpt55_with_skills(
                client=openai_client,
                model_name=model_name,
                root_dir=root_dir,
                system_prompt=deepagent_system_prompt,
                question=question,
                image_base64=image_base64,
                reasoning_effort=reasoning_effort,
                seed=args.seed,
            )
            elapsed_time = time.time() - start_time
            parsed = normalize_parsed_answer(answer, token_usage, elapsed_time)
            model_used = getattr(response, "model", None)
            print_eval_result(parsed, elapsed_time, model_used=model_used)
        else:
            try:
                from deepagents import create_deep_agent
                from deepagents.backends.filesystem import FilesystemBackend
            except ImportError as exc:
                raise ImportError("deepagents is required for --skills yes") from exc

            llm = create_chat_openai(model_name, reasoning_effort, openai_api_key, args.seed)
            backend = FilesystemBackend(root_dir=root_dir, virtual_mode=False)
            skills_relative = ["skills/master-skill"]

            print(f"[INFO] FilesystemBackend root_dir: {root_dir}")
            print(f"[INFO] Skills (relative to root_dir): {skills_relative}")
            print(f"[INFO] Best-effort seed: {args.seed}")

            agent = create_deep_agent(
                model=llm,
                system_prompt=deepagent_system_prompt,
                skills=skills_relative,
                backend=backend,
                debug=False,
            )
            agent_input = {
                "messages": [{"role": "user", "content": content}],
                "reasoning": {"effort": reasoning_effort, "summary": "auto"},
            }
            config = {"configurable": {"thread_id": f"seed-{args.seed}-img{img_index}-q{q_index}-{model_name}-{args.skills}"}}
            start_time = time.time()
            response = agent.invoke(agent_input, config=config)
            elapsed_time = time.time() - start_time
            messages = response.get("messages", [])
            answer = messages[-1].content if messages else None

            last_msg = messages[-1] if messages else None
            token_usage = None
            model_used = None
            if last_msg is not None:
                usage = getattr(last_msg, "usage_metadata", None)
                if isinstance(usage, dict):
                    token_usage = usage
                if token_usage is None:
                    rm = getattr(last_msg, "response_metadata", None)
                    if isinstance(rm, dict):
                        token_usage = rm.get("usage") or rm.get("token_usage") or rm.get("usage_metadata")
                        model_used = rm.get("model") or rm.get("model_name")

            parsed = normalize_parsed_answer(answer, token_usage, elapsed_time)
            print_eval_result(parsed, elapsed_time, model_used=model_used)

        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            logs_dir,
            f"debug_{image_type}_{blur_type}_img{img_index}_q{q_index}_{model_name}_skills_seed{args.seed}_{ts}.txt",
        )
        header = "--- Tiny Test Eval Result ---\n"
    else:
        import openai

        print(f"[INFO] Best-effort seed: {args.seed}")
        openai_client = openai.OpenAI(api_key=openai_api_key)
        start_time = time.time()

        if model_name in RESPONSES_API_MODELS:
            print(f"[INFO] Using Responses API path for {model_name} + no skills + reasoning_effort.")
            answer, token_usage, response = run_gpt55_no_skills(
                client=openai_client,
                model_name=model_name,
                system_prompt=chatcompletion_system_prompt,
                question=question,
                image_base64=image_base64,
                reasoning_effort=reasoning_effort,
                seed=args.seed,
            )
        else:
            user_content = [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ]
            if model_name in CHAT_REASONING_MODELS:
                response = completion_with_seed_fallback(
                    openai_client,
                    seed=args.seed,
                    model=model_name,
                    messages=[
                        {"role": "system", "content": chatcompletion_system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    reasoning_effort=reasoning_effort,
                )
            else:
                response = completion_with_seed_fallback(
                    openai_client,
                    seed=args.seed,
                    model=model_name,
                    messages=[
                        {"role": "system", "content": chatcompletion_system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                )
            answer = response.choices[0].message.content if response.choices else None
            usage = getattr(response, "usage", None)
            token_usage = None
            if usage:
                token_usage = {
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }

        elapsed_time = time.time() - start_time
        parsed = normalize_parsed_answer(answer, token_usage, elapsed_time)
        print_eval_result(parsed, elapsed_time)

        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            logs_dir,
            f"debug_{image_type}_{blur_type}_img{img_index}_q{q_index}_{model_name}_noskills_seed{args.seed}_{ts}.txt",
        )
        header = "--- Tiny Test Eval Result (No Skills) ---\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(f"Image: {image_path}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Reasoning effort: {reasoning_effort}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Blur type: {blur_type}\n")
        if skills:
            f.write(f"Skills directory used: {skills_dir_repr}\n")
        f.write(f"Image processed: {img is not None}\n")
        f.write("\nReasoning:\n" + str(parsed.get("reasoning", "[Missing reasoning]")) + "\n")
        f.write("\nFinal Answer:\n" + str(parsed.get("final_answer", "[Missing final_answer]")) + "\n")
        f.write("\nToken Usage:\n" + str(parsed.get("token_usage")) + "\n")
        f.write("\nElapsed Time (s):\n" + str(parsed.get("elapsed_time")) + "\n")
    print(f"Answer saved to: {out_path}")


if __name__ == "__main__":
    main()
