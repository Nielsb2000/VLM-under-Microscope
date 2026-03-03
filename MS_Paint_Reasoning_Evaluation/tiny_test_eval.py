# Tiny Test Eval Script for MS Paint Reasoning Evaluation
import sys
from PIL import Image
try:
    from deepagents import create_deep_agent
except ImportError:
    create_deep_agent = None
from langchain_openai import ChatOpenAI

def main():
    import os
    # Usage: python tiny_test_eval.py <image_type> <blur_type> <img_index> <q_index> <model_name> [reasoning_effort] [reasoning_summary]
    if len(sys.argv) < 6:
        print("Usage: python tiny_test_eval.py <image_type> <blur_type> <img_index> <q_index> <model_name> [reasoning_effort] [reasoning_summary]")
        sys.exit(1)
    image_type = sys.argv[1]
    blur_type = sys.argv[2]
    if blur_type == 'no_blur':
        blur_type = 'none'
    img_index = sys.argv[3]
    q_index = int(sys.argv[4])
    model_name = sys.argv[5]
    reasoning_effort = sys.argv[6] if len(sys.argv) > 6 else 'medium'
    reasoning_summary = sys.argv[7] if len(sys.argv) > 7 else 'auto'

    # If using gpt-4o, ignore reasoning_effort in model creation
    if model_name == "gpt-4o":
        llm_kwargs = {"model": model_name, "api_key": os.environ.get("OPENAI_API_KEY")}
    else:
        llm_kwargs = {"model": model_name, "reasoning_effort": reasoning_effort, "api_key": os.environ.get("OPENAI_API_KEY")}

    # Minimal directory logic
    IMAGE_TYPE_FOLDERS = {
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
    IMG_ROOT = "MS_Paint_Reasoning_Evaluation/MS_paint_images"
    QUESTION_ROOT = os.path.join(IMG_ROOT, "MS paint questions")
    folder = IMAGE_TYPE_FOLDERS.get((image_type, blur_type))
    img_dir = os.path.join(IMG_ROOT, folder)
    img_file = f"img{img_index}.png"
    image_path = os.path.join(img_dir, img_file)
    q_file = f"Questions{img_index}.txt"
    q_path = os.path.join(QUESTION_ROOT, q_file)
    with open(q_path, "r") as f:
        questions = [q.strip() for q in f if q.strip()]
    question = questions[q_index - 1]
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if model_name == "gpt-4o":
        llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)
        print("[INFO] reasoning_effort parameter ignored for gpt-4o.")
    else:
        llm = ChatOpenAI(model=model_name, reasoning_effort=reasoning_effort, api_key=OPENAI_API_KEY)

    # Load image before using img.save(...)
    img = Image.open(image_path)
    root_dir = os.path.abspath(os.path.dirname(__file__))
    skills_dir_repr = os.path.join(root_dir, "skills")

    from io import BytesIO
    import base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    content = [
        {"type": "text", "text": question},
        {"type": "image", "base64": image_base64, "mime_type": "image/png"}
    ]

    messages = []
    answer = None
    elapsed_time = 0.0

    if create_deep_agent is None:
        raise RuntimeError("deepagents package is not available but skills mode was requested.")
    try:
        from deepagents.backends.filesystem import FilesystemBackend
    except Exception as e:
        raise RuntimeError(
            "Could not import FilesystemBackend from deepagents.backends.filesystem. "
            "Make sure deepagents is installed and the backend module is available."
        ) from e

    backend = FilesystemBackend(root_dir=root_dir, virtual_mode=False)
    skills_relative = ["skills"]

    print(f"[INFO] FilesystemBackend root_dir: {root_dir}")
    print(f"[INFO] Skills (relative to root_dir): {skills_relative}")

    agent = create_deep_agent(
        model=llm,
        system_prompt=(
            "You are an MS Paint reasoning assistant.\n"
            "2. If the question is about shape recognition (e.g., identifying, describing, or reasoning about shapes), you MUST use the recognizing-shapes/SKILL.md skill as referenced by the master-skill.\n"
            "3. For every image domain (color, grayscale, inverted grayscale) and for every shape recognition task, you MUST use read_file to open and read the example images provided in the relevant skill files.\n"
            "4. When referencing a skill, open and describe the provided example image(s) for that domain or shape using read_file, and compare them to the input image.\n"
            "5. For each skill you use, cite the skill file in your reasoning.\n"
            "6. Do NOT use scripts, write files, or use python code.\n"
            "7. The input image is provided inline; do NOT load it from the filesystem. Only use read_file for example/reference images inside skill folders.\n"
            "OUTPUT: Return a single valid JSON object with exactly these keys: 'reasoning', 'final_answer', 'token_usage', 'elapsed_time'. No extra text.\n"
            "Example: {\"reasoning\": \"...\", \"final_answer\": \"...\", \"token_usage\": 123, \"elapsed_time\": 2.34}"
        ),
        skills=skills_relative,
        backend=backend,
        tools=[],
        debug=False,
    )

    agent_input = {
        "messages": [
            {"role": "user", "content": content}
        ],
        "reasoning": {
            "effort": reasoning_effort,
            "summary": reasoning_summary,
        }
    }
    import time
    config = {"configurable": {"thread_id": "default"}}
    start_time = time.time()
    response = agent.invoke(agent_input, config=config)
    elapsed_time = time.time() - start_time
    messages = response.get("messages", [])
    answer = messages[-1].content if messages else None



    import json, re

    # Parse model output JSON if possible
    parsed = None
    if answer:
        try:
            parsed = json.loads(answer)
        except Exception:
            m = re.search(r'\{.*\}', answer, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

    # Extract token usage and model name from LangChain message metadata
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

    # Always use measured elapsed time
    elapsed_time_measured = float(elapsed_time)

    # If model returned JSON, overwrite token_usage/elapsed_time with real values
    if isinstance(parsed, dict):
        parsed["token_usage"] = token_usage if token_usage is not None else parsed.get("token_usage")
        parsed["elapsed_time"] = elapsed_time_measured
    else:
        # If it didn't return JSON, create one so your eval output is consistent
        parsed = {
            "reasoning": str(answer),
            "final_answer": str(answer),
            "token_usage": token_usage,
            "elapsed_time": elapsed_time_measured,
        }

    # Print
    print("\nReasoning:\n", parsed.get("reasoning", "[Missing reasoning]"))
    print("\nFinal Answer:\n", parsed.get("final_answer", "[Missing final_answer]"))
    import json as _json
    print("\nToken Usage:\n", _json.dumps(parsed.get("token_usage", "n/a")))
    print("\nElapsed Time (s):\n", round(parsed.get("elapsed_time", elapsed_time_measured), 3))
    if model_used:
        print("\n[INFO] Model used (from response metadata):", model_used)

    # Save answer to debug file (overwrite each run)
    out_path = os.path.join(os.path.dirname(__file__), f"tiny_test_debug_skills.txt")
    with open(out_path, "w") as f:
        f.write("--- Tiny Test Eval Result ---\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Model: {model_name}\n")
        if model_used:
            f.write(f"Model used (from response metadata): {model_used}\n")
        f.write(f"Blur type: {blur_type}\n")
        f.write(f"Skills directory used: {skills_dir_repr}\n")
        f.write(f"Image processed: {img is not None}\n")
        if parsed:
            f.write("\nReasoning:\n" + str(parsed.get("reasoning", "[Missing reasoning]")) + "\n")
            f.write("\nFinal Answer:\n" + str(parsed.get("final_answer", "[Missing final_answer]")) + "\n")
            f.write("\nToken Usage:\n" + str(parsed.get("token_usage", token_usage)) + "\n")
            f.write("\nElapsed Time (s):\n" + str(parsed.get("elapsed_time", round(elapsed_time, 2))) + "\n")
        else:
            f.write("\nFull Reasoning and Answer:\n" + str(answer) + "\n")
            f.write(f"\nToken Usage:\n{token_usage}\n")
            f.write(f"\nElapsed Time (s):\n{round(elapsed_time, 2)}\n")
    print(f"Answer saved to: {out_path}")

if __name__ == "__main__":
    main()