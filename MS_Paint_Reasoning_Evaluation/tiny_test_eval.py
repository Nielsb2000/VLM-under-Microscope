# Tiny Test Eval Script for MS Paint Reasoning Evaluation
import sys
from PIL import Image
try:
    from deepagents import create_deep_agent
except ImportError:
    create_deep_agent = None
from langchain_openai import ChatOpenAI
from deepagents.backends.filesystem import FilesystemBackend

def main():
    import os
    import time
    # Usage: python tiny_test_eval.py <image_type> <blur_type> <img_index> <q_index> <model_name> [reasoning_effort] [--skills yes|no]
    import argparse
    parser = argparse.ArgumentParser(description="Tiny Test Eval Script for MS Paint Reasoning Evaluation")
    parser.add_argument("image_type")
    parser.add_argument("blur_type")
    parser.add_argument("img_index")
    parser.add_argument("q_index", type=int)
    parser.add_argument("model_name")
    parser.add_argument("reasoning_effort", nargs="?", default="medium")
    parser.add_argument("--skills", choices=["yes", "no"], default="yes", help="Enable skills (deepagent) or no skills (chatcompletion)")
    args = parser.parse_args()

    image_type = args.image_type
    blur_type = args.blur_type
    if blur_type == 'no_blur':
        blur_type = 'none'
    img_index = args.img_index
    q_index = args.q_index
    model_name = args.model_name
    reasoning_effort = args.reasoning_effort
    skills = args.skills == "yes"

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



    backend = FilesystemBackend(root_dir=root_dir, virtual_mode=False)
    skills_relative = ["skills/master-skill"]

    print(f"[INFO] FilesystemBackend root_dir: {root_dir}")
    print(f"[INFO] Skills (relative to root_dir): {skills_relative}")


    # System prompt for deepagent (skills)
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
        "  * Does it have color? → Use colored-images skill\n"
        "  * Only grayscale (no color)? → Use grayscale-images skill\n"
        "  * Inverted/negative grayscale? → Use inverted-grayscale-images skill\n"
        "- Read the QUESTION to check if it asks about shape identification:\n"
        "  * Keywords: 'shape', 'identify', 'what shape', 'circle', 'square', 'triangle', 'star', etc.\n"
        "  * If YES → ALSO use recognizing-shapes skill\n"
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
        "\n"
        "OUTPUT FORMAT:\n"
        "Return a single valid JSON object with these exact keys: 'reasoning', 'final_answer', 'token_usage', 'elapsed_time'\n"
        "Example: {\"reasoning\": \"Step 1: Read master-skill/SKILL.md and found 4 skills. Step 2: Image has color, question asks about shapes. Step 3: Read colored-images and recognizing-shapes skills...\", \"final_answer\": \"...\", \"token_usage\": 123, \"elapsed_time\": 2.34}"
    )

    # System prompt for OpenAI ChatCompletion (no skills)
    chatcompletion_system_prompt = (
        "You are an MS Paint reasoning assistant.\n"
        "Always use chain-of-thought (CoT) reasoning.\n"
        "Explain your reasoning step by step before giving the final answer.\n"
        "If the question asks for an updated image, describe the changes and return the modified image if possible.\n"
        "OUTPUT: Return a single valid JSON object with exactly these keys: 'reasoning', 'final_answer', 'token_usage', 'elapsed_time'. No extra text.\n"
        "Example: {\"reasoning\": \"...\", \"final_answer\": \"...\", \"token_usage\": 123, \"elapsed_time\": 2.34}"
    )

    # Choose agent based on skills
    if skills:
        agent = create_deep_agent(
            model=llm,
            system_prompt=deepagent_system_prompt,
            skills=skills_relative,
            backend=backend,
            debug=False,
        )
        agent_input = {
            "messages": [
                {"role": "user", "content": content}
            ],
            "reasoning": {
                "effort": reasoning_effort,
                "summary": "auto",
            }
        }
        config = {"configurable": {"thread_id": "default"}}
        start_time = time.time()
        response = agent.invoke(agent_input, config=config)
        elapsed_time = time.time() - start_time
        messages = response.get("messages", [])
        answer = messages[-1].content if messages else None

        # --- Output handling for skills mode ---
        import json, re
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
        elapsed_time_measured = float(elapsed_time)
        if isinstance(parsed, dict):
            parsed["token_usage"] = token_usage if token_usage is not None else parsed.get("token_usage")
            parsed["elapsed_time"] = elapsed_time_measured
        else:
            parsed = {
                "reasoning": str(answer),
                "final_answer": str(answer),
                "token_usage": token_usage,
                "elapsed_time": elapsed_time_measured,
            }
        print("\nReasoning:\n", parsed.get("reasoning", "[Missing reasoning]"))
        print("\nFinal Answer:\n", parsed.get("final_answer", "[Missing final_answer]"))
        import json as _json
        print("\nToken Usage:\n", _json.dumps(parsed.get("token_usage", "n/a")))
        print("\nElapsed Time (s):\n", round(parsed.get("elapsed_time", elapsed_time_measured), 3))
        if model_used:
            print("\n[INFO] Model used (from response metadata):", model_used)
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

    else:
        import openai
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        # Compose user message with image and question in OpenAI API format
        user_content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]
        start_time = time.time()
        if model_name in ["gpt-5.1", "gpt-5.2", "gpt-5-pro"]:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": chatcompletion_system_prompt},
                    {"role": "user", "content": user_content}
                ],
                reasoning_effort=reasoning_effort,
                # Note: gpt-5.x models do not support temperature parameter, use default
            )
        else:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": chatcompletion_system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
            )
        elapsed_time = time.time() - start_time
        answer = response.choices[0].message.content if response.choices else None
        messages = [response.choices[0].message] if response.choices else []

        # --- Output handling for no-skills mode ---
        import json, re
        parsed = None
        token_usage = None
        input_tokens = None
        output_tokens = None
        total_tokens = None
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
        # Extract token usage from OpenAI response. This is always available for non-streaming calls.
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            token_usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        elapsed_time_measured = float(elapsed_time)
        if isinstance(parsed, dict):
            parsed["token_usage"] = token_usage if token_usage is not None else parsed.get("token_usage")
            parsed["elapsed_time"] = elapsed_time_measured
        else:
            parsed = {
                "reasoning": str(answer),
                "final_answer": str(answer),
                "token_usage": token_usage,
                "elapsed_time": elapsed_time_measured,
            }
        print("\nReasoning:\n", parsed.get("reasoning", "[Missing reasoning]"))
        print("\nFinal Answer:\n", parsed.get("final_answer", "[Missing final_answer]"))
        import json as _json
        print("\nToken Usage:\n", _json.dumps(parsed.get("token_usage", "n/a")))
        print("\nElapsed Time (s):\n", round(parsed.get("elapsed_time", elapsed_time_measured), 3))
        out_path = os.path.join(os.path.dirname(__file__), f"tiny_test_debug_noskills.txt")
        with open(out_path, "w") as f:
            f.write("--- Tiny Test Eval Result (No Skills) ---\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Blur type: {blur_type}\n")
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