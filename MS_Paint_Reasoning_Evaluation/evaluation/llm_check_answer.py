# Simple LLM-based answer checker for MS Paint Reasoning Evaluation

# Usage: python llm_check_answer.py <img_index> <q_index> <model_name> <image_type> <blur_type>
# Output: 1 (if correct), 0 (otherwise)


import sys
import openai
import os

def debug(msg):
    print(msg, file=sys.stderr)


# Usage: python llm_check_answer.py <img_index> <q_index> <model_name> <image_type> <blur_type>
# - <img_index>: integer 1-8 (corresponds to img1.png, Questions1.txt, Answers1.txt, etc)
# - <q_index>: integer (1-based, which question in QuestionsX.txt)
# - <model_name>: gpt-4o, gpt-5.1, or gpt-5.2
# - <image_type>: color, greyscale, inverted_greyscale
# - <blur_type>: no_blur, med_blur, heavy_blur


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    debug("[ERROR] No OPENAI_API_KEY found.")
    print("0")
    sys.exit(0)

# Read GT answers from file

     # ...existing code...

     # Example usage: python llm_check_answer.py <model_answer> <gt_answers_file> <openai_api_key> [model=gpt-4-turbo] [reasoning=high]
     # Output: 1 if the model answer is deemed correct by the LLM, 0 otherwise.


if len(sys.argv) != 6:
    debug("Usage: python llm_check_answer.py <img_index> <q_index> <model_name> <image_type> <blur_type>")
    print("0")
    sys.exit(0)

img_index = int(sys.argv[1])
q_index = int(sys.argv[2])
model_name = sys.argv[3]
image_type = sys.argv[4]
blur_type = sys.argv[5]


# Compose file paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
questions_path = os.path.join(base_dir, "MS_paint_images", "MS paint questions", f"Questions{img_index}.txt")
gt_path = os.path.join(base_dir, "MS_paint_images", "MS paint answers", f"Answers{img_index}.txt")

# Find answer file
if blur_type == "no_blur":
    blur_dir = "no_blur"
else:
    blur_dir = blur_type
results_dir = os.path.join(base_dir, "Results", f"{image_type}_{blur_dir}_medium", f"img{img_index}", f"q{q_index}")
import glob
# Prefer _skills.txt file if present, else fallback to _noskills.txt, else default
skills_file = os.path.join(results_dir, f"answer_{model_name}_skills.txt")
noskills_file = os.path.join(results_dir, f"answer_{model_name}_noskills.txt")
default_file = os.path.join(results_dir, f"answer_{model_name}.txt")
if os.path.exists(skills_file):
    answer_file = skills_file
elif os.path.exists(noskills_file):
    answer_file = noskills_file
elif os.path.exists(default_file):
    answer_file = default_file
else:
    # Fallback: try any matching file
    answer_file_pattern = os.path.join(results_dir, f"answer_{model_name}_*.txt")
    answer_files = glob.glob(answer_file_pattern)
    if answer_files:
        answer_file = answer_files[0]
    else:
        answer_file = default_file

debug(f"[DEBUG] Questions file: {questions_path}")
debug(f"[DEBUG] GT answers file: {gt_path}")
debug(f"[DEBUG] Model answer file: {answer_file}")


# Read question
try:
    with open(questions_path, "r") as f:
        questions = [line.strip() for line in f if line.strip()]
    question = questions[q_index - 1]
    debug(f"[DEBUG] Loaded question: {question}")
except Exception:
    debug("[ERROR] Failed to load question.")
    print("0")
    sys.exit(0)




# Read model answer: extract after 'Final Answer' if present, else use whole raw output (excluding log/info lines)
try:
    with open(answer_file, "r") as f:
        answer_text = f.read()
    # Remove log/info/debug lines
    answer_lines = [line for line in answer_text.splitlines() if line.strip() and not (
        line.startswith("[INFO]") or line.startswith("[DEBUG]") or line.startswith("[ERROR]") or line.startswith("Answer saved to:")
    )]
    filtered_text = "\n".join(answer_lines)
    # Try to extract after 'Final Answer'
    import re
    match = re.search(r"Final Answer\s*[:\-]?\s*(.*)", filtered_text, re.IGNORECASE | re.DOTALL)
    if match:
        model_answer = match.group(1).strip()
        # If answer is empty, try to get the next non-empty line
        if not model_answer:
            lines = filtered_text.splitlines()
            for i, line in enumerate(lines):
                if re.match(r"Final Answer", line, re.IGNORECASE):
                    for next_line in lines[i+1:]:
                        if next_line.strip():
                            model_answer = next_line.strip()
                            break
                    break
        # Remove trailing token usage or elapsed time info if present
        for marker in ["Token Usage:", "Elapsed Time (s):", "[INFO]", "[DEBUG]", "[ERROR]", "Answer saved to:"]:
            idx = model_answer.find(marker)
            if idx != -1:
                model_answer = model_answer[:idx].strip()
        debug(f"[DEBUG] Extracted after 'Final Answer': {model_answer}")
        if not model_answer:
            model_answer = filtered_text.strip()
            debug(f"[DEBUG] Fallback to full filtered text: {model_answer}")
    else:
        model_answer = filtered_text.strip()
        debug(f"[DEBUG] No 'Final Answer' found, using full text.")
    if not model_answer:
        debug("[ERROR] No model answer found.")
        print("0")
        sys.exit(0)
    debug(f"[DEBUG] Loaded model answer: {model_answer}")
except Exception:
    debug("[ERROR] Failed to load model answer.")
    print("0")
    sys.exit(0)



# Read GT answers from file (only use answer for the specific question index)
try:
    with open(gt_path, 'r') as f:
        gt_lines = [line.strip() for line in f if line.strip()]
    # Only use the answer line for the current question index
    if q_index <= len(gt_lines):
        gt_line = gt_lines[q_index - 1]
        debug(f"[DEBUG] Loaded GT answer line: {gt_line}")
    else:
        debug("[ERROR] GT answer for question index not found.")
        print("0")
        sys.exit(0)
except Exception:
    debug("[ERROR] Failed to load GT answers.")
    print("0")
    sys.exit(0)

# Flatten GT answers: allow 'or' separated answers on a line
gt_answers = []
if ' or ' in gt_line:
    gt_answers.extend([a.strip() for a in gt_line.split(' or ') if a.strip()])
else:
    gt_answers.append(gt_line)

system_prompt = (
    "You are a strict answer checker for MS Paint Reasoning Evaluation. "
    "Given a question, a model's answer, and a ground truth answer, output 1 if the model's answer is correct (matches the ground truth in meaning), and 0 otherwise. "
    "Only output a single digit: 1 for correct, 0 for incorrect. Do not explain. Be strict: only output 1 if the answer is clearly correct."
)

# Check each GT answer with the LLM, output 1 if any match

for gt_answer in gt_answers:
    user_prompt = f"""
Question: {question}
Model Answer: {model_answer}
Ground Truth Answer: {gt_answer}
"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
    )
    result = response.choices[0].message.content.strip()
    debug(f"[DEBUG] LLM raw result: {result}")
    if result == "1":
        print("1")
        sys.exit(0)

# If none matched
print("0")
