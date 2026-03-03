# Batch evaluation script for MS Paint Reasoning using tiny_test_eval.py

import os
import sys
import argparse
import subprocess
import json

IMG_DIR = "MS_paint_images"
QUESTIONS_DIR = os.path.join(IMG_DIR, "MS paint questions")
RESULTS_DIR = "Results"

# Expose for testing
def get_image_type_folders():
	return IMAGE_TYPE_FOLDERS

# Map image type and blur to subdirectory
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
SYSTEM_PROMPT = (
	"You are a careful visual reasoning assistant. "
	"Always use chain-of-thought (CoT) reasoning. "
	"Explain your reasoning step by step before giving the final answer. "
	"If the question asks for an updated image, describe the changes and return the modified image if possible."
)

# Map blur level to subdirectory for images and results
BLUR_LEVELS = {
	"none": "",  # original images
	"med_blur": "med_blur_images",
	"heavy_blur": "heavy_blur_images"
}


def get_img_dir(image_type, blur_level):
	# Accept both 'no_blur' and 'none' as equivalent for backward compatibility
	lookup_blur = 'none' if blur_level == 'no_blur' else blur_level
	folder = IMAGE_TYPE_FOLDERS.get((image_type, lookup_blur))
	if not folder:
		raise ValueError(f"Unsupported combination: {image_type}, {blur_level}")
	return os.path.join(os.path.dirname(__file__), IMG_DIR, folder)

def get_results_dir(blur_level):
	if blur_level == "none":
		return os.path.join(os.path.dirname(__file__), RESULTS_DIR)
	else:
		return os.path.join(os.path.dirname(__file__), RESULTS_DIR, BLUR_LEVELS[blur_level])

# --- Batch evaluation script for MS Paint Reasoning using tiny_test_eval.py and llm_check_answer.py ---
def main():
	parser = argparse.ArgumentParser(description="Batch evaluate MS Paint Reasoning using tiny_test_eval.py and llm_check_answer.py.")
	parser.add_argument('--blur-levels', nargs='+', default=['no_blur'], choices=['no_blur', 'med_blur', 'heavy_blur', 'none'], help="Blur levels to process.")
	parser.add_argument('--image-types', nargs='+', default=['color'], choices=['color', 'greyscale', 'inverted_greyscale'], help="Image types to process.")
	parser.add_argument('--models', nargs='+', default=["gpt-4o", "gpt-5.1", "gpt-5.2"], choices=["gpt-4o", "gpt-5.1", "gpt-5.2"], help="Models to use.")
	parser.add_argument('--reasoning-effort', default='low', choices=["low",'medium', 'high'], help="Reasoning effort: low, medium, or high.")
	parser.add_argument('--reasoning-summary', default='auto', choices=['auto', 'detailed', 'none'], help="Reasoning summary style.")
	args = parser.parse_args()

	base_dir = os.path.dirname(os.path.abspath(__file__))
	img_dir_root = os.path.join(base_dir, "MS_paint_images")
	questions_dir = os.path.join(img_dir_root, "MS paint questions")
	results_dir_root = os.path.join(base_dir, "Results")
	dashboard_data_dir = os.path.join(results_dir_root, "dashboard_data")
	os.makedirs(dashboard_data_dir, exist_ok=True)

	# Map legacy 'none' to 'no_blur'
	blur_levels = [('no_blur' if b == 'none' else b) for b in args.blur_levels]
	for blur_level in blur_levels:
		for image_type in args.image_types:
			results_dict = {}
			IMAGE_TYPE_FOLDERS = {
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
			folder = IMAGE_TYPE_FOLDERS.get((image_type, blur_level))
			if not folder:
				print(f"[ERROR] Unsupported combination: {image_type}, {blur_level}")
				continue
			img_dir = os.path.join(img_dir_root, folder)
			if not os.path.exists(img_dir):
				print(f"[ERROR] Image directory not found: {img_dir}")
				continue
			img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])
			for img_file in img_files:
				img_name = os.path.splitext(img_file)[0]
				img_index = img_name[3:]  # e.g., img1 -> 1
				q_file = f"Questions{img_index}.txt"
				q_path = os.path.join(questions_dir, q_file)
				if not os.path.exists(q_path):
					print(f"[WARN] No questions file for {img_file}, skipping.")
					continue
				with open(q_path, "r") as f:
					questions = [q.strip() for q in f if q.strip()]
				for q_idx, question in enumerate(questions, 1):
					for model_name in args.models:
						print(f"\n--- Evaluating {img_name} Q{q_idx} with {model_name} (type: {image_type}, blur: {blur_level}, effort: {args.reasoning_effort}, summary: {args.reasoning_summary}) ---")
						tiny_test_path = os.path.join(base_dir, "tiny_test_eval.py")
						cmd = [sys.executable, tiny_test_path, image_type, blur_level, img_index, str(q_idx), model_name, args.reasoning_effort, args.reasoning_summary]
						try:
							result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
							output = result.stdout
							err = result.stderr
							if err:
								print(f"[tiny_test_eval.py stderr] {err}")
						except Exception as e:
							print(f"[ERROR] Exception running tiny_test_eval.py: {e}")
							continue
						blur_dir = blur_level
						if blur_dir == "no_blur":
							blur_dir = "no_blur"
						answer_dir = os.path.join(results_dir_root, f"{image_type}_{blur_dir}_medium", f"img{img_index}", f"q{q_idx}")
						os.makedirs(answer_dir, exist_ok=True)
						answer_file = os.path.join(answer_dir, f"answer_{model_name}_skills.txt")
						with open(answer_file, "w") as out_f:
							out_f.write(output)
						print(f"[INFO] Answer written to {answer_file}")

						# Debug: print answer file contents and all relevant variables
						print(f"[DEBUG] --- Debugging Info ---")
						print(f"[DEBUG] img_file: {img_file}")
						print(f"[DEBUG] img_name: {img_name}")
						print(f"[DEBUG] img_index: {img_index}")
						print(f"[DEBUG] q_file: {q_file}")
						print(f"[DEBUG] q_path: {q_path}")
						print(f"[DEBUG] question: {question}")
						print(f"[DEBUG] answer_file: {answer_file}")
						print(f"[DEBUG] answer_dir: {answer_dir}")
						print(f"[DEBUG] model_name: {model_name}")
						print(f"[DEBUG] blur_level: {blur_level}")
						print(f"[DEBUG] image_type: {image_type}")
						print(f"[DEBUG] tiny_test_eval.py cmd: {[sys.executable, tiny_test_path, image_type, blur_level, img_index, str(q_idx), model_name, args.reasoning_effort, args.reasoning_summary]}")
						try:
							with open(answer_file, "r") as af:
								answer_contents = af.read()
							print(f"[DEBUG] Contents of {answer_file}:\n{answer_contents}")
						except Exception as e:
							print(f"[DEBUG] Could not read answer file: {e}")

						# Parse token usage and elapsed time from tiny_test_eval.py output
						import re, json
						token_usage = None
						elapsed_time = None
						print(f"[DEBUG] Raw output from tiny_test_eval.py:\n{output}")
						m = re.search(r'\{.*\}', output, re.DOTALL)
						parsed = None
						if m:
							try:
								parsed = json.loads(m.group(0))
								print(f"[DEBUG] Parsed JSON from output: {parsed}")
							except Exception as e:
								print(f"[DEBUG] Failed to parse JSON: {e}")
								parsed = None
						if parsed and isinstance(parsed, dict):
							token_usage = parsed.get("token_usage")
							elapsed_time = parsed.get("elapsed_time")
							if token_usage is None:
								if "input_tokens" in parsed or "output_tokens" in parsed:
									token_usage = {
										"input_tokens": parsed.get("input_tokens"),
										"output_tokens": parsed.get("output_tokens")
									}
						if token_usage is None:
							m2 = re.search(r'Token Usage:\s*([\{\[].*[\}\]])', output)
							if m2:
								try:
									token_usage = json.loads(m2.group(1))
								except Exception:
									token_usage = None
						if token_usage is None:
							m_in = re.search(r"input_tokens['\"]?\s*:\s*([0-9]+)", output)
							m_out = re.search(r"output_tokens['\"]?\s*:\s*([0-9]+)", output)
							if m_in or m_out:
								token_usage = {}
								if m_in:
									token_usage["input_tokens"] = int(m_in.group(1))
								if m_out:
									token_usage["output_tokens"] = int(m_out.group(1))
						if elapsed_time is None:
							m3 = re.search(r'Elapsed Time \(s\):\s*([0-9.]+)', output)
							if m3:
								try:
									elapsed_time = float(m3.group(1))
								except Exception:
									elapsed_time = None

						llm_check_path = os.path.join(base_dir, "llm_check_answer.py")
						cmd_check = [sys.executable, llm_check_path, img_index, str(q_idx), model_name, image_type, blur_level]
						print(f"[DEBUG] Running llm_check_answer.py with: {' '.join(cmd_check)}")
						print(f"[DEBUG] llm_check_answer.py args: {cmd_check}")
						try:
							result_check = subprocess.run(cmd_check, capture_output=True, text=True, timeout=60)
							print(f"[DEBUG] llm_check_answer.py STDOUT:\n{result_check.stdout}")
							print(f"[DEBUG] llm_check_answer.py STDERR:\n{result_check.stderr}")
							llm_result = result_check.stdout.strip()
							if llm_result and llm_result[0] in {"0", "1"}:
								llm_result = llm_result[0]
							else:
								llm_result = "0"
						except Exception as e:
							print(f"[ERROR] LLM check failed: {e}")
							llm_result = "0"
						print(f"[DEBUG] Final llm_result: {llm_result}")

						input_tokens = None
						output_tokens = None
						if isinstance(token_usage, dict):
							input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
							output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
						print(f"[DEBUG] Extracted token_usage: {token_usage}")
						print(f"[DEBUG] input_tokens: {input_tokens}, output_tokens: {output_tokens}, elapsed_time: {elapsed_time}")
						results_dict[f"{img_name}_Q{q_idx}_{model_name}"] = (
							int(llm_result),
							(input_tokens, output_tokens),
							elapsed_time
						)
				else:
					print(f"[SUCCESS] All questions processed for {q_path}")
					continue
			else:
				print(f"[SUCCESS] All images processed for {img_dir}")
				continue

		# Write results to file for each blur level, image type, and reasoning effort
		models_str = "-".join(args.models)
		result_file_name = f"llm_results_{image_type}_{blur_level}_{models_str}_{args.reasoning_effort}_skills.json"
		result_file_path = os.path.join(dashboard_data_dir, result_file_name)
		with open(result_file_path, "w") as rf:
			json.dump(results_dict, rf, indent=2)
		print(f"\n[INFO] LLM results written to {result_file_path}")

if __name__ == "__main__":
	main()
