# Simple evaluation script for MS Paint Reasoning

import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
from gpt4_model_MS_Paint import GPT4VisionMSPaint

IMG_DIR = "MS_paint_images"
QUESTIONS_DIR = os.path.join(IMG_DIR, "MS paint questions")
RESULTS_DIR = "Results"

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
		return os.path.join(os.path.dirname(__file__), RESULTS_DIR, "none")
	else:
		return os.path.join(os.path.dirname(__file__), RESULTS_DIR, blur_level)





def main():
	parser = argparse.ArgumentParser(description="Evaluate MS Paint Reasoning on different blur levels, image types, and reasoning modes.")
	parser.add_argument('--blur-levels', nargs='+', default=['heavy_blur'], choices=['no_blur', 'med_blur', 'heavy_blur', 'none'],
				   help="Which blur levels to process. Choices: no_blur, med_blur, heavy_blur. Default: heavy_blur. 'none' is accepted as an alias for 'no_blur'.")
	parser.add_argument('--image-types', nargs='+', default=['color'], choices=['color', 'greyscale', 'inverted_greyscale'],
					help="Which image types to process. Choices: color, greyscale, inverted_greyscale. Default: color.")
	parser.add_argument('--models', nargs='+', default=["gpt-4o", "gpt-5.1", "gpt-5.2"],
					choices=["gpt-4o", "gpt-5.1", "gpt-5.2"],
					help="Which models to use. Default: all.")
	parser.add_argument('--reasoning-mode', default='none', choices=['none', 'low', 'medium', 'high'],
				   help="Reasoning mode to use. Choices: none, low, medium, high. Default: none.")
	parser.add_argument('--img-index', type=int, default=None, help="If set, only run for this image index (e.g., 1 for img1.png)")
	parser.add_argument('--q-index', type=int, default=None, help="If set, only run for this question index (1-based)")
	args = parser.parse_args()

	model_names = args.models
	reasoning_mode = args.reasoning_mode
	# Exclude gpt-4o for any reasoning mode except 'none'
	if reasoning_mode != 'none':
		if 'gpt-4o' in model_names:
			print(f"[WARNING] gpt-4o does not support reasoning mode '{reasoning_mode}'. It will be skipped.")
			model_names = [m for m in model_names if m != 'gpt-4o']
		if not model_names:
			print(f"No supported models for reasoning mode '{reasoning_mode}'. Only gpt-5.1 and gpt-5.2 are allowed.")
			return
	questions_dir = os.path.join(os.path.dirname(__file__), QUESTIONS_DIR)


	# Map legacy 'none' to 'no_blur' for all downstream logic
	blur_levels = [('no_blur' if b == 'none' else b) for b in args.blur_levels]
	for blur_level in blur_levels:
		for image_type in args.image_types:
			img_dir = get_img_dir(image_type, blur_level)
			# Results directory now includes image type, blur, and reasoning mode
			results_dir = os.path.join(os.path.dirname(__file__), RESULTS_DIR, f"{image_type}_{blur_level}_{reasoning_mode}")
			os.makedirs(results_dir, exist_ok=True)
			img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])

			# If --img-index is set, filter to only that image
			if args.img_index is not None:
				img_files = [f"img{args.img_index}.png"] if f"img{args.img_index}.png" in img_files else []

			for img_file in img_files:
				img_name = os.path.splitext(img_file)[0]
				# Find corresponding questions file
				q_file = f"Questions{img_name[3:]}.txt"  # img1.png -> Questions1.txt
				q_path = os.path.join(questions_dir, q_file)
				img_path = os.path.join(img_dir, img_file)
				if not os.path.exists(q_path):
					print(f"No questions file for {img_file}, skipping.")
					continue
				# Load image and questions
				img = Image.open(img_path)
				with open(q_path, "r") as f:
					questions = [q.strip() for q in f if q.strip()]

				# If --q-index is set, filter to only that question
				question_indices = [args.q_index] if args.q_index is not None and 1 <= args.q_index <= len(questions) else list(range(1, len(questions)+1))

				for idx in question_indices:
					q = questions[idx-1]
					q_dir = os.path.join(results_dir, img_name, f"q{idx}")
					os.makedirs(q_dir, exist_ok=True)
					import subprocess
					for model_name in model_names:
						print(f"\n--- Evaluating {img_name} Q{idx} with {model_name} (type: {image_type}, blur: {blur_level}) ---")
						# Call tiny_test_eval.py as subprocess
						tiny_test_path = os.path.join(os.path.dirname(__file__), "tiny_test_eval.py")
						# Use sys.executable or 'python3' if sys.executable is not set
						python_exec = sys.executable if sys.executable else 'python3'
						args_sub = [
							python_exec, tiny_test_path,
							image_type, blur_level, img_name[3:], str(idx), model_name
						]
						try:
							result = subprocess.run(args_sub, capture_output=True, text=True, check=True)
						except subprocess.CalledProcessError as e:
							print(f"[ERROR] tiny_test_eval.py failed: {e.stderr}")
							raise
						# Read answer from tiny_test_debug.txt
						debug_path = os.path.join(os.path.dirname(__file__), "tiny_test_debug.txt")
						if not os.path.exists(debug_path):
							print(f"[ERROR] tiny_test_debug.txt not found after running tiny_test_eval.py.")
							raise RuntimeError("tiny_test_debug.txt missing.")
						with open(debug_path, "r") as f:
							debug_content = f.read()
						# Extract Reasoning, Final Answer, Token Usage, and Elapsed Time
						import re
						reasoning_match = re.search(r"Reasoning:\n(.*?)\n\nFinal Answer:", debug_content, re.DOTALL)
						final_answer_match = re.search(r"Final Answer:\n(.*?)\n", debug_content, re.DOTALL)
						token_usage_match = re.search(r"Token Usage:\n(.*?)\n", debug_content, re.DOTALL)
						elapsed_time_match = re.search(r"Elapsed Time \(s\):\n(.*?)\n", debug_content, re.DOTALL)
						reasoning = reasoning_match.group(1).strip() if reasoning_match else "[Missing reasoning]"
						final_answer = final_answer_match.group(1).strip() if final_answer_match else "[Missing final_answer]"
						token_usage = token_usage_match.group(1).strip() if token_usage_match else "[Missing token_usage]"
						elapsed_time = elapsed_time_match.group(1).strip() if elapsed_time_match else "[Missing elapsed_time]"
						# Save output
						out_path = os.path.join(q_dir, f"answer_{model_name}.txt")
						with open(out_path, "w") as out_f:
							out_f.write(f"Reasoning:\n{reasoning}\n\nFinal Answer:\n{final_answer}\n\nToken Usage:\n{token_usage}\n\nElapsed Time (s):\n{elapsed_time}\n")
						print(f"Answered {img_name} Q{idx} ({model_name}, type: {image_type}, blur: {blur_level}): {out_path}")

if __name__ == "__main__":
	main()
