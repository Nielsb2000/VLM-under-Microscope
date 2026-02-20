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

def get_img_dir(blur_level):
	if blur_level == "none":
		return os.path.join(os.path.dirname(__file__), IMG_DIR)
	else:
		return os.path.join(os.path.dirname(__file__), IMG_DIR, BLUR_LEVELS[blur_level])

def get_results_dir(blur_level):
	if blur_level == "none":
		return os.path.join(os.path.dirname(__file__), RESULTS_DIR)
	else:
		return os.path.join(os.path.dirname(__file__), RESULTS_DIR, blur_level)




def main():
	parser = argparse.ArgumentParser(description="Evaluate MS Paint Reasoning on different blur levels and reasoning modes.")
	parser.add_argument('--blur-levels', nargs='+', default=['heavy_blur'], choices=['none', 'med_blur', 'heavy_blur'],
						help="Which blur levels to process. Choices: none, med_blur, heavy_blur. Default: heavy_blur.")
	parser.add_argument('--models', nargs='+', default=["gpt-4o", "gpt-5.1", "gpt-5.2"],
						choices=["gpt-4o", "gpt-5.1", "gpt-5.2"],
						help="Which models to use. Default: all.")
	parser.add_argument('--reasoning-mode', default='medium', choices=['low', 'medium', 'high'],
						help="Reasoning mode to use. Choices: low, medium, high. Default: medium.")
	args = parser.parse_args()

	model_names = args.models
	reasoning_mode = args.reasoning_mode
	# Safety: Exclude gpt-4o for high reasoning
	if reasoning_mode == 'high':
		model_names = [m for m in model_names if m in ('gpt-5.1', 'gpt-5.2')]
		if not model_names:
			print("No supported models for high reasoning. Only gpt-5.1 and gpt-5.2 are allowed.")
			return
	questions_dir = os.path.join(os.path.dirname(__file__), QUESTIONS_DIR)

	for blur_level in args.blur_levels:
		img_dir = get_img_dir(blur_level)
		# Results directory now includes reasoning mode
		results_dir = os.path.join(os.path.dirname(__file__), RESULTS_DIR, f"{blur_level}_{reasoning_mode}")
		os.makedirs(results_dir, exist_ok=True)
		img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])

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
			for idx, q in enumerate(questions, 1):
				q_dir = os.path.join(results_dir, img_name, f"q{idx}")
				os.makedirs(q_dir, exist_ok=True)
				import time
				for model_name in model_names:
					print(f"\n--- Evaluating {img_name} Q{idx} with {model_name} (blur: {blur_level}, reasoning: {reasoning_mode}) ---")
					model = GPT4VisionMSPaint(model_name=model_name, reasoning_mode=reasoning_mode)
					# Add reasoning mode to prompt as per OpenAI docs
					prompt = f"{SYSTEM_PROMPT}\n\nReasoning mode: {reasoning_mode}\nQuestion: {q}"
					start_time = time.time()
					try:
						_, answer, token_usage = model.generate(prompt, img, reasoning_mode=reasoning_mode)
					except Exception as e:
						answer = f"[ERROR] {e}"
						token_usage = None
					elapsed = time.time() - start_time
					answer += "\n\n---\n"
					if token_usage:
						answer += f"Token usage: {token_usage}\n"
					else:
						answer += "Token usage: unavailable\n"
					answer += f"Elapsed time: {elapsed:.2f} seconds\n"
					out_path = os.path.join(q_dir, f"answer_{model_name}.txt")
					with open(out_path, "w") as out_f:
						out_f.write(answer)
					print(f"Answered {img_name} Q{idx} ({model_name}, blur: {blur_level}, reasoning: {reasoning_mode}): {out_path}")

if __name__ == "__main__":
	main()
