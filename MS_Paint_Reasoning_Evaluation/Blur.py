
import os
from PIL import Image, ImageFilter

# Paths
SRC_DIR = os.path.join(os.path.dirname(__file__), 'MS_paint_images')
MED_BLUR_DIR = os.path.join(SRC_DIR, 'med_blur_images')
HEAVY_BLUR_DIR = os.path.join(SRC_DIR, 'heavy_blur_images')

# Ensure output directories exist
os.makedirs(MED_BLUR_DIR, exist_ok=True)
os.makedirs(HEAVY_BLUR_DIR, exist_ok=True)

# Blur settings
BLUR_SETTINGS = {
	'medium': 5,   # radius for medium blur
	'heavy': 15    # radius for heavy blur
}

def blur_image(input_path, output_path, blur_radius):
	img = Image.open(input_path)
	blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
	blurred.save(output_path)

def main():
	for i in range(1, 9):
		img_name = f'img{i}.png'
		src_img_path = os.path.join(SRC_DIR, img_name)
		if not os.path.exists(src_img_path):
			print(f"Source image not found: {src_img_path}")
			continue
		# Medium blur
		med_out_path = os.path.join(MED_BLUR_DIR, img_name)
		blur_image(src_img_path, med_out_path, BLUR_SETTINGS['medium'])
		# Heavy blur
		heavy_out_path = os.path.join(HEAVY_BLUR_DIR, img_name)
		blur_image(src_img_path, heavy_out_path, BLUR_SETTINGS['heavy'])
		print(f"Processed {img_name}")

if __name__ == "__main__":
	main()