import os
from PIL import Image, ImageFilter, ImageOps

# Directories
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, 'MS_paint_images', 'original_images')

# Output folders for all types
OUT_DIRS = {
    'color': os.path.join(BASE_DIR, 'MS_paint_images', 'original_images'),
    'greyscale': os.path.join(BASE_DIR, 'MS_paint_images', 'greyscale_images'),
    'inverted_greyscale': os.path.join(BASE_DIR, 'MS_paint_images', 'inverted_greyscale_images'),
    'original_med_blur': os.path.join(BASE_DIR, 'MS_paint_images', 'original_med_blur_images'),
    'med_blur_greyscale': os.path.join(BASE_DIR, 'MS_paint_images', 'med_blur_greyscale_images'),
    'med_blur_inverted_greyscale': os.path.join(BASE_DIR, 'MS_paint_images', 'med_blur_inverted_greyscale_images'),
    'original_heavy_blur': os.path.join(BASE_DIR, 'MS_paint_images', 'original_heavy_blur_images'),
    'heavy_blur_greyscale': os.path.join(BASE_DIR, 'MS_paint_images', 'heavy_blur_greyscale_images'),
    'heavy_blur_inverted_greyscale': os.path.join(BASE_DIR, 'MS_paint_images', 'heavy_blur_inverted_greyscale_images'),
}

BLUR_SETTINGS = {
    'med_blur': 5,
    'heavy_blur': 15
}

for d in OUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

def process_and_save(img, img_name, out_dir, process_fn=None):
    out_path = os.path.join(out_dir, img_name)
    if process_fn:
        img = process_fn(img)
    img.save(out_path)

def main():
    for img_name in sorted(os.listdir(SRC_DIR)):
        if not img_name.lower().endswith('.png'):
            continue
        src_img_path = os.path.join(SRC_DIR, img_name)
        img = Image.open(src_img_path)
        # Save color original
        process_and_save(img, img_name, OUT_DIRS['color'])
        # Save greyscale
        process_and_save(img, img_name, OUT_DIRS['greyscale'], lambda im: im.convert('L'))
        # Save inverted greyscale
        process_and_save(img, img_name, OUT_DIRS['inverted_greyscale'], lambda im: ImageOps.invert(im.convert('L')))
        # Medium blur
        med_blur = img.filter(ImageFilter.GaussianBlur(radius=BLUR_SETTINGS['med_blur']))
        process_and_save(med_blur, img_name, OUT_DIRS['original_med_blur'])
        process_and_save(med_blur, img_name, OUT_DIRS['med_blur_greyscale'], lambda im: im.convert('L'))
        process_and_save(med_blur, img_name, OUT_DIRS['med_blur_inverted_greyscale'], lambda im: ImageOps.invert(im.convert('L')))
        # Heavy blur
        heavy_blur = img.filter(ImageFilter.GaussianBlur(radius=BLUR_SETTINGS['heavy_blur']))
        process_and_save(heavy_blur, img_name, OUT_DIRS['original_heavy_blur'])
        process_and_save(heavy_blur, img_name, OUT_DIRS['heavy_blur_greyscale'], lambda im: im.convert('L'))
        process_and_save(heavy_blur, img_name, OUT_DIRS['heavy_blur_inverted_greyscale'], lambda im: ImageOps.invert(im.convert('L')))
        print(f"Processed {img_name}")

if __name__ == "__main__":
    main()
