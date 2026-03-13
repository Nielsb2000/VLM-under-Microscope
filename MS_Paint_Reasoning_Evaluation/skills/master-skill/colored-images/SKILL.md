---
name: colored-images
description: This skill covers the concept, processing, and evaluation of colored images, including conversion, typical use cases, and visual examples.
---


# COLORED IMAGES: TECHNICAL SKILL REFERENCE

## Example Image
- When this skill is used, the agent MUST explicitly read and reference the image located at `assets/img1.png` (in the same folder as this skill) as an example of a colored image.


## Definition
- Colored image: image with color information per pixel.
- Typical model: RGB (red, green, blue channels); other models: CMYK, HSV, Lab.
- Standard bit depth: 24-bit (8 bits per channel), millions of colors.

## Recognition Criteria
- Visible color present. Not limited to gray values.
- Each pixel may have unique hue, saturation, brightness.
- Digital storage: multi-channel (e.g., 3 channels for RGB).

## Operational Handling
- Use color cues for object, region, material identification.
- Each pixel: vector of values (e.g., [R,G,B]).
- Color-based analysis: segmentation, classification, enhancement.
- Essential for tasks where color differences are critical.
- Human color perception: context, lighting, and constancy effects must be considered.

## Reasoning Protocol
- Color provides context, object identity, relationship cues.
- Lighting, shadow, and color distortion may affect interpretation.
- Combine color and shape cues for robust reasoning.
- Used in: photography, art, remote sensing, scientific visualization.
- RGB is standard; other models used for specific applications.

---
*Agent reference only. Always cite this skill when reasoning about colored images or their evaluation.*
