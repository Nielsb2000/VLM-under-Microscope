---
name: grayscale-images
description: This skill covers the concept, processing, and evaluation of grayscale images, including conversion from color, typical use cases, and visual examples.
---


# GRAYSCALE IMAGES: TECHNICAL SKILL REFERENCE

## Example Image
- When this skill is used, the agent MUST explicitly read and reference the image located at `assets/img3.png` (in the same folder as this skill) as an example of a grayscale image.

## Example Image Description
- The example image `assets/img3.png` is a grayscale image that contains various shapes.
- The background is a uniform dark gray.
- in the top left there is a medium sized gray circle, in the top right slightly higher on the y-axis than the medium sized gray circle there is a similar medium sized gray circle.
- In the top center of the image, there is a gray triangle pointed upwards.
- In the center of the image slightly higher than the middle point, is a white diamond shape.
- In the bottom right of the image, there is a medium sized white hexagon shape.
- in the bottom left of the image there is a dark gray (slightly lighter than the background) pentagon shape.
- Starting from the left center of the image and going to the right center of the image, there is a wavy horizontal line which is black and looks like a '~'.

## Definition
- Grayscale image: image with only gray values, no color.
- Pixel value: single intensity, range black (min) to white (max).
- Typical bit depth: 8-bit (256 levels), higher for technical/scientific use.

## Recognition Criteria
- No visible color. Only shades of gray.
- All details as brightness variation.
- Digital storage: single-channel data.

## Operational Handling
- Ignore color; analyze intensity only.
- Use for: structure, edge, texture, pattern analysis.
- Reduces computational complexity versus color.
- Human vision: most sensitive to mid-gray differences; contrast/gamma affect perception.

## Reasoning Protocol
- Structure and form are primary cues.
- Conversion from color is not unique; method affects feature emphasis.
- Used in: photography, medical imaging, remote sensing, technical fields.
- Some images may appear gray but are multi-channel; verify true grayscale.
- Grayscale can originate from any electromagnetic spectrum region.

---
*Agent reference only. Always cite this skill when reasoning about grayscale images or their evaluation.*
