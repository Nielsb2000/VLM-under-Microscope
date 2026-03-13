---
name: inverted-grayscale-images
description: This skill covers the concept, processing, and evaluation of inverted grayscale images, including conversion, typical use cases, and visual examples.
---


# INVERTED GRAYSCALE IMAGES: TECHNICAL SKILL REFERENCE

## Example Image
- When this skill is used, the agent MUST explicitly read and reference the image located at `assets/img1.png` (in the same folder as this skill) as an example of an inverted grayscale image.

## Definition
- Inverted grayscale image: grayscale image with pixel intensity values reversed.
- Black (low intensity) becomes white (high intensity); white becomes black; all intermediate grays are flipped.
- Equivalent to a photographic negative in monochrome.

## Recognition Criteria
- No color present. Only gray values.
- Bright regions in original become dark; dark regions become bright.
- Visual appearance: negative of standard grayscale.
- Features or spatial relationships may be more visible or less visible compared to original.

## Operational Handling
- All reasoning about brightness must be inverted: light <-> dark.
- Use for: robustness testing, feature highlighting, alternative visualization.
- Inversion is a deterministic, pixel-wise operation: output_pixel = max_intensity - input_pixel.
- Perceptual effect: strong; may alter object/background salience.

## Reasoning Protocol
- Always check: meaning of light/dark is reversed.
- Foreground/background roles may swap.
- Feature prominence may increase or decrease after inversion.
- Use cases: scientific imaging, photography, digital art, model evaluation.

---
*Agent reference only. Always cite this skill when reasoning about inverted grayscale images or their evaluation.*
