---
name: master-skill
description: Central routing guide for all MS Paint image reasoning skills. Start here to determine which specialized skill(s) to use for any image evaluation task.
---

# MASTER SKILL ROUTING GUIDE (MS PAINT REASONING EVALUATION)

## Purpose
Central routing authority for all MS Paint image reasoning skills. Use this file to select the correct specialized skill for any image evaluation task.


## Skill Routing Table

| Task / Image Domain         | Skill File Link                                         | When to Use |
|-----------------------------|--------------------------------------------------------|-------------|
| Grayscale images            | [grayscale-images/SKILL.md](./grayscale-images/SKILL.md) | For any image containing only shades of gray, no color |
| Colored images              | [colored-images/SKILL.md](./colored-images/SKILL.md)     | For any image containing visible color |
| Inverted grayscale images   | [inverted-grayscale-images/SKILL.md](./inverted-grayscale-images/SKILL.md) | For grayscale images with brightness values reversed (negative) |
| Recognizing shapes          | [recognizing-shapes/SKILL.md](./recognizing-shapes/SKILL.md) | Use when the question requires identifying, describing, or reasoning about specific shapes in an image (e.g., circle, square, star, lightning, etc.) |


## Routing Protocol

Step 1: Identify the type of image or the nature of the question.
- If the question is about the overall image domain (color, grayscale, inverted):
	- Use the appropriate skill for that domain as listed in the table above.
- If the question specifically asks about the presence, identification, or description of shapes (e.g., "What shape is this?", "Describe the shapes present", "Is there a star or lightning bolt?"):
	- Use the [recognizing-shapes/SKILL.md](./recognizing-shapes/SKILL.md) skill.

Step 2: Locate the relevant skill in the routing table above.

Step 3: Follow the link to the specialized skill file.
- Read the skill file for domain- or task-specific guidance, evaluation criteria, and reasoning protocols.

Step 4: If the skill file references subskills, tools, or example images, follow those links recursively.

Step 5: When reasoning or answering, always cite:
- This master skill file
- The specialized skill file(s) used


## Available Skills
- [grayscale-images/SKILL.md](./grayscale-images/SKILL.md): Grayscale image reasoning and evaluation
- [colored-images/SKILL.md](./colored-images/SKILL.md): Colored image reasoning and evaluation
- [inverted-grayscale-images/SKILL.md](./inverted-grayscale-images/SKILL.md): Inverted grayscale image reasoning and evaluation
- [recognizing-shapes/SKILL.md](./recognizing-shapes/SKILL.md): Shape recognition and description (use for any question about identifying or describing shapes)

---
*Always start with this master skill file. It governs all routing and skill usage for MS Paint image evaluation.*
