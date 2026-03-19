---
name: master-skill-img-context
description: Routes spatial reasoning questions to image + domain context variant skills for mazenav, spatialgrid, and spatialmap.
---

# Spatial Reasoning Master Skill — Image + Context Variant

## Purpose
Route spatial reasoning questions to the correct task-specific skill based on the question content.

## Task Routing Table

Examine the question text and identify the task using these keywords:

| Keywords in question text | Task | Skill file to read next |
|---|---|---|
| "Maze", "Blue", "right turn", "left turn", "S to E", "path", "#######", "X marks" | Maze Navigation | `skills/mazenav-img-context/SKILL.md` |
| "5x5 grid", "animal", "blocks contain", "cat", "dog", "elephant", "giraffe", "rabbit" | Spatial Grid | `skills/spatialgrid-img-context/SKILL.md` |
| "map", "direction", "Northeast", "Northwest", "Southwest", "Southeast", "relative to" | Spatial Map | `skills/spatialmap-img-context/SKILL.md` |

## Workflow

**STEP 1**: You have read this master skill (done).

**STEP 2**: Identify the task type from the question text using the table above.

**STEP 3**: Call `read_file` on the matching task skill file.

**STEP 4**: Follow the instructions in that skill file to solve the question.

## Important Notes
- Always read the task skill before answering. The task skill contains the solving strategy.
- The image input is provided inline in the message — do NOT call read_file for images unless
  explicitly listed as a path inside a skill file.
- Use read_file only for skill files and for example image paths listed within skill files.
