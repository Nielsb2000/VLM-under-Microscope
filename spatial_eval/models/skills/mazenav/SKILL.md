---
name: mazenav
description: Counts right turns, left turns, or total turns along the solution path of a maze from S to E.
---

# Maze Navigation Skill

## Task Description
Count the number of right turns, left turns, or total turns along the solution path of a maze from Start (S) to End (E).

---

## Input Modalities

### VQA (Image Only)
The image shows a maze with colored blocks:
- **Black blocks**: walls (impassable)
- **White blocks**: open navigable paths
- **Green block**: Start point (S)
- **Red block**: End point (E)
- **Blue blocks**: the solution path from S to E that you must trace

Trace the blue blocks from green (S) to red (E) and record each move direction.

### VTQA (Image + ASCII Text)
The question text contains an ASCII representation of the maze, formatted like:

```
#######
#E# # #
#X# # #
#X#  S#
#X###X#
#XXXXX#
#######
```

Symbol meanings:
- `#` = wall
- ` ` (space) = open path (not the solution)
- `X` = the solution path (equivalent to blue blocks)
- `S` = start point
- `E` = end point

**In VTQA mode, use the ASCII text to trace the path — it is more precise than the image for counting.**

---

## Worked Examples (VQA Mode — Read These Before Answering)

MANDATORY for VQA tasks: Call `read_file` on each example image below and study how the path is traced and turns are counted.

**Example 1**: `skills/mazenav/assets/example1.png`
- Question: "How many right turns are there in the path from S to E?"
- Trace: Follow the blue blocks from the green (S) cell to the red (E) cell, recording each direction.
- Answer: **C. 2** — there are exactly 2 clockwise 90° direction changes along the blue path.

**Example 2**: `skills/mazenav/assets/example2.png`
- Question: "How many right turns are there in the path from S to E?"
- Trace: Same approach — follow blue blocks, note where direction changes by 90° clockwise.
- Answer: **B. 3** — there are exactly 3 right turns along the blue path.

Use these examples to calibrate your visual tracing before solving the actual question.

---

## Solving Strategy

### Step 1: Extract the path sequence
- Start at `S` (or green block). Move to each adjacent `X` (or blue block). End at `E` (or red block).
- Record each step as a cardinal direction: **UP**, **DOWN**, **LEFT**, **RIGHT**.
- In ASCII: row decreases = UP, row increases = DOWN, column increases = RIGHT, column decreases = LEFT.

### Step 2: Identify turns
Compare each consecutive pair of directions:
- **Right turn (clockwise 90°)**:  UP→RIGHT, RIGHT→DOWN, DOWN→LEFT, LEFT→UP
- **Left turn (counter-clockwise 90°)**: UP→LEFT, LEFT→DOWN, DOWN→RIGHT, RIGHT→UP
- **Straight**: same direction as previous — NOT a turn.

### Step 3: Count based on question type
- "How many right turns?" → count only clockwise 90° changes
- "How many left turns?" → count only counter-clockwise changes
- "How many total turns?" → count all direction changes (right + left)

### Worked example
Path sequence: RIGHT, RIGHT, DOWN, DOWN, LEFT → turns at positions 3 (RIGHT→DOWN = right turn) and 5 (DOWN→LEFT = right turn) → **2 right turns**.

---

## Output Format
Respond with the matching option letter followed by the count.

**Example**: `C. 2`

Always double-check by re-tracing the path before giving your final answer.
