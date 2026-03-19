---
name: mazenav
description: Maze navigation image + domain context with color-coding rules for interpreting and solving maze paths.
---

# Maze Navigation — Image + Domain Context

These example images explain the maze format and how to interpret it.
Call `read_file` on each image path listed before answering.

---

## Example 1 — Understanding the Visual Format

Image: `skills/mazenav/assets/maze497_q0.png`

In a maze image:
- **Black blocks** are walls (impassable)
- **White blocks** are open paths (not necessarily the solution)
- **Green block** is the Start (S)
- **Red block** is the Exit (E)
- **Blue blocks** mark the solution path from S to E

**Counting right turns**: Trace the blue path from S to E step by step, recording
your cardinal direction at each move (UP, DOWN, LEFT, RIGHT).
A **right turn** is a 90° clockwise direction change:
  UP→RIGHT, RIGHT→DOWN, DOWN→LEFT, LEFT→UP

---

## Example 2 — Counting Total Turns

Image: `skills/mazenav/assets/maze498_q1.png`

**Total turns** = every direction change on the blue path, both clockwise and
anticlockwise. Walk the path step by step and increment your counter each time
the direction changes, regardless of which way it turns (left or right).

---

## Example 3 — Directional Relationship of S and E

Image: `skills/mazenav/assets/maze499_q2.png`

**Directional questions** ask about the spatial relationship between S and E,
completely ignoring the path in between. Only compare the pixel positions of the
green block (S) and the red block (E).
- "Directly below with no horizontal displacement" → same column, E has a higher
  row index (lower on screen) than S.
- "Directly above" → same column, E has a lower row index (higher on screen).
- "Directly to the left/right" → same row, different column.

---

Now apply this understanding to answer the actual question.
Give your final answer as: [Option Letter]. [Answer]
