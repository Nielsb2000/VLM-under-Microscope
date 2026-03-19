---
name: spatialgrid-img-context
description: Spatial grid image + domain context with layout rules for interpreting 5x5 animal grids.
---

# Spatial Grid — Image + Domain Context

These example images explain the grid format and how to interpret it.
Call `read_file` on each image path listed before answering.

---

## Example 1 — Understanding the Visual Format

Image: `skills/spatialgrid-img-context/assets/grid497_q0.png`

In a spatial grid image:
- The image shows a **5×5 grid** of cells = 25 cells total.
- Each cell contains exactly **one animal icon** from: cat, dog, elephant, giraffe, rabbit.
- **Rows** are numbered 1–5 from **top to bottom**.
- **Columns** are numbered 1–5 from **left to right**.

**Counting questions** ("How many blocks contain [animal]?"):
Scan every row from left to right, top to bottom. Count every cell whose icon
matches the target animal. Total should be ≤ 25.

---

## Example 2 — Identifying a Position

Image: `skills/spatialgrid-img-context/assets/grid498_q1.png`

**Position questions** ask what animal is at a specific cell, e.g.:
- "top-left corner" = row 1, column 1
- "first row, second column" = row 1, column 2

To locate a cell: find the correct row (counting from the top), then count across
to the correct column. Read the icon in that cell carefully.

---

## Example 3 — Second Column of First Row

Image: `skills/spatialgrid-img-context/assets/grid499_q2.png`

Identifying row 1 col 2:
1. The top row is row 1.
2. Count two icons from the left — the second icon is the answer.
Read the animal icon in that position carefully before selecting your option.

---

Now apply this understanding to answer the actual question.
Give your final answer as: [Option Letter]. [Answer]
