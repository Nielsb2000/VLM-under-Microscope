---
name: spatialgrid
description: Answer questions about a 5x5 grid of animal images: count occurrences or identify the animal at a specific position.
---

# Spatial Grid Skill

## Task Description
Answer questions about a 5×5 grid containing animal images — either counting how many cells contain a specific animal, or identifying which animal is at a specific position.

---

## Grid Layout
- The grid is **5 rows × 5 columns** = 25 cells total.
- Each cell contains exactly one animal from: **cat, dog, elephant, giraffe, rabbit**.
- Rows are numbered 1–5 from **top to bottom**. Columns are numbered 1–5 from **left to right**.

---

## Input Modalities

### VQA (Image Only)
The image shows a 5×5 grid of illustrated animal icons. Read each cell carefully from top-left to bottom-right, row by row. Count or locate the target animal from the visual icons.

### VTQA (Image + Text Table)
The question text contains a pipe-delimited textual representation of the grid, such as:

```
elephant | rabbit   | rabbit   | dog     | giraffe
cat      | rabbit   | elephant | dog     | cat
elephant | elephant | giraffe  | giraffe | rabbit
rabbit   | elephant | elephant | rabbit  | rabbit
elephant | cat      | elephant | cat     | cat
```

**In VTQA mode, use the text table — it is unambiguous and eliminates visual recognition errors.**

---

## Worked Examples (VQA Mode — Read These Before Answering)

MANDATORY for VQA tasks: Call `read_file` on each example image below and study how to read the 5×5 grid and count animals.

**Example 1**: `skills/spatialgrid/assets/example1.png`
- Question: "How many blocks contain dog?"
- Method: Scan each row of the 5×5 grid from top-left to bottom-right, counting dog icons.
- Answer: **C. 2** — exactly 2 cells in this grid contain a dog.

**Example 2**: `skills/spatialgrid/assets/example2.png`
- Question: "How many blocks contain rabbit?"
- Method: Same systematic row-by-row scan, counting rabbit icons.
- Answer: **C. 3** — exactly 3 cells in this grid contain a rabbit.

Use these examples to calibrate your icon recognition and counting before solving the actual question.

---

## Question Types and Solving Strategies

### Count Questions: "How many blocks contain [animal]?"
1. In VTQA: scan every cell in the text table and count occurrences of the named animal.
2. In VQA: systematically scan the image row by row, counting each matching icon.
3. Verify your count before answering (total should be ≤ 25).

### Position Questions: "What animal is in row R, column C?" or "What is in block (R, C)?"
1. In VTQA: go to row R (1-indexed from top), column C (1-indexed from left) in the text table.
2. In VQA: locate the cell at row R, column C in the image.

---

## Output Format
Respond with the matching option letter followed by the count or animal name.

**Examples**:
- `C. 2` (for count questions)
- `B. elephant` (for identification questions)
