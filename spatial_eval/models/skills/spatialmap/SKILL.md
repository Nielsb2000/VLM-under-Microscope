# Spatial Map Skill

## Task Description
Determine the cardinal or intercardinal direction of one named object relative to another on a 2D map. Questions ask: "In which direction is **X** relative to **Y**?"

---

## Direction Reference

```
         Northwest  |  North  |  Northeast
         -----------+---------+-----------
            West    | (center)|    East
         -----------+---------+-----------
         Southwest  |  South  |  Southeast
```

Rules for determining direction of Object A relative to Object B (i.e., where is A, standing at B?):
- A is to the **right** (higher x / further right) of B → A is **East** of B
- A is to the **left** (lower x / further left) of B → A is **West** of B
- A is **above** (higher up on screen / lower y-value) B → A is **North** of B
- A is **below** (lower on screen / higher y-value) B → A is **South** of B
- Diagonals: combine both axes — upper-right → **Northeast**, upper-left → **Northwest**, lower-right → **Southeast**, lower-left → **Southwest**

---

## Worked Examples (VQA Mode — Read These Before Answering)

MANDATORY for VQA tasks: Call `read_file` on each example image below and study how to compare object positions on the map.

**Example 1**: `skills/spatialmap/assets/example1.png`
- Question: "In which direction is Planetarium Prints relative to Police Supply Store?"
- Method: Locate both labels in the image. Planetarium Prints is to the upper-right of Police Supply Store.
- Answer: **A. Northeast** — upper-right = Northeast.

**Example 2**: `skills/spatialmap/assets/example2.png`
- Question: "In which direction is Wolf's Wardrobe relative to Tremor Toys?"
- Method: Locate both labels. Wolf's Wardrobe is to the lower-right of Tremor Toys.
- Answer: **C. Southeast** — lower-right = Southeast.

Use these examples to calibrate your label-finding and direction-mapping before solving the actual question.

---

## Input Modalities

### VQA (Image Only)
The image shows a 2D map with named objects at various positions.
1. Locate Object X and Object Y in the image by their labels.
2. Compare their pixel positions: is X to the right/left/above/below Y?
3. Map the relative position to one of: Northeast, Northwest, Southwest, Southeast.

### VTQA (Image + Textual Relations)
The question text contains pre-computed spatial relationships between objects, such as:

```
Police Supply Store is in the map.
Narwhal's Novelties is to the Northwest of Police Supply Store.
Planetarium Prints is to the Northeast of Police Supply Store.
Oz Oddities is to the Southwest of Police Supply Store.
```

**In VTQA mode:**
1. Scan the text for the line: "[X] is to the [DIRECTION] of [Y]" where X is the query object and Y is the reference object.
2. That line gives the direct answer.
3. If no direct line exists, chain relationships: find intermediate objects connecting X to Y and reason transitively.

---

## Solving Strategy

1. Parse the question: "In which direction is **[X]** relative to **[Y]**?" — X is the object you are locating; Y is the reference.
2. In VTQA: look for the exact line "[X] is to the [DIR] of [Y]". This is the answer.
3. In VQA: visually determine whether X is upper-right, upper-left, lower-right, or lower-left of Y.
4. The answer is always one of: **Northeast, Northwest, Southwest, Southeast**.

---

## Output Format
Respond with the matching option letter followed by the direction word.

**Example**: `A. Northeast`
