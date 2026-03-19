---
name: spatialmap
description: Spatial map image + domain context with spatial direction rules for interpreting top-down maps.
---

# Spatial Map — Image + Domain Context

These example images explain the map format and how to interpret spatial directions.
Call `read_file` on each image path listed before answering.

---

## Example 1 — Understanding the Visual Format and Directions

Image: `skills/spatialmap/assets/map2497_q0.png`

In a spatial map image:
- The image shows a **2D map** with named objects (stores, landmarks) scattered at
  various positions.
- Each object is labelled with its name directly in the image.

**Direction rules** (for "In which direction is X relative to Y?"):
- X to the **right** of Y → **East**; to the **left** → **West**
- X **above** Y (higher on screen) → **North**; **below** → **South**
- Combine axes for diagonals: upper-right → **Northeast**, upper-left → **Northwest**,
  lower-right → **Southeast**, lower-left → **Southwest**

The answer is always one of: Northeast, Northwest, Southwest, Southeast.

---

## Example 2 — Finding Which Object Is in a Direction

Image: `skills/spatialmap/assets/map2498_q1.png`

**"Which object is in the [DIRECTION] of [Y]?"**  
1. Locate Y on the map.
2. Identify the quadrant in the given direction (e.g. Southwest = lower-left of Y).
3. Find which labelled object sits in that quadrant.
4. Select the matching option.

---

## Example 3 — Counting Objects in a Direction

Image: `skills/spatialmap/assets/map2499_q2.png`

**"How many objects are in the [DIRECTION] of [Y]?"**
1. Locate Y on the map.
2. Draw an imaginary axis through Y in the specified direction (e.g. North = above Y).
3. Count every other labelled object that lies strictly in that direction.
4. If no objects lie in that direction, the answer is 0.

---

Now apply this understanding to answer the actual question.
Give your final answer as: [Option Letter]. [Answer]
