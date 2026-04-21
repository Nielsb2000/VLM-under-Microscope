---
name: paint-service
description: How to interact with the Paint Service annotation application — a live browser-based canvas tool running at http://localhost:3000. Use this skill whenever the user asks you to zoom, pan, annotate, crop, inspect the canvas, or load an image in the paint app.
---

# Paint Service Skill

The Paint Service is a live annotation tool running at **http://localhost:3000**.
You interact with it exclusively through the `paint_canvas` tool.

The human sees the canvas in their browser in real time.
Every action you take appears on their screen instantly via SSE.

---

## When to use this skill

Read this skill when the user says anything like:
- "zoom in on the bounding box"
- "draw a rectangle around X"
- "load this image in the paint app"
- "what annotations are on the canvas?"
- "crop to the selected region"
- "move the cursor to X"
- "highlight this area"
- "show me what's on the canvas"

---

## Core principle: always reason from state first

**Never guess coordinates.** Always read the canvas state first, then compute
what you need from the data returned.

```python
state = paint_canvas("state")
# state contains:
# - state["canvas"]   → {width, height, backgroundImage}
# - state["objects"]  → list of annotations with coords, ids, createdBy
# - state["viewport"] → {zoom, panX, panY}
# - state["cursor"]   → {x, y, visible, label}
```

---

## "Zoom in on the bounding box I drew"

This is the primary workflow. Work through it step by step.

### Step 1 — read state and find the annotation

```python
state = paint_canvas("state")
objects = state["objects"]

# Find the most recent human-drawn rectangle
rects = [o for o in objects if o["type"] == "rect" and o.get("createdBy") == "human"]
if not rects:
    print("No bounding box found on the canvas.")
    # stop here
rect = rects[-1]
```

### Step 2 — compute crop coordinates yourself (with optional padding)

```python
padding = 20   # extra pixels of context around the box

x = max(0, rect["x"] - padding)
y = max(0, rect["y"] - padding)
w = rect["width"]  + padding * 2
h = rect["height"] + padding * 2
```

### Step 3 — move cursor to the centre so the user sees where you're looking

```python
cx = rect["x"] + rect["width"]  / 2
cy = rect["y"] + rect["height"] / 2
paint_canvas("cursor_move", {"x": cx, "y": cy, "label": "cropping here…"})
```

### Step 4 — crop the actual image at those coordinates

```python
result = paint_canvas("crop", {"x": x, "y": y, "width": w, "height": h})
# The paint app now shows the cropped sub-image as the new background.
# The canvas resets to exactly w × h pixels.
# Annotations inside the crop are translated to the new origin.
```

### Step 5 — hide cursor

```python
paint_canvas("cursor_hide")
```

---

## Zooming based on any region, not just an annotation

Read the canvas dimensions, reason about the region, then call crop directly:

```python
state = paint_canvas("state")
W = state["canvas"]["width"]
H = state["canvas"]["height"]

# e.g. top-left quarter
paint_canvas("crop", {"x": 0, "y": 0, "width": W // 2, "height": H // 2})
```

---

## Viewport zoom (no image change)

If the user just wants to navigate without changing the underlying image:

```python
# Fit a region into the viewport
paint_canvas("zoom_to_object", {"id": rect["id"], "padding": 40})
paint_canvas("zoom_to_region", {"x": 100, "y": 80, "width": 300, "height": 200})

# Reset to full view
paint_canvas("reset_viewport")
```

Use `crop` for pixel-level detail inspection.
Use `zoom_to_region/zoom_to_object` for navigation only.

---

## Drawing annotations

```python
paint_canvas("rect", {
    "x": 100, "y": 80, "width": 200, "height": 150,
    "stroke": "#ff0000", "strokeWidth": 2,
    "label": "crack", "createdBy": "model"
})

paint_canvas("arrow", {"x1": 50, "y1": 50, "x2": 300, "y2": 200,
                        "stroke": "#ffaa00", "label": "look here"})

paint_canvas("dot", {"cx": 400, "cy": 300, "radius": 6, "fill": "#ff0000"})

paint_canvas("text", {"x": 100, "y": 70, "text": "defect", "fontSize": 16, "fill": "#ff0000"})

# Multiple at once
paint_canvas("bulk", {"operations": [
    {"type": "rect", "x": 50, "y": 60, "width": 180, "height": 120,
     "stroke": "#ff0000", "createdBy": "model"},
    {"type": "text", "x": 55, "y": 55, "text": "region A", "createdBy": "model"},
]})
```

---

## Loading images

```python
imgs = paint_canvas("list_images")   # see what's uploaded
paint_canvas("load_image_by_name", {"filename": "example.png"})
paint_canvas("load_image", {"path": "/workspace/my_image.png"})
paint_canvas("load_image", {"url": "https://example.com/img.png"})
```

---

## Cursor control

```python
paint_canvas("cursor_move", {"x": 350, "y": 240, "label": "examining…"})
paint_canvas("cursor_hide")
```

---

## Managing objects

```python
paint_canvas("list")
paint_canvas("update", {"id": "obj_abc", "label": "confirmed", "stroke": "#ff6600"})
paint_canvas("delete", {"id": "obj_abc"})
paint_canvas("clear")
```

---

## Export

```python
paint_canvas("export_png", {"save_to": "/workspace/result.png"})
data = paint_canvas("export_json")
```

---

## Coordinate system

- Origin (0,0) is top-left. x increases right, y increases down.
- All coordinates are canvas pixels.
- Canvas size: state["canvas"]["width"] × state["canvas"]["height"].

---

## Tool reference

| Action | What it does |
|--------|-------------|
| `state` | Read canvas, objects, viewport |
| `list_images` | See available uploaded images |
| `load_image` | Load from path or URL |
| `load_image_by_name` | Load a previously uploaded file |
| `rect`, `ellipse`, `arrow`, `dot`, `line`, `text`, `freehand` | Draw shapes |
| `bulk` | Draw multiple shapes at once |
| `list` | Inspect all objects |
| `update` | Change object properties |
| `delete` | Remove one object |
| `clear` | Remove all objects |
| `zoom` | Set viewport zoom level |
| `pan` | Move viewport |
| `zoom_to_region` | Navigate to a bbox (viewport only, no image change) |
| `zoom_to_object` | Navigate to an annotation (viewport only, no image change) |
| `crop` | Crop image to coordinates and reload — agent computes coords from state |
| `reset_viewport` | Back to 1:1 zoom |
| `cursor_move` | Show/move the model cursor |
| `cursor_hide` | Hide the model cursor |
| `export_png` | Save flat PNG |
| `export_json` | Get annotation JSON |
