---
name: sem-service
description: How to interact with the SEM Service annotation application — a live browser-based canvas tool running at http://localhost:3000. Use this skill whenever the user asks you to zoom, pan, annotate, crop, inspect the canvas, or load an image in the paint app.
---

# SEM Service Skill

The SEM Service is a live annotation tool running at **http://localhost:3000**.
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
- "count the particles in the image"
- "describe / inspect / analyse what you see in the viewport"
- "use your VLM to look at the current tile"
- "what do you see in the SEM image?"

---

## Inspecting the canvas with VLM (`get_canvas_image` → `analyze_sandbox_image`)

To count, describe, or analyse what is currently shown in the SEM viewport, always
use the canvas export — **not** a browser screenshot tool:

```python
# 1. Export the current canvas (background image + any annotations) as a PNG
img = paint_canvas("get_canvas_image")          # returns {saved_to, width, height}
# saved_to is a sandbox path, e.g. /workspace/screenshots/paint_<ts>/canvas_export.png

# 2. Ask the VLM a question about it
result = analyze_sandbox_image(
    img["saved_to"],
    question="Count the number of dark particles visible in this SEM image. Return only the integer count."
)
print(result["answer"])   # e.g. "7"
```

> **Rule: NEVER use `screenshot_and_ask` to inspect the SEM canvas.**
> `screenshot_and_ask` takes a live browser screenshot and will capture whatever
> the browser is showing (possibly a new-tab page, a different window, etc.).
> `get_canvas_image` exports the SEM service canvas directly — it always contains
> the correct image regardless of what is visible on screen.

This pattern works in **any** mode (image, grid, atlas) and after segmentation overlays
have been applied.  The exported PNG includes the background tile **plus** all
annotation objects drawn on top.

---

## Core principle: always reason from state first

**Never guess coordinates.** Always read the canvas state first, then compute
what you need from the data returned.

---

## Verify state after every action — `get_sem_status`

After **every** `paint_canvas`, `segment_viewport`, or camera navigation call,
you **MUST** call `get_sem_status` before your next action:

```python
paint_canvas("camera_goto", {"x": 2, "y": 1})
status = get_sem_status()   # waits 1.5 s, then returns:
# status["background"]     → current background filename
# status["filters"]        → {brightness, contrast}
# status["tile_position"]  → {region, fw, x, y} (grid mode only)
# status["ui_mode"]        → "grid" | "image" | "atlas"
# status["segmentation"]   → {enabled, count, ...}
# status["session"]        → {filterAdjustments, vlmSnapshots}
```

Do not assume the action succeeded until you have read `get_sem_status`.

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

**Rectangular crop** (bounding box):
```python
result = paint_canvas("crop", {"x": x, "y": y, "width": w, "height": h})
# The paint app now shows the cropped sub-image as the new background.
# The canvas resets to exactly w × h pixels.
# Annotations inside the crop are translated to the new origin.
```

**Shape-masked crop** (use the drawn shape as the crop boundary — useful for ellipses):
```python
result = paint_canvas("crop", {"shapeId": "obj_abc123"})
# Crops to the bounding box of the shape AND masks to its exact outline.
# 'shapeId' is obtained from paint_canvas("list") or paint_canvas("state").
```

**IMPORTANT**: Always pass params as the second argument dict — never as flat keyword arguments:
```python
# CORRECT ✓
paint_canvas("crop", {"x": 100, "y": 100, "width": 400, "height": 300})
paint_canvas("crop", {"shapeId": "obj_abc123"})

# WRONG ✗  — do not pass x/y/shapeId as top-level args
paint_canvas(action="crop", x=100, y=100, width=400, height=300)
```

After crop, the canvas automatically switches to **Image mode** (even if you were in Grid mode). The grid tile position is remembered — click 🔬 Grid to return to it.

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

## Circular / ellipse crop

When the user asks to crop in a **circle** or **ellipse**, you cannot do this with a plain rectangular crop. You must:

1. **Draw an ellipse** annotation at the desired centre and radius.
2. **Get its `id`** from the returned object.
3. **Crop using `shapeId`** — the server clips to the ellipse outline and crops the bounding box.

```python
# Step 1 — draw the ellipse (cx, cy = centre; rx, ry = half-axes)
ellipse_obj = paint_canvas("ellipse", {
    "cx": 512, "cy": 512, "rx": 200, "ry": 200,
    "stroke": "#00ff00", "strokeWidth": 2, "createdBy": "model"
})
ellipse_id = ellipse_obj["id"]

# Step 2 — crop to that ellipse (bounding box + circular mask)
result = paint_canvas("crop", {"shapeId": ellipse_id})
# The result is a square image where pixels outside the circle are transparent.
```

For a **perfect circle** centred on a region of interest: set `rx == ry`. Compute centre from whatever coordinates you read from state, then add padding if desired:

```python
state = paint_canvas("state")
W, H = state["canvas"]["width"], state["canvas"]["height"]

# Circle around the centre of the image, radius = 40% of the shorter side
r = int(min(W, H) * 0.4)
cx, cy = W // 2, H // 2

ellipse_obj = paint_canvas("ellipse", {"cx": cx, "cy": cy, "rx": r, "ry": r,
                                        "stroke": "#00ff00", "createdBy": "model"})
result = paint_canvas("crop", {"shapeId": ellipse_obj["id"]})
```

**Do NOT** do a rectangular crop and then try to process the image separately — use `shapeId` directly.

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
| `load_tile_grid` | Scan tile dataset, load default (or specified) tile as background |
| `camera_left` | Move to left neighbor tile (x-1) |
| `camera_right` | Move to right neighbor tile (x+1) |
| `camera_up` | Move to upper neighbor tile (y-1) |
| `camera_down` | Move to lower neighbor tile (y+1) |
| `camera_goto` | Jump directly to tile by `{x, y}` (optionally `region`, `fw`) |
| `camera_state` | Read current uiMode and tileGrid position |
| `set_canvas_mode` | Switch between `"image"` and `"grid"` modes |
| `set_filters` | Set `{brightness, contrast}` (0–300) |
| `randomize_filters` | Scramble filters and capture reference histogram |
| `atlas_enter` | Enter stitched atlas view for the current region + fw |
| `atlas_exit` | Return from atlas view to grid mode |
| `atlas_state` | Read atlas state including `tile_coords` (key-point lookup) and `tiles_traversed` (every tile a shape enters) annotation mappings |
| `atlas_fit` | Fit the atlas view to the viewport |
| `segment_viewport` | Run SAM2 segmentation on the current view (requires `segmentationEnabled=true`) |
| `get_sem_status` | Wait, then return background, filters, tile_position, ui_mode, segmentation, session |

---

## Grid Dataset Mode (tile navigation)

The paint canvas can operate in **Grid Dataset** mode to navigate a mosaic of microscopy tiles
(SEM/TEM images tiled as `Region{NNN}_y{YY}_x{XX}_fw{FW}um[_1].tiff`).

### Initialise

```python
# Load dataset and jump to default tile (lowest region, y=0, x=0)
paint_canvas("load_tile_grid")

# Jump to a specific tile on init
paint_canvas("load_tile_grid", {"region": "Region018", "fw": 300, "x": 2, "y": 1})
```

Annotations are cleared when a new tile is loaded.

### Check current position

```python
s = paint_canvas("camera_state")
# s["tileGrid"] → {loaded, datasetName, currentRegion, currentFw, currentX, currentY, tileCount, regions}
```

### Navigate

```python
paint_canvas("camera_left")   # x − 1
paint_canvas("camera_right")  # x + 1
paint_canvas("camera_up")     # y − 1
paint_canvas("camera_down")   # y + 1

# If no neighbour exists the response contains {"error": "No tile at …", "currentTile": {…}}
# and the canvas position is unchanged — do not assume movement happened.
```

### Jump to coordinates

```python
paint_canvas("camera_goto", {"x": 3, "y": 1})
# Override region or fw if needed:
paint_canvas("camera_goto", {"x": 0, "y": 0, "region": "Region020", "fw": 120})
```

### Switch back to normal image mode

```python
paint_canvas("set_canvas_mode", {"mode": "image"})
```

### Compare the same tile across two focal widths

To compare tile (x=2, y=1) at fw=851 vs fw=869:

```python
# Step 1 — make sure grid mode is active
paint_canvas("load_tile_grid")   # only needed if not already in grid mode

# Step 2 — jump to tile at first fw
paint_canvas("camera_goto", {"x": 2, "y": 1, "fw": 851})
img_a = paint_canvas("get_canvas_image", {"filename": "fw851_x2_y1.png"})
# img_a["saved_to"] → sandbox path you can read/analyse

# Step 3 — jump to same tile at second fw (region stays the same)
paint_canvas("camera_goto", {"x": 2, "y": 1, "fw": 869})
img_b = paint_canvas("get_canvas_image", {"filename": "fw869_x2_y1.png"})

# Step 4 — now you have two image paths; compare them visually or programmatically
# sandbox_read_file(img_a["saved_to"]) etc.
```

**Important**: `camera_goto` with a different `fw` switches to the matching tile for that focal width. The region stays the same. Both images are saved to `/workspace/screenshots/paint_<timestamp>/` and are readable by the sandbox.

### Systematic row scan

```python
paint_canvas("load_tile_grid")        # start at x=0, y=0
s = paint_canvas("camera_state")
y = s["tileGrid"]["currentY"]

while True:
    img = paint_canvas("get_canvas_image")
    # … analyse img["saved_to"] …
    result = paint_canvas("camera_right")
    if "error" in result:
        break   # hit right boundary
```

---

## Atlas Mode

> **Rule: always use `atlas_state`, never parse SVG paths manually.**
> When in atlas mode, call `paint_canvas("atlas_state")` to read annotation data.
> Every annotation in the response already contains `tiles_traversed` — the
> pre-computed ordered list of tiles the shape enters.  Do **not** call
> `paint_canvas("state")` in atlas mode and do **not** write `execute` / Python
> code to parse SVG path strings or compute tile indices yourself.

Atlas mode stitches all tiles of the current region + focal width into a single
overview image. Useful for selecting a structural ROI before zooming into
individual tiles.

```python
# Enter atlas (must already be in grid mode with a tile loaded)
paint_canvas("atlas_enter")
get_sem_status()   # wait for render; ui_mode → "atlas"

# Fit to viewport
paint_canvas("atlas_fit")

# Read atlas state — each annotation carries two coordinate fields:
#   tile_coords      → key-point breakdown (start/end/center/origin) with
#                      {tile_x, tile_y, pixel_x, pixel_y} per point
#   tiles_traversed  → ORDERED list of every {tile_x, tile_y} the shape
#                      enters (full traversal, not just endpoints)
state = paint_canvas("atlas_state")

# Exit atlas — returns to the tile that was active when you entered
paint_canvas("atlas_exit")
get_sem_status()   # ui_mode → "grid"
```

**When to use atlas mode:**
- Spatial overview of a region before drilling into individual tiles
- Finding which tile contains a structural feature (use `tile_coords` from `atlas_state`)
- Case Study 2: follow a drawn path — read `tiles_traversed` to get the exact tile list

**Atlas + annotation coordinate mapping (single point):**

```python
paint_canvas("atlas_enter")
# Draw a dot on the feature of interest
dot = paint_canvas("dot", {"cx": 540, "cy": 320, "radius": 6,
                            "fill": "#ff0000", "createdBy": "model"})
# Get which tile that dot falls in
state = paint_canvas("atlas_state")
annotation = next(a for a in state["annotations"] if a["id"] == dot["id"])
coords = annotation["tile_coords"]["center"]
# coords → {"tile_x": 0, "tile_y": 0, "pixel_x": 540, "pixel_y": 320}

# Navigate directly to that tile
paint_canvas("atlas_exit")
paint_canvas("camera_goto", {"x": coords["tile_x"], "y": coords["tile_y"]})
get_sem_status()
```

**Path following via `tiles_traversed` (Case Study 2):**

When the user draws a line, arrow, freehand stroke, rect, or ellipse on the atlas,
`atlas_state` returns a `tiles_traversed` list on each annotation — every tile that
shape crosses, computed by DDA grid traversal (lines/arrows/freehand) or perimeter
sampling (rects/ellipses).  Use this to drive systematic tile navigation:

```python
# 1. Enter atlas and read the drawn path
paint_canvas("atlas_enter")
get_sem_status()
state = paint_canvas("atlas_state")

# 2. Find the path annotation (drawn by human or eval script)
path_obj = next(
    a for a in state["annotations"]
    if a["createdBy"] == "eval" and a["type"] in ("line", "freehand", "arrow")
)
tiles_to_visit = path_obj["tiles_traversed"]
# → [{"tile_x": 3, "tile_y": 14}, {"tile_x": 4, "tile_y": 14}, ...]

# 3. Exit atlas and visit each tile
paint_canvas("atlas_exit")
get_sem_status()

results = {}
for tile in tiles_to_visit:
    paint_canvas("camera_goto", {"x": tile["tile_x"], "y": tile["tile_y"]})
    get_sem_status()
    img = paint_canvas("get_canvas_image")
    answer = analyze_sandbox_image(
        img["saved_to"],
        question="Count the number of particles visible in this SEM image. Return only the integer count."
    )
    results[f"({tile['tile_x']}, {tile['tile_y']})"] = answer["answer"]

# 4. Report
for tile_key, count in results.items():
    print(f"tile {tile_key}: {count}")
print(f"total: {sum(int(re.search(r'\\d+', v).group()) for v in results.values() if re.search(r'\\d+', v))}")
```

---

## Image Filters

```python
# Set brightness + contrast (0–300)
paint_canvas("set_filters", {"brightness": 130, "contrast": 85})
get_sem_status()   # verify filters applied

# Randomize (Case Study 1 start — captures reference histogram first)
paint_canvas("randomize_filters")
get_sem_status()   # filters now randomised
```

---

## SAM2 Segmentation

The `segment_viewport` tool calls the `agent-api` SAM2 endpoint and overlays
the result on the canvas. The human must have `segmentationEnabled = true` in
the UI (click **▶ Run** panel) before calling.

```python
result = segment_viewport()
# result → {ok, count, centroids: [[x,y],...], bboxes: [[x,y,w,h],...], mask_png: "<path>"}
```

- SAM2 is **deterministic**: same image + same hyperparameters = identical output every run.
- Only the VLM interpretation varies across runs.
- Call `get_sem_status()` after to confirm segmentation count in state.

---

## Histogram Evaluation Workflow (Case Study 1)

Goal: restore a randomised SEM image to its reference histogram.

```python
# 1. Start a run — randomises filters AND saves reference histogram
paint_canvas("randomize_filters")
get_sem_status()    # confirm filters changed

# 2. Iteratively adjust filters toward the reference
for iteration in range(5):
    # Read the current score
    score_data = paint_canvas("histogram_score")   # {score, wasserstein, clipping_penalty}
    if score_data["score"] < 0.05:
        break   # good enough

    # Adjust based on histogram shape (read reference and current)
    ref   = paint_canvas("histogram_reference")    # {bins: [...], range: [lo, hi]}
    cur   = paint_canvas("histogram_current")       # same schema + score

    # Make an informed adjustment
    paint_canvas("set_filters", {"brightness": <new_b>, "contrast": <new_c>})
    get_sem_status()

# 3. Capture result once satisfied
import subprocess
subprocess.run([
    "python",
    "/workspace/skills/master-skill/sem-histogram-eval/sem_histogram_error.py",
    "--paint-url", "http://host.docker.internal:3000"
])
# Output: timestamped result JSON + PNG exports saved to /workspace/screenshots/
```

**Scoring**: lower is better.
- `wasserstein` — earth-mover distance between current and reference histogram
- `clipping_penalty` — penalty for histogram bins clipped at 0 or 255
- `score` = wasserstein + clipping_penalty
