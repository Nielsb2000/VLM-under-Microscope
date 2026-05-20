# Paint Service

A lightweight, Dockerized image annotation canvas that supports simultaneous **human browser interaction** and **AI model REST API control** on the same shared canvas, with real-time sync via Server-Sent Events (SSE).

---

## Quick Start

### 1. Build and Run

```bash
cd paint-service

# Using Docker directly (uploads lost on container restart)
docker build -t paint-service .
docker run -d --name paint-service -p 3000:3000 paint-service

# OR using docker-compose (uploads persisted via volume mount)
docker-compose up -d
```

### 2. Verify It Is Running

```bash
curl http://localhost:3000/api/canvas/state
```

Open in your browser: **http://localhost:3000**

### 3. Stop / Restart

```bash
docker-compose down
docker-compose restart
docker-compose logs -f paint-service
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│   Browser (Human)                                    │
│     Fabric.js canvas  ←──SSE──┐                     │
│     Draw, select, move        │                     │
└───────────────┬───────────────┘                     │
                │ HTTP POST /api/draw/*               │
                ▼                                     │
┌──────────────────────────────────────────────────────┤
│   Docker: paint-service (port 3000)                  │
│                                                      │
│   ┌────────────────────────────────────────────┐    │
│   │  Express server  (server/index.js)          │    │
│   │   │                                         │    │
│   │   ├─ /api/canvas   canvas lifecycle, crop   │    │
│   │   ├─ /api/draw     shape drawing endpoints  │    │
│   │   ├─ /api/objects  CRUD for annotations     │    │
│   │   ├─ /api/export   PNG / JSON export         │    │
│   │   ├─ /api/viewport zoom / pan / cursor      │    │
│   │   └─ /api/images   upload management        │    │
│   │                                             │    │
│   │  state.js  (in-memory + SSE broadcast)      │    │
│   │  renderer.js (server-side PNG via canvas)   │    │
│   └────────────────────────────────────────────┘    │
│                                                      │
│   uploads/   (background images + crop exports)     │
└──────────────────────────────────────────────────────┘
                ▲
                │ HTTP POST /api/draw/*, /api/canvas/*, /api/viewport/*
                │
┌───────────────┴──────────────────────────────────────┐
│   AI Agent (paint_canvas LangChain tool)             │
│     agent_tools.py → _paint() → REST API             │
└──────────────────────────────────────────────────────┘
```

---

## Versions & Dependencies

| Component | Version |
|-----------|---------|
| Node.js | ≥ 18 |
| Express | ^4.18.2 |
| canvas (node-canvas) | ^2.11.2 |
| multer | ^1.4.5-lts.1 |
| uuid | ^9.0.0 |
| Fabric.js (browser) | 5.x (CDN) |

### Native build deps (installed in Docker image)

`libcairo2-dev`, `libpango1.0-dev`, `libjpeg-dev`, `librsvg2-dev` — required by node-canvas for server-side PNG rendering.

---

## Folder Structure

```
paint-service/
├── Dockerfile
├── docker-compose.yml
├── package.json
├── test_api.sh              # Smoke test script
├── uploads/                 # Background images + crop outputs (volume-mounted)
├── client/
│   ├── index.html           # Single-page app shell
│   ├── app.js               # Fabric.js canvas logic + SSE client
│   └── style.css            # Dark Catppuccin theme
└── server/
    ├── index.js             # Express entry point, route mounting
    ├── state.js             # In-memory state + SSE broadcast
    ├── renderer.js          # Server-side PNG render (node-canvas)
    └── routes/
        ├── canvas.js        # Canvas lifecycle, load-image, crop
        ├── draw.js          # Shape drawing endpoints
        ├── objects.js       # Object CRUD
        ├── export.js        # PNG / JSON export
        ├── viewport.js      # Zoom, pan, cursor control
        └── images.js        # List + load uploaded images
```

---

## Human Browser UI

Open **http://localhost:3000** in your browser.

### Drawing Tools

| Key | Tool |
|-----|------|
| `S` | Select / move |
| `R` | Rectangle |
| `E` | Ellipse |
| `A` | Arrow |
| `D` | Dot / marker |
| `P` | Freehand pen |
| `T` | Text |
| `Del` | Delete selected object |
| `Esc` | Return to select tool |

### Top Bar Controls

- **Load Image** — upload a local file or paste a URL
- **Colour picker** — stroke / fill colour
- **Stroke width** — line thickness slider
- **Export PNG** — download annotated canvas as PNG
- **Export JSON** — download annotation data as JSON
- **Clear** — remove all annotations (keep background)

### Real-Time Sync

All model actions (draw, crop, zoom, cursor) appear on the browser instantly via SSE — no refresh needed.

---

## Model REST API

Every annotation object carries `id`, `type`, `createdBy` (`"human"` | `"model"`), `createdAt`, and optional `label`.

### Canvas State

```bash
# Full state (canvas dims, objects, viewport, cursor)
curl http://localhost:3000/api/canvas/state

# Reset / new canvas
curl -X POST http://localhost:3000/api/canvas/new \
  -H 'Content-Type: application/json' -d '{"width":1200,"height":800}'

# Clear annotations (keep background)
curl -X POST http://localhost:3000/api/canvas/clear

# Live SSE stream — emits full state JSON on every change
curl -N http://localhost:3000/api/canvas/events
```

### Load a Background Image

```bash
# From URL
curl -X POST http://localhost:3000/api/canvas/load-image \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/image.jpg"}'

# From local file (multipart)
curl -X POST http://localhost:3000/api/canvas/load-image \
  -F 'image=@/path/to/image.png'
```

### Drawing Shapes

```bash
# Rectangle
curl -X POST http://localhost:3000/api/draw/rect \
  -H 'Content-Type: application/json' \
  -d '{"x":100,"y":120,"width":200,"height":140,"stroke":"#ff0000","strokeWidth":3,"label":"defect","createdBy":"model"}'

# Ellipse
curl -X POST http://localhost:3000/api/draw/ellipse \
  -H 'Content-Type: application/json' \
  -d '{"cx":400,"cy":300,"rx":80,"ry":50,"stroke":"#00ff00","createdBy":"model"}'

# Arrow
curl -X POST http://localhost:3000/api/draw/arrow \
  -H 'Content-Type: application/json' \
  -d '{"x1":50,"y1":50,"x2":300,"y2":200,"stroke":"#0000ff","strokeWidth":2,"createdBy":"model"}'

# Dot
curl -X POST http://localhost:3000/api/draw/dot \
  -H 'Content-Type: application/json' \
  -d '{"cx":600,"cy":400,"radius":8,"fill":"#ff00ff","createdBy":"model"}'

# Line
curl -X POST http://localhost:3000/api/draw/line \
  -H 'Content-Type: application/json' \
  -d '{"x1":0,"y1":0,"x2":800,"y2":600,"stroke":"#ffff00","createdBy":"model"}'

# Text
curl -X POST http://localhost:3000/api/draw/text \
  -H 'Content-Type: application/json' \
  -d '{"x":60,"y":40,"text":"look here","fontSize":20,"fill":"#ffffff","createdBy":"model"}'

# Bulk (multiple shapes in one request)
curl -X POST http://localhost:3000/api/canvas/ops \
  -H 'Content-Type: application/json' \
  -d '{
    "operations": [
      {"type":"rect","x":50,"y":60,"width":180,"height":120,"stroke":"#ff0000","createdBy":"model"},
      {"type":"text","x":55,"y":55,"text":"defect region","fontSize":16,"createdBy":"model"}
    ]
  }'
```

### Object CRUD

```bash
curl http://localhost:3000/api/objects                      # list all
curl http://localhost:3000/api/objects/<id>                 # get one

curl -X PATCH http://localhost:3000/api/objects/<id> \      # update
  -H 'Content-Type: application/json' \
  -d '{"label":"confirmed","stroke":"#ff6600"}'

curl -X DELETE http://localhost:3000/api/objects/<id>       # delete
```

### Crop (Image Sub-Region)

Crops the background image to the given bounding box and loads it as the new canvas background. All annotations are cleared. Viewport resets to 1:1.

```bash
# Rectangular crop (canvas coordinates)
curl -X POST http://localhost:3000/api/canvas/crop \
  -H 'Content-Type: application/json' \
  -d '{"x":200,"y":150,"width":400,"height":300}'

# Shape-masked crop (ellipse/freehand — pixels outside shape filled with canvas background colour)
curl -X POST http://localhost:3000/api/canvas/crop \
  -H 'Content-Type: application/json' \
  -d '{"shapeId":"obj_abc123"}'
```

### Viewport Control

Changes what the human sees in real time. Does **not** alter the underlying image.

```bash
# Set zoom (zoom=1 is 1:1, higher = zoomed in)
curl -X POST http://localhost:3000/api/viewport/zoom \
  -H 'Content-Type: application/json' \
  -d '{"zoom":2,"centerX":512,"centerY":384}'

# Pan (canvas point at viewport top-left)
curl -X POST http://localhost:3000/api/viewport/pan \
  -H 'Content-Type: application/json' \
  -d '{"x":100,"y":80}'

# Fit a region into view
curl -X POST http://localhost:3000/api/viewport/zoom-to-region \
  -H 'Content-Type: application/json' \
  -d '{"x":200,"y":150,"width":400,"height":300,"padding":40}'

# Fit a specific annotation into view by id
curl -X POST http://localhost:3000/api/viewport/zoom-to-object \
  -H 'Content-Type: application/json' \
  -d '{"id":"obj_abc123","padding":40}'

# Reset to 1:1
curl -X POST http://localhost:3000/api/viewport/reset

# Read current viewport
curl http://localhost:3000/api/viewport
```

### Model Cursor

Shows a visible cyan cursor overlay on the human's screen.

```bash
# Show cursor at canvas position
curl -X POST http://localhost:3000/api/viewport/cursor \
  -H 'Content-Type: application/json' \
  -d '{"x":480,"y":320,"visible":true,"label":"examining…"}'

# Hide cursor
curl -X DELETE http://localhost:3000/api/viewport/cursor
```

### Export

```bash
# Flattened PNG (background + all annotations burned in)
curl http://localhost:3000/api/export/png -o annotated.png

# Structured JSON
curl http://localhost:3000/api/export/json

# Package (returns download URLs for both)
curl -X POST http://localhost:3000/api/export/package
```

### Image Management

```bash
curl http://localhost:3000/api/images                       # list uploaded files

curl -X POST http://localhost:3000/api/images/load \        # load existing file
  -H 'Content-Type: application/json' \
  -d '{"filename":"abc.jpg"}'
```

---

## Annotation JSON Schema

```json
{
  "canvas": {
    "width": 1024,
    "height": 768,
    "backgroundImage": "a50ff844.jpg"
  },
  "objects": [
    {
      "id": "obj_a1b2c3d4",
      "type": "rect",
      "x": 100, "y": 120, "width": 200, "height": 140,
      "stroke": "#ff0000", "fill": "transparent", "strokeWidth": 3,
      "label": "possible defect",
      "createdBy": "human",
      "createdAt": "2026-04-21T08:24:45.397Z"
    }
  ],
  "viewport": { "zoom": 1, "panX": 0, "panY": 0 },
  "cursor": { "x": null, "y": null, "visible": false, "label": "" }
}
```

**Per-type geometry fields:**

| Type | Fields |
|------|--------|
| `rect` | `x, y, width, height` |
| `ellipse` | `cx, cy, rx, ry` |
| `arrow` | `x1, y1, x2, y2` |
| `dot` | `cx, cy, radius` |
| `line` | `x1, y1, x2, y2` |
| `text` | `x, y, text, fontSize, fontFamily` |
| `freehand` | `path` (SVG path array), `left, top, scaleX, scaleY` |

---

## Agent Integration (Python)

The service is exposed to the AI agent through the `paint_canvas` LangChain tool in `agent_tools.py`. All actions are dispatched via a single tool call:

```python
from agent_tools import paint_canvas

# Read state
state = paint_canvas.invoke({"action": "state"})

# Draw a rectangle
paint_canvas.invoke({"action": "rect", "params": {
    "x": 100, "y": 120, "width": 200, "height": 140,
    "stroke": "#ff0000", "label": "defect", "createdBy": "model"
}})

# Crop to a bounding box
paint_canvas.invoke({"action": "crop", "params": {
    "x": 200, "y": 150, "width": 400, "height": 300
}})

# Get canvas as base64 image (for vision model)
result = paint_canvas.invoke({"action": "get_canvas_image"})
# result["data_url"] → "data:image/png;base64,…"
```

The skill file at `/workspace/skills/master-skill/paint-service/SKILL.md` provides the agent with step-by-step workflows for common tasks (crop, zoom, annotate, etc.).

---

## Running Tests

```bash
# Start the service first, then:
bash test_api.sh http://localhost:3000
```

---

## Troubleshooting

### Container won't start

```bash
# Check if port 3000 is already in use
sudo lsof -i :3000

# View container logs
docker-compose logs paint-service
```

### Uploaded images not persisted after restart

Ensure the `uploads/` volume mount is active:

```bash
# Check mount in docker-compose.yml
grep -A2 volumes paint-service/docker-compose.yml

# Verify directory exists on host
ls -la paint-service/uploads/
```

### Canvas shows blank / no background

- Confirm the image file exists: `curl http://localhost:3000/api/images`
- Check state: `curl http://localhost:3000/api/canvas/state`
- Reload by filename: `curl -X POST http://localhost:3000/api/images/load -H 'Content-Type: application/json' -d '{"filename":"your_file.jpg"}'`

### SSE stream disconnects

Fabric.js SSE client auto-reconnects. If the model is not seeing live updates, confirm the service is running and the SSE endpoint is reachable:

```bash
curl -N http://localhost:3000/api/canvas/events
```


## Quick start

```bash
# Build and run
cd paint-service
docker build -t paint-service .
docker run -p 3000:3000 paint-service

# Or with docker-compose (persists uploads across restarts)
docker-compose up -d
```

Open **http://localhost:3000** in your browser.

---

## Human UI

| Shortcut | Tool |
|----------|------|
| `S` | Select / move |
| `R` | Rectangle |
| `E` | Ellipse |
| `A` | Arrow |
| `D` | Dot / marker |
| `P` | Freehand pen |
| `T` | Text |
| `Del` | Delete selected |
| `Esc` | Back to select |

Use the top bar to **load an image**, pick a **colour** and **stroke width**, and export **PNG / JSON**.

---

## Model API

All endpoints are under `/api/`.  
Every annotation object carries `id`, `type`, `createdBy` (`"human"` or `"model"`), `createdAt`, and optional `label`.

### Canvas lifecycle

```bash
# Create / reset canvas
curl -X POST http://localhost:3000/api/canvas/new \
  -H 'Content-Type: application/json' -d '{"width":1200,"height":800}'

# Clear annotations (keep background)
curl -X POST http://localhost:3000/api/canvas/clear

# Get full state
curl http://localhost:3000/api/canvas/state
```

### Load a background image

```bash
# From URL
curl -X POST http://localhost:3000/api/canvas/load-image \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/image.jpg"}'

# From file
curl -X POST http://localhost:3000/api/canvas/load-image \
  -F 'image=@/path/to/image.png'
```

### Drawing

```bash
# Rectangle
curl -X POST http://localhost:3000/api/draw/rect \
  -H 'Content-Type: application/json' \
  -d '{"x":100,"y":120,"width":200,"height":140,"stroke":"#ff0000","strokeWidth":3,"label":"defect","createdBy":"model"}'

# Ellipse
curl -X POST http://localhost:3000/api/draw/ellipse \
  -H 'Content-Type: application/json' \
  -d '{"cx":400,"cy":300,"rx":80,"ry":50,"stroke":"#00ff00","createdBy":"model"}'

# Arrow
curl -X POST http://localhost:3000/api/draw/arrow \
  -H 'Content-Type: application/json' \
  -d '{"x1":50,"y1":50,"x2":300,"y2":200,"stroke":"#0000ff","strokeWidth":2,"createdBy":"model"}'

# Dot
curl -X POST http://localhost:3000/api/draw/dot \
  -H 'Content-Type: application/json' \
  -d '{"cx":600,"cy":400,"radius":8,"fill":"#ff00ff","createdBy":"model"}'

# Line
curl -X POST http://localhost:3000/api/draw/line \
  -H 'Content-Type: application/json' \
  -d '{"x1":0,"y1":0,"x2":800,"y2":600,"stroke":"#ffff00","createdBy":"model"}'

# Text
curl -X POST http://localhost:3000/api/draw/text \
  -H 'Content-Type: application/json' \
  -d '{"x":60,"y":40,"text":"mark this area","fontSize":20,"fill":"#ffffff","createdBy":"model"}'
```

### Bulk operations (efficient for models)

```bash
curl -X POST http://localhost:3000/api/canvas/ops \
  -H 'Content-Type: application/json' \
  -d '{
    "operations": [
      {"type":"rect","x":50,"y":60,"width":180,"height":120,"stroke":"#ff0000","strokeWidth":3,"createdBy":"model"},
      {"type":"text","x":60,"y":40,"text":"defect region","fontSize":20,"createdBy":"model"}
    ]
  }'
```

### Object CRUD

```bash
# List all objects
curl http://localhost:3000/api/objects

# Get one
curl http://localhost:3000/api/objects/<id>

# Update (PATCH)
curl -X PATCH http://localhost:3000/api/objects/<id> \
  -H 'Content-Type: application/json' \
  -d '{"label":"confirmed","stroke":"#ff6600"}'

# Delete
curl -X DELETE http://localhost:3000/api/objects/<id>
```

### Export

```bash
# Flattened PNG (background + all annotations burned in)
curl http://localhost:3000/api/export/png -o annotated.png

# Structured JSON
curl http://localhost:3000/api/export/json -o annotations.json

# Package descriptor (returns URLs for both)
curl -X POST http://localhost:3000/api/export/package
```

### Live state stream (SSE)

```bash
curl -N http://localhost:3000/api/canvas/events
# Emits full state as JSON on every change
```

---

## Annotation JSON schema

```json
{
  "canvas": {
    "width": 1200,
    "height": 800,
    "backgroundImage": "uploads/abc.png"
  },
  "objects": [
    {
      "id": "obj_a1b2c3d4",
      "type": "rect",
      "x": 100, "y": 120, "width": 200, "height": 140,
      "stroke": "#ff0000", "fill": "transparent", "strokeWidth": 3,
      "label": "possible defect",
      "createdBy": "human",
      "createdAt": "2026-04-20T12:00:00.000Z"
    }
  ]
}
```

**Per-type geometry fields:**

| Type | Fields |
|------|--------|
| `rect` | `x, y, width, height` |
| `ellipse` | `cx, cy, rx, ry` |
| `arrow` | `x1, y1, x2, y2` |
| `dot` | `cx, cy, radius` |
| `line` | `x1, y1, x2, y2` |
| `text` | `x, y, text, fontSize, fontFamily` |
| `freehand` | `path` (SVG path string), `left, top` |

---

## Running tests

```bash
# Start service first, then:
bash test_api.sh http://localhost:3000
```

---

## Architecture

```
paint-service/
├── server/
│   ├── index.js          Express entry point
│   ├── state.js          In-memory canvas state + SSE broadcast
│   ├── renderer.js       Server-side PNG rendering (node-canvas)
│   └── routes/
│       ├── canvas.js     /api/canvas/*  (lifecycle, image, bulk ops, SSE)
│       ├── draw.js       /api/draw/*    (per-shape endpoints)
│       ├── objects.js    /api/objects/* (CRUD)
│       └── export.js     /api/export/*  (PNG, JSON, package)
├── client/
│   ├── index.html        Shell with toolbar + Fabric.js canvas
│   ├── app.js            Frontend logic (tools, SSE sync, export)
│   └── style.css
├── uploads/              Uploaded images (volume-mounted in Docker)
├── Dockerfile
├── docker-compose.yml
└── test_api.sh
```

Human edits and model API calls share the **same in-memory state**. The browser subscribes to a Server-Sent Events stream (`/api/canvas/events`) and incrementally updates the canvas whenever any change occurs — whether triggered by a human or the model API.
