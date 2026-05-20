# SEM Service

A Dockerized SEM image annotation canvas with live browser UI and REST API control. Supports simultaneous human and AI agent interaction on the same shared canvas via Server-Sent Events (SSE).

---

## Quick Start

```bash
# From inside sem-service/ (standalone)
cd sem-service
docker-compose up -d

# Or from the project root (starts sem-service + agent-api together)
docker-compose up -d
```

Open in your browser: **http://localhost:3000**

```bash
# Verify
curl http://localhost:3000/api/canvas/state

# Logs
docker-compose logs -f sem-service

# Restart
docker-compose restart sem-service
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Browser (Human)                                     │
│    Fabric.js canvas  ←── SSE ──┐                     │
│    Draw, select, annotate      │                     │
└───────────────┬────────────────┘                     │
                │ HTTP REST API                        │
                ▼                                      │
┌──────────────────────────────────────────────────────┤
│  Docker: sem-service (port 3000)                     │
│                                                      │
│  Express server  (server/index.js)                   │
│   ├─ /api/canvas     canvas lifecycle, crop, SSE     │
│   ├─ /api/draw       shape drawing                   │
│   ├─ /api/objects    annotation CRUD                 │
│   ├─ /api/export     PNG / JSON export               │
│   ├─ /api/viewport   zoom / pan / cursor             │
│   ├─ /api/histogram  reference/randomized/result     │
│   ├─ /api/atlas      tile atlas mode                 │
│   ├─ /api/camera     tile grid navigation            │
│   ├─ /api/dataset    dataset listing                 │
│   ├─ /api/session    session counters                │
│   └─ /api/randomize  filter randomization            │
│                                                      │
│  state.js      in-memory state + SSE broadcast       │
│  renderer.js   server-side PNG render (node-canvas)  │
│  histogramUtils.js  histogram computation + scoring  │
│  tileGrid.js   tile dataset index                    │
└──────────────────────────────────────────────────────┘
                ▲
                │
┌───────────────┴──────────────────────────────────────┐
│  AI Agent (paint_canvas / get_sem_status tools)      │
│    agent_tools.py → _paint() → REST API              │
│    agent_api.py   → SAM2 /segment endpoint (3001)    │
└──────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
sem-service/
├── Dockerfile
├── docker-compose.yml
├── package.json
├── test_api.sh              # Smoke test script
├── client/
│   ├── index.html           # Single-page app shell
│   ├── app.js               # Fabric.js canvas + SSE client
│   └── style.css            # Dark Catppuccin theme
└── server/
    ├── index.js             # Express entry point, route mounting
    ├── state.js             # In-memory state + SSE broadcast
    ├── renderer.js          # Server-side PNG render (node-canvas)
    ├── histogramUtils.js    # Histogram computation, scoring, persistence
    ├── tileGrid.js          # Tile dataset index + navigation helpers
    └── routes/
        ├── canvas.js        # Canvas lifecycle, load-image, crop, SSE
        ├── draw.js          # Shape drawing endpoints
        ├── objects.js       # Object CRUD
        ├── export.js        # PNG / JSON export
        ├── viewport.js      # Zoom, pan, cursor control
        ├── images.js        # List + load uploaded images
        ├── histogram.js     # Histogram capture, reference, scoring
        ├── randomize.js     # Filter randomization + reference snapshot
        ├── atlas.js         # Atlas mode (stitched tile view)
        ├── camera.js        # Tile grid navigation
        ├── dataset.js       # Dataset listing endpoints
        ├── session.js       # Session counter tracking
        └── handoff.js       # Agent handoff utilities
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

### Header Controls

- **Load Image** — upload a local file or paste a URL
- **Colour picker** — stroke / fill colour
- **Stroke width** — line thickness slider
- **Fit** — fit image to viewport
- **🗺️ Atlas** — enter stitched atlas view of current region
- **Export PNG / JSON** — download annotated canvas
- **Clear** — remove all annotations (keep background)
- **Filter sliders** — brightness / contrast adjusters
- **🎲 Randomize** — scramble filters (starts a case-study run)

### Segmentation Panel (agent-facing)

`[● Centroids] [▭ BBoxes] [⬡ Mask] [T Text] [▶ Run] [✕ Clear]`

Toggle which SAM2 segmentation overlays are shown. **Run** triggers the agent's `segment_viewport` tool. **T Text** controls whether coordinate arrays are returned to the agent.

### Trace Panel

Displays the agent's latest reasoning trace. Use `[⬇]` to download the full trace as a structured `.txt` file.

---

## REST API Reference

Every annotation object carries `id`, `type`, `createdBy` (`"human"` | `"model"`), `createdAt`, and optional `label`.

### Canvas State

```bash
curl http://localhost:3000/api/canvas/state          # full state
curl -X POST http://localhost:3000/api/canvas/new   -H 'Content-Type: application/json' -d '{"width":1200,"height":800}'
curl -X POST http://localhost:3000/api/canvas/clear  # clear annotations
curl -N http://localhost:3000/api/canvas/events      # SSE stream
```

### Load a Background Image

```bash
# From URL
curl -X POST http://localhost:3000/api/canvas/load-image   -H 'Content-Type: application/json' -d '{"url":"https://example.com/img.jpg"}'

# From local file
curl -X POST http://localhost:3000/api/canvas/load-image -F 'image=@/path/to/img.png'
```

### Drawing Shapes

```bash
curl -X POST http://localhost:3000/api/draw/rect   -H 'Content-Type: application/json'   -d '{"x":100,"y":120,"width":200,"height":140,"stroke":"#ff0000","label":"defect","createdBy":"model"}'

curl -X POST http://localhost:3000/api/draw/ellipse   -H 'Content-Type: application/json'   -d '{"cx":400,"cy":300,"rx":80,"ry":50,"stroke":"#00ff00","createdBy":"model"}'

curl -X POST http://localhost:3000/api/draw/arrow   -H 'Content-Type: application/json'   -d '{"x1":50,"y1":50,"x2":300,"y2":200,"stroke":"#0000ff","createdBy":"model"}'

curl -X POST http://localhost:3000/api/draw/dot   -H 'Content-Type: application/json'   -d '{"cx":600,"cy":400,"radius":8,"fill":"#ff00ff","createdBy":"model"}'

# Bulk (single SSE broadcast)
curl -X POST http://localhost:3000/api/canvas/ops   -H 'Content-Type: application/json'   -d '{"operations":[{"type":"rect","x":50,"y":60,"width":180,"height":120,"stroke":"#ff0000","createdBy":"model"}]}'
```

### Object CRUD

```bash
curl http://localhost:3000/api/objects                # list all
curl -X PATCH http://localhost:3000/api/objects/<id>   -H 'Content-Type: application/json' -d '{"label":"confirmed"}'
curl -X DELETE http://localhost:3000/api/objects/<id>
```

### Filters

```bash
# Set brightness (0–300) and contrast (0–300); 100 = neutral
curl -X PUT http://localhost:3000/api/canvas/filters   -H 'Content-Type: application/json' -d '{"brightness":120,"contrast":90}'

# Read current filters
curl http://localhost:3000/api/canvas/state | python3 -m json.tool | grep filters
```

### Randomize (start a case-study run)

```bash
# Captures reference histogram, then randomises brightness/contrast
curl -X POST http://localhost:3000/api/randomize
```

### Histogram Evaluation

```bash
curl http://localhost:3000/api/histogram/reference         # reference bins JSON
curl http://localhost:3000/api/histogram/randomized        # randomized bins JSON
curl http://localhost:3000/api/histogram/current           # current bins + score JSON
curl http://localhost:3000/api/histogram/reference-image   # reference image PNG
curl http://localhost:3000/api/histogram/randomized-image  # randomized image PNG
curl http://localhost:3000/api/histogram/result-image      # result image PNG
```

### Tile Grid Navigation

```bash
curl http://localhost:3000/api/camera/state          # current tile position
curl -X POST http://localhost:3000/api/camera/goto   -H 'Content-Type: application/json' -d '{"x":2,"y":1,"region":"Region011","fw":120}'
curl -X POST http://localhost:3000/api/camera/right   # move to x+1
curl -X POST http://localhost:3000/api/camera/left    # move to x-1
curl -X POST http://localhost:3000/api/camera/up      # move to y-1
curl -X POST http://localhost:3000/api/camera/down    # move to y+1
```

### Atlas Mode

```bash
curl -X POST http://localhost:3000/api/atlas/enter    # enter atlas for current region/fw
curl -X POST http://localhost:3000/api/atlas/exit     # return to grid mode
curl http://localhost:3000/api/atlas/manifest?region=Region011&fw=120  # tile manifest
```

### Segmentation State

```bash
# Read current segmentation overlay
curl http://localhost:3000/api/canvas/state | python3 -m json.tool | grep segmentation

# Enable/disable segmentation panel
curl -X PUT http://localhost:3000/api/canvas/segmentation-enabled   -H 'Content-Type: application/json' -d '{"enabled":true}'

# Enable/disable coordinate text in agent responses
curl -X PUT http://localhost:3000/api/canvas/segmentation-text-enabled   -H 'Content-Type: application/json' -d '{"enabled":true}'

# Clear segmentation overlay
curl -X DELETE http://localhost:3000/api/canvas/segmentation
```

### Session Counters

```bash
curl http://localhost:3000/api/session
# {"filterAdjustments": 5, "vlmSnapshots": 3}
```

### Export

```bash
curl http://localhost:3000/api/export/png -o annotated.png    # flattened PNG
curl http://localhost:3000/api/export/json -o annotations.json
```

---

## Annotation JSON Schema

```json
{
  "canvas": { "width": 1920, "height": 1200, "backgroundImage": "/tile-assets/Region011_y00_x00_fw120um.tiff" },
  "objects": [
    {
      "id": "obj_a1b2c3d4", "type": "rect",
      "x": 100, "y": 120, "width": 200, "height": 140,
      "stroke": "#ff0000", "fill": "transparent", "strokeWidth": 3,
      "label": "particle", "createdBy": "model", "createdAt": "2026-05-20T10:00:00.000Z"
    }
  ],
  "viewport": { "zoom": 1, "panX": 0, "panY": 0 },
  "filters": { "brightness": 100, "contrast": 100 },
  "session": { "filterAdjustments": 0, "vlmSnapshots": 0 }
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
| `freehand` | `path` (SVG path array), `left, top` |

---

## Agent Integration

The service is controlled by the `paint_canvas` LangChain tool in `agent_tools.py`:

```python
from agent_tools import paint_canvas, get_sem_status

# Always verify state after any action
paint_canvas("camera_goto", {"x": 2, "y": 1})
status = get_sem_status()   # waits 1.5s then returns background, filters, tile_position, etc.

# Annotate
paint_canvas("rect", {"x": 100, "y": 120, "width": 200, "height": 140,
                       "stroke": "#ff0000", "label": "particle", "createdBy": "model"})

# Run SAM2 segmentation (requires segmentationEnabled=true in UI)
from agent_tools_vision import make_segment_viewport_tool
segment_viewport = make_segment_viewport_tool()
result = segment_viewport()
# result → {ok, count, centroids, bboxes, mask_png}
```

The skill file at `/workspace/skills/master-skill/sem-service/SKILL.md` provides the agent with step-by-step workflows.

---

## Running Tests

```bash
# Start service first, then:
bash test_api.sh http://localhost:3000
```

---

## Troubleshooting

### Container won't start

```bash
sudo lsof -i :3000                         # check port conflict
docker-compose logs sem-service            # view logs
```

### Canvas shows blank / no background

```bash
curl http://localhost:3000/api/images      # list uploaded images
curl http://localhost:3000/api/canvas/state | python3 -m json.tool
```

### SSE stream disconnects

The Fabric.js SSE client auto-reconnects. Verify the service is running:

```bash
curl -N http://localhost:3000/api/canvas/events
```

### Dataset tiles not loading

Ensure the dataset volume is mounted correctly in `docker-compose.yml` and that tiles match the pattern `Region{NNN}_y{YY}_x{XX}_fw{FW}um[_1].tiff`.
