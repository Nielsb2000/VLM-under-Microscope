// server/routes/canvas.js
const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');
const state = require('../state');

const UPLOADS_DIR = path.join(__dirname, '..', '..', 'uploads');
fs.mkdirSync(UPLOADS_DIR, { recursive: true });

const upload = multer({ dest: UPLOADS_DIR });

// ---- SSE stream ----
router.get('/events', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
  res.write('\n');
  // Push current state immediately on connect
  res.write(`data: ${JSON.stringify(state.getState())}\n\n`);
  state.addSseClient(res);
  req.on('close', () => state.removeSseClient(res));
});

// ---- Canvas lifecycle ----

// POST /api/canvas/new
router.post('/new', (req, res) => {
  const { width = 1200, height = 800 } = req.body || {};
  state.resetCanvas(Number(width), Number(height));
  res.json({ ok: true, canvas: state.getState().canvas });
});

// POST /api/canvas/clear
router.post('/clear', (req, res) => {
  state.clearObjects();
  res.json({ ok: true });
});

// GET /api/canvas/state
router.get('/state', (req, res) => {
  res.json(state.getState());
});

// ---- Image loading ----

// POST /api/canvas/load-image  (multipart OR JSON { url })
router.post('/load-image', upload.single('image'), async (req, res) => {
  try {
    let filename;

    if (req.file) {
      // Multipart file upload — add original extension
      const ext = path.extname(req.file.originalname) || '.png';
      filename = req.file.filename + ext;
      fs.renameSync(req.file.path, path.join(UPLOADS_DIR, filename));
    } else if (req.body && req.body.url) {
      const { url } = req.body;
      // Validate URL is http/https to prevent SSRF against internal services
      const parsed = new URL(url);
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        return res.status(400).json({ error: 'Only http/https URLs are allowed' });
      }
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Fetch failed: ${response.status}`);
      const buffer = Buffer.from(await response.arrayBuffer());
      filename = `url_${Date.now()}.png`;
      fs.writeFileSync(path.join(UPLOADS_DIR, filename), buffer);
    } else {
      return res.status(400).json({ error: 'Provide either an image file or a url field' });
    }

    // Read image dimensions
    let imgWidth = state.getState().canvas.width;
    let imgHeight = state.getState().canvas.height;
    try {
      const img = await loadImage(path.join(UPLOADS_DIR, filename));
      imgWidth = img.width;
      imgHeight = img.height;
    } catch (_) { /* keep canvas dims */ }

    state.setBackground(filename, imgWidth, imgHeight);
    res.json({ ok: true, filename, width: imgWidth, height: imgHeight });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/canvas/background
router.get('/background', (req, res) => {
  const bg = state.getState().canvas.backgroundImage;
  if (!bg) return res.status(404).json({ error: 'No background image' });
  const imgPath = path.join(UPLOADS_DIR, bg);
  if (!fs.existsSync(imgPath)) return res.status(404).json({ error: 'File not found' });
  res.sendFile(imgPath);
});

// ---- Crop background to region and reload ----

// POST /api/canvas/crop  { x, y, width, height, keepAnnotations? }
// Crops the current background image to the given canvas-coordinate bounding box,
// saves the result as a new upload, and loads it as the new background.
// Annotations that fall outside the crop are removed; inside ones are translated.
// POST /api/canvas/crop  { x, y, width, height, keepAnnotations?, shapeId? }
// Crops the current background image to the given canvas-coordinate bounding box.
// If shapeId is provided, the named annotation is used as a clipping mask — pixels
// outside the shape are transparent in the output (useful for ellipses, freehand, etc.).
// Annotations that fall outside the crop are removed; inside ones are translated.
router.post('/crop', async (req, res) => {
  try {
    const { keepAnnotations = true, shapeId } = req.body || {};
    let { x = 0, y = 0, width, height } = req.body || {};

    const s = state.getState();

    // If shapeId given, derive bounding box from that object
    if (shapeId) {
      const shapeObj = s.objects.find(o => o.id === shapeId);
      if (!shapeObj) return res.status(404).json({ error: `Shape '${shapeId}' not found` });
      const bb = _shapeBoundingBox(shapeObj);
      x = bb.x; y = bb.y; width = bb.width; height = bb.height;
    }

    if (!width || !height) {
      return res.status(400).json({ error: 'width and height are required (or provide shapeId)' });
    }

    const bg = s.canvas.backgroundImage;
    if (!bg) return res.status(400).json({ error: 'No background image loaded' });

    const srcPath = path.join(UPLOADS_DIR, bg);
    if (!fs.existsSync(srcPath)) return res.status(404).json({ error: 'Background file not found' });

    // Load source image
    const srcImg = await loadImage(srcPath);
    const srcW = srcImg.width;
    const srcH = srcImg.height;

    // Canvas coords → source image pixel coords
    const canvasW = s.canvas.width;
    const canvasH = s.canvas.height;
    const scaleX = srcW / canvasW;
    const scaleY = srcH / canvasH;

    const sx = Math.max(0, Math.round(x * scaleX));
    const sy = Math.max(0, Math.round(y * scaleY));
    const sw = Math.min(Math.round(width  * scaleX), srcW - sx);
    const sh = Math.min(Math.round(height * scaleY), srcH - sy);

    if (sw <= 0 || sh <= 0) {
      return res.status(400).json({ error: 'Crop region is outside or too small' });
    }

    // Save crop at native size (preserves correct coordinate mapping)
    const out = createCanvas(sw, sh);
    const ctx = out.getContext('2d');

    // If shapeId given, clip to shape before drawing (shape coords → output canvas coords)
    if (shapeId) {
      // Fill with canvas background colour so areas outside the clip match the UI background
      ctx.fillStyle = '#11111b';
      ctx.fillRect(0, 0, sw, sh);
      const shapeObj = s.objects.find(o => o.id === shapeId);
      _applyClipPath(ctx, shapeObj, x, y, scaleX, scaleY, sx, sy);
    }

    ctx.drawImage(srcImg, sx, sy, sw, sh, 0, 0, sw, sh);

    const filename = `crop_${Date.now()}.png`;
    const outPath = path.join(UPLOADS_DIR, filename);
    fs.writeFileSync(outPath, out.toBuffer('image/png'));

    // After crop the canvas is a new image — clear all annotations
    state.replaceObjects([]);

    // Set canvas to crop dimensions and reset viewport to 1:1
    state.setBackground(filename, sw, sh);
    state.resetViewport();

    res.json({ ok: true, filename, width: sw, height: sh });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

/** Return the axis-aligned bounding box of any shape object. */
function _shapeBoundingBox(obj) {
  switch (obj.type) {
    case 'rect':
      return { x: obj.x, y: obj.y, width: obj.width, height: obj.height };
    case 'ellipse':
      return { x: obj.cx - obj.rx, y: obj.cy - obj.ry,
               width: obj.rx * 2, height: obj.ry * 2 };
    case 'dot': {
      const r = obj.radius || obj.r || 4;
      return { x: obj.cx - r, y: obj.cy - r, width: r * 2, height: r * 2 };
    }
    case 'arrow': case 'line':
      return { x: Math.min(obj.x1, obj.x2), y: Math.min(obj.y1, obj.y2),
               width: Math.abs(obj.x2 - obj.x1), height: Math.abs(obj.y2 - obj.y1) };
    case 'freehand': {
      const l = obj.left || 0, t = obj.top || 0;
      const w = obj.width || 100, h = obj.height || 100;
      return { x: l, y: t, width: w, height: h };
    }
    default:
      return { x: obj.x || 0, y: obj.y || 0, width: obj.width || 100, height: obj.height || 100 };
  }
}

/**
 * Apply a canvas clipping path matching the given shape.
 * All coordinates are in source-image pixel space.
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} obj  — shape state object
 * @param {number} bx   — bounding-box x in canvas coords (top-left of crop region)
 * @param {number} by   — bounding-box y in canvas coords
 * @param {number} scaleX — canvas-coord to source-pixel x scale
 * @param {number} scaleY — canvas-coord to source-pixel y scale
 * @param {number} sx   — source x offset (sx pixel of source image = 0 in output canvas)
 * @param {number} sy   — source y offset
 */
function _applyClipPath(ctx, obj, bx, by, scaleX, scaleY, sx, sy) {
  // Convert a canvas coordinate to output-canvas pixel coordinate
  const toOutX = (cx) => Math.round(cx * scaleX) - sx;
  const toOutY = (cy) => Math.round(cy * scaleY) - sy;

  ctx.beginPath();
  switch (obj.type) {
    case 'ellipse': {
      const ecx = toOutX(obj.cx), ecy = toOutY(obj.cy);
      const erx = Math.round(obj.rx * scaleX), ery = Math.round(obj.ry * scaleY);
      ctx.ellipse(ecx, ecy, erx, ery, 0, 0, 2 * Math.PI);
      break;
    }
    case 'dot': {
      const r = obj.radius || obj.r || 4;
      ctx.arc(toOutX(obj.cx), toOutY(obj.cy), Math.round(r * Math.min(scaleX, scaleY)), 0, 2 * Math.PI);
      break;
    }
    case 'freehand': {
      // Fabric.js stores SVG-like path commands as an array of arrays
      const rawPath = obj.path;
      const oLeft = obj.left || 0, oTop = obj.top || 0;
      const oScaleX = obj.scaleX || 1, oScaleY = obj.scaleY || 1;
      if (Array.isArray(rawPath)) {
        for (const cmd of rawPath) {
          const [op, ...pts] = cmd;
          switch (op.toUpperCase()) {
            case 'M': ctx.moveTo(toOutX(oLeft + pts[0] * oScaleX), toOutY(oTop + pts[1] * oScaleY)); break;
            case 'L': ctx.lineTo(toOutX(oLeft + pts[0] * oScaleX), toOutY(oTop + pts[1] * oScaleY)); break;
            case 'Q': ctx.quadraticCurveTo(
              toOutX(oLeft + pts[0] * oScaleX), toOutY(oTop + pts[1] * oScaleY),
              toOutX(oLeft + pts[2] * oScaleX), toOutY(oTop + pts[3] * oScaleY)); break;
            case 'C': ctx.bezierCurveTo(
              toOutX(oLeft + pts[0] * oScaleX), toOutY(oTop + pts[1] * oScaleY),
              toOutX(oLeft + pts[2] * oScaleX), toOutY(oTop + pts[3] * oScaleY),
              toOutX(oLeft + pts[4] * oScaleX), toOutY(oTop + pts[5] * oScaleY)); break;
            case 'Z': ctx.closePath(); break;
          }
        }
      } else {
        // Fallback: use bounding box rect
        ctx.rect(toOutX(oLeft), toOutY(oTop),
                 Math.round((obj.width || 100) * oScaleX * scaleX),
                 Math.round((obj.height || 100) * oScaleY * scaleY));
      }
      break;
    }
    default: {
      // Fallback: rectangular clip from the shape's bounding box
      const bb = _shapeBoundingBox(obj);
      ctx.rect(toOutX(bb.x), toOutY(bb.y),
               Math.round(bb.width * scaleX), Math.round(bb.height * scaleY));
    }
  }
  ctx.closePath();
  ctx.clip();
}

/** Translate an annotation object by (dx, dy) and clip to (0,0,maxW,maxH).
 *  Returns null if the object is fully outside the region. */
function _translateObj(obj, dx, dy, maxW, maxH) {
  const o = { ...obj };
  switch (o.type) {
    case 'rect': {
      o.x += dx; o.y += dy;
      if (o.x + o.width  < 0 || o.x > maxW) return null;
      if (o.y + o.height < 0 || o.y > maxH) return null;
      return o;
    }
    case 'ellipse': {
      o.cx += dx; o.cy += dy;
      if (o.cx + o.rx < 0 || o.cx - o.rx > maxW) return null;
      if (o.cy + o.ry < 0 || o.cy - o.ry > maxH) return null;
      return o;
    }
    case 'dot': {
      o.cx += dx; o.cy += dy;
      if (o.cx < 0 || o.cx > maxW || o.cy < 0 || o.cy > maxH) return null;
      return o;
    }
    case 'arrow': case 'line': {
      o.x1 += dx; o.y1 += dy; o.x2 += dx; o.y2 += dy;
      // Keep if either endpoint is inside
      const inBounds = (px, py) => px >= 0 && px <= maxW && py >= 0 && py <= maxH;
      if (!inBounds(o.x1, o.y1) && !inBounds(o.x2, o.y2)) return null;
      return o;
    }
    case 'text': {
      o.x += dx; o.y += dy;
      if (o.x < -200 || o.x > maxW || o.y < -50 || o.y > maxH) return null;
      return o;
    }
    case 'freehand': {
      o.left = (o.left || 0) + dx;
      o.top  = (o.top  || 0) + dy;
      return o;
    }
    default:
      return o;
  }
}

// ---- Bulk operations ----

// POST /api/canvas/ops
router.post('/ops', (req, res) => {
  const { operations } = req.body || {};
  if (!Array.isArray(operations)) {
    return res.status(400).json({ error: 'operations must be an array' });
  }
  const results = operations.map((op) =>
    state.addObject({ ...op, createdBy: op.createdBy || 'model' }),
  );
  res.json({ ok: true, objects: results });
});

module.exports = router;
