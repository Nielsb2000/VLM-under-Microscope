// server/routes/viewport.js — model-controlled viewport and cursor
const express = require('express');
const router = express.Router();
const state = require('../state');

// GET /api/viewport  — current viewport + cursor state
router.get('/', (req, res) => {
  const s = state.getState();
  res.json({ viewport: s.viewport, cursor: s.cursor });
});

// POST /api/viewport/zoom  { zoom, centerX, centerY }
// Sets zoom and optionally centers on a canvas coordinate.
router.post('/zoom', (req, res) => {
  const { zoom = 1, centerX, centerY } = req.body || {};
  const s = state.getState();
  let panX = s.viewport.panX;
  let panY = s.viewport.panY;
  if (centerX != null && centerY != null) {
    // Pan so that (centerX, centerY) stays centered after zoom change.
    const W = s.canvas.width;
    const H = s.canvas.height;
    panX = centerX * zoom - W / 2;
    panY = centerY * zoom - H / 2;
  }
  const vp = state.setViewport(zoom, panX, panY);
  res.json({ ok: true, viewport: vp });
});

// POST /api/viewport/pan  { x, y }
// Sets the pan so canvas point (x, y) is at the top-left of the viewport.
// Use relative: true + dx/dy for incremental moves.
router.post('/pan', (req, res) => {
  const { x, y, dx, dy, relative = false } = req.body || {};
  const s = state.getState();
  let panX = s.viewport.panX;
  let panY = s.viewport.panY;
  if (relative) {
    panX += (dx || 0);
    panY += (dy || 0);
  } else {
    panX = x != null ? x : panX;
    panY = y != null ? y : panY;
  }
  const vp = state.setViewport(s.viewport.zoom, panX, panY);
  res.json({ ok: true, viewport: vp });
});

// POST /api/viewport/zoom-to-region  { x, y, width, height, padding }
// Fit a canvas bounding box into the full view with optional padding.
router.post('/zoom-to-region', (req, res) => {
  const { x, y, width, height, padding = 60 } = req.body || {};
  if (x == null || y == null || width == null || height == null) {
    return res.status(400).json({ error: 'x, y, width, height required' });
  }
  const vp = state.zoomToRegion(x, y, width, height, padding);
  res.json({ ok: true, viewport: vp });
});

// POST /api/viewport/zoom-to-object  { id, padding }
// Convenience: look up an object by id and zoom to its bounding box.
router.post('/zoom-to-object', (req, res) => {
  const { id, padding = 60 } = req.body || {};
  const obj = state.getObject(id);
  if (!obj) return res.status(404).json({ error: 'Object not found' });

  let x, y, w, h;
  switch (obj.type) {
    case 'rect':
      ({ x, y } = obj); w = obj.width; h = obj.height; break;
    case 'ellipse':
      x = obj.cx - obj.rx; y = obj.cy - obj.ry; w = obj.rx * 2; h = obj.ry * 2; break;
    case 'dot':
      x = obj.cx - obj.radius; y = obj.cy - obj.radius; w = obj.radius * 2; h = obj.radius * 2; break;
    case 'arrow': case 'line':
      x = Math.min(obj.x1, obj.x2); y = Math.min(obj.y1, obj.y2);
      w = Math.abs(obj.x2 - obj.x1); h = Math.abs(obj.y2 - obj.y1); break;
    case 'text':
      x = obj.x; y = obj.y; w = (obj.text || '').length * (obj.fontSize || 16) * 0.6; h = obj.fontSize || 16; break;
    default:
      return res.status(400).json({ error: `Cannot compute bbox for type: ${obj.type}` });
  }

  const vp = state.zoomToRegion(x, y, w, h, padding);
  res.json({ ok: true, viewport: vp, bbox: { x, y, width: w, height: h } });
});

// POST /api/viewport/reset  — back to 1:1 zoom, origin pan
router.post('/reset', (req, res) => {
  state.resetViewport();
  res.json({ ok: true, viewport: state.getState().viewport });
});

// POST /api/viewport/cursor  { x, y, visible, label }
// Place/move the model cursor to canvas coordinates.
router.post('/cursor', (req, res) => {
  const { x = null, y = null, visible = true, label = '' } = req.body || {};
  state.setCursor(x, y, visible, label);
  res.json({ ok: true, cursor: state.getState().cursor });
});

// DELETE /api/viewport/cursor  — hide the cursor
router.delete('/cursor', (req, res) => {
  state.setCursor(null, null, false, '');
  res.json({ ok: true });
});

// POST /api/viewport/filters  { brightness, contrast, saturation }
// All values are percentages: 100 = normal, 0 = min, 200 = doubled.
// Omit a field to leave it unchanged.
router.post('/filters', (req, res) => {
  const { brightness, contrast, saturation } = req.body || {};
  const filters = state.setFilters({ brightness, contrast, saturation });
  res.json({ ok: true, filters });
});

// GET /api/viewport/filters
router.get('/filters', (req, res) => {
  res.json(state.getState().filters);
});

module.exports = router;
