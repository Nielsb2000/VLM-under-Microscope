// server/routes/draw.js — one endpoint per shape type
const express = require('express');
const router = express.Router();
const state = require('../state');

function makeDrawHandler(type, required = []) {
  return (req, res) => {
    const body = req.body || {};
    for (const field of required) {
      if (body[field] == null) {
        return res.status(400).json({ error: `Missing required field: ${field}` });
      }
    }
    const obj = state.addObject({ ...body, type, createdBy: body.createdBy || 'model' });
    res.status(201).json(obj);
  };
}

// POST /api/draw/rect  { x, y, width, height, stroke, fill, strokeWidth, label, createdBy }
router.post('/rect', makeDrawHandler('rect', ['x', 'y', 'width', 'height']));

// POST /api/draw/ellipse  { cx, cy, rx, ry, stroke, fill, strokeWidth, label, createdBy }
router.post('/ellipse', makeDrawHandler('ellipse', ['cx', 'cy', 'rx', 'ry']));

// POST /api/draw/arrow  { x1, y1, x2, y2, stroke, strokeWidth, label, createdBy }
router.post('/arrow', makeDrawHandler('arrow', ['x1', 'y1', 'x2', 'y2']));

// POST /api/draw/dot  { cx, cy, radius, fill, stroke, strokeWidth, label, createdBy }
router.post('/dot', makeDrawHandler('dot', ['cx', 'cy']));

// POST /api/draw/line  { x1, y1, x2, y2, stroke, strokeWidth, label, createdBy }
router.post('/line', makeDrawHandler('line', ['x1', 'y1', 'x2', 'y2']));

// POST /api/draw/text  { x, y, text, fontSize, fill, fontFamily, label, createdBy }
router.post('/text', makeDrawHandler('text', ['x', 'y', 'text']));

// POST /api/draw/freehand  { path, left, top, stroke, strokeWidth, label, createdBy }
router.post('/freehand', makeDrawHandler('freehand', ['path']));

module.exports = router;
