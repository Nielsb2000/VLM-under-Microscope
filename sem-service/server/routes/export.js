// server/routes/export.js
const express = require('express');
const router = express.Router();
const state = require('../state');
const { renderToPng } = require('../renderer');

// GET /api/export/png
router.get('/png', async (req, res) => {
  try {
    const png = await renderToPng(state.getState());
    state.incrementVlmSnapshots();
    res.set('Content-Type', 'image/png');
    res.set('Content-Disposition', 'attachment; filename="annotated.png"');
    res.send(png);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/export/json
router.get('/json', (req, res) => {
  res.set('Content-Disposition', 'attachment; filename="annotations.json"');
  res.json(state.getState());
});

// POST /api/export/package — returns references to both exports
router.post('/package', async (req, res) => {
  try {
    const png = await renderToPng(state.getState());
    const json = state.getState();
    res.json({
      png: '/api/export/png',
      json: '/api/export/json',
      objectCount: json.objects.length,
      canvas: json.canvas,
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

module.exports = router;
