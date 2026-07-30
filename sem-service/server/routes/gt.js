// server/routes/gt.js — write manual particle count annotations to the GT store
//
// POST /api/gt/annotate
//   Reads the number of objects currently on the canvas and the loaded image's
//   sample_id (extracted from canvas.backgroundImage), then upserts an entry
//   in the GT JSON file at GT_PATH.
//
//   Response: { ok, sample_id, gt_count, gt_path }
//   Errors:   { error: "..." }

'use strict';

const express = require('express');
const router  = express.Router();
const path    = require('path');
const fs      = require('fs');
const state   = require('../state');

// Mounted at /app/gt-data inside the container (see docker-compose.yml)
const GT_PATH = process.env.GT_PATH || '/app/gt-data/particle_gt.json';

function _loadStore() {
  try {
    if (fs.existsSync(GT_PATH)) {
      const raw = fs.readFileSync(GT_PATH, 'utf8').trim();
      if (raw) return JSON.parse(raw);
    }
  } catch (_) {}
  return {};
}

function _saveStore(store) {
  fs.mkdirSync(path.dirname(GT_PATH), { recursive: true });
  fs.writeFileSync(GT_PATH, JSON.stringify(store, null, 2) + '\n', 'utf8');
}

// Extract sample_id from backgroundImage path.
// e.g. "/dataset-assets/Particles/L2_abc123.jpg"  →  "L2_abc123"
function _extractSampleId(bgPath) {
  if (!bgPath) return null;
  return path.basename(bgPath, path.extname(bgPath));
}

// POST /api/gt/annotate
router.post('/annotate', (req, res) => {
  try {
    const s        = state.getState();
    const bgPath   = s.canvas.backgroundImage;
    const sampleId = _extractSampleId(bgPath);

    if (!sampleId) {
      return res.status(400).json({ error: 'No image loaded — load a particle image first.' });
    }

    const gtCount  = s.objects.length;
    const existing = _loadStore();
    const prev     = existing[sampleId] || null;

    existing[sampleId] = {
      gt_count:    gtCount,
      annotated_at: new Date().toISOString(),
      ...(prev?.notes ? { notes: prev.notes } : {}),
    };

    _saveStore(existing);

    console.log(`[gt] saved ${sampleId} → ${gtCount} particles${prev ? ` (was ${prev.gt_count})` : ''}`);
    res.json({ ok: true, sample_id: sampleId, gt_count: gtCount, gt_path: GT_PATH, prev });
  } catch (err) {
    console.error('[gt] annotate error:', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/gt  — return the full store (handy for debugging)
router.get('/', (req, res) => {
  try {
    res.json(_loadStore());
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
