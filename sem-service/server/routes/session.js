// server/routes/session.js — case-study session statistics
'use strict';

const express = require('express');
const router  = express.Router();
const state   = require('../state');

// GET /api/session/stats
// Returns counters accumulated since the last Randomize call:
//   filterAdjustments — how many times the agent called POST /api/viewport/filters
//   vlmSnapshots      — how many times the agent called GET /api/export/png (get_canvas_image)
router.get('/stats', (req, res) => {
  const s = state.getState();
  res.json({
    ...state.getSessionStats(),
    currentFilters: s.filters,
  });
});

module.exports = router;
