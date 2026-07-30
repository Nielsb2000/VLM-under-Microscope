// server/routes/cs4_gt.js — CS4 pattern search GT management
//
// POST /api/cs4/save-pattern
//   Body: { sample_id?, image_data_url, tiles, notes? }
//   Saves pattern image to cs4-data/patterns/ and upserts a GT entry in
//   pattern_search_gt.json.
//
//   Response: { ok, sample_id, pattern_path, gt_tiles, updated }
//   Errors:   { error: "..." }

'use strict';

const express = require('express');
const router  = express.Router();
const path    = require('path');
const fs      = require('fs');

const CS4_DATA_DIR = process.env.CS4_DATA_DIR || '/app/cs4-data';
const GT_PATH      = path.join(CS4_DATA_DIR, 'ground_truth', 'pattern_search_gt.json');
const PATTERNS_DIR = path.join(CS4_DATA_DIR, 'patterns');

function _loadGt() {
  try {
    if (fs.existsSync(GT_PATH)) {
      const raw = fs.readFileSync(GT_PATH, 'utf8').trim();
      if (raw) return JSON.parse(raw);
    }
  } catch (_) {}
  return { dataset_name: 'case_study_4_pattern_search_gt', updated_at: null, entries: [] };
}

function _saveGt(store) {
  store.updated_at = new Date().toISOString();
  fs.mkdirSync(path.dirname(GT_PATH), { recursive: true });
  fs.writeFileSync(GT_PATH, JSON.stringify(store, null, 2) + '\n', 'utf8');
}

// Parse "(2,3), (2,4)" → ["(2,3)", "(2,4)"]
function _parseTiles(tilesStr) {
  if (!tilesStr || !tilesStr.trim()) return [];
  const matches = tilesStr.match(/\(\s*\d+\s*,\s*\d+\s*\)/g) || [];
  return matches.map(t => t.replace(/\s/g, ''));
}

// POST /api/cs4/save-pattern
router.post('/save-pattern', (req, res) => {
  try {
    const { sample_id: rawId, image_data_url, source_region, tiles: tilesInput, notes } = req.body;

    if (!image_data_url) {
      return res.status(400).json({ error: 'image_data_url is required' });
    }

    // Strip data URI prefix and decode
    const b64Match = image_data_url.match(/^data:image\/\w+;base64,(.+)$/s);
    if (!b64Match) {
      return res.status(400).json({ error: 'image_data_url must be a valid data URI (data:image/...;base64,...)' });
    }
    const imageBuffer = Buffer.from(b64Match[1], 'base64');

    // Determine sample ID — fallback to timestamp if empty
    const sampleId = (rawId || '').trim() || `pattern_${Date.now()}`;
    const filename = `${sampleId}.png`;
    const absPath  = path.join(PATTERNS_DIR, filename);
    const relPath  = `data/case_study_4/patterns/${filename}`;

    // Write image
    fs.mkdirSync(PATTERNS_DIR, { recursive: true });
    fs.writeFileSync(absPath, imageBuffer);

    // Parse tile coordinates
    const gtTiles = _parseTiles(tilesInput || '');

    // Upsert GT entry
    const store   = _loadGt();
    const idx     = store.entries.findIndex(e => e.sample_id === sampleId);
    const isUpdate = idx >= 0;

    const entry = {
      sample_id:            sampleId,
      target_pattern_image: relPath,
      target_present:       true,
      gt_tiles:             gtTiles,
      source_region:        source_region?.trim() || null,
      notes:                notes?.trim() || null,
    };

    if (isUpdate) {
      store.entries[idx] = entry;
    } else {
      store.entries.push(entry);
    }
    _saveGt(store);

    console.log(`[cs4_gt] ${isUpdate ? 'updated' : 'added'} pattern "${sampleId}" → ${relPath} tiles=[${gtTiles.join(', ') || 'none'}]`);
    res.json({ ok: true, sample_id: sampleId, pattern_path: relPath, gt_tiles: gtTiles, updated: isUpdate });
  } catch (err) {
    console.error('[cs4_gt] save-pattern error:', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/cs4/patterns — list all saved pattern entries
router.get('/patterns', (req, res) => {
  try {
    res.json(_loadGt());
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
