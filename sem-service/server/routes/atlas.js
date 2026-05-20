// server/routes/atlas.js — Atlas mode: stitch all tiles of a region/fw into a virtual explorable canvas
//
// GET  /api/atlas/manifest          { region?, fw? }  → tile grid manifest (dimensions + tile URLs)
// POST /api/atlas/enter             { region?, fw? }  → enter atlas mode (uses current grid region/fw)
// POST /api/atlas/exit                               → return to tile-grid mode
// POST /api/atlas/viewport          { zoom, panX, panY } → sync client viewport silently (for export)

'use strict';

const express   = require('express');
const router    = express.Router();
const sharp     = require('sharp');
const tileGrid  = require('../tileGrid');
const state     = require('../state');

// Compute atlas manifest (tile dimensions + grid shape) for a region/fw
async function _getManifest(region, fw) {
  const tiles = tileGrid.getTilesForRegion(region, fw);
  if (tiles.length === 0) return null;

  // Get pixel dimensions from first tile
  const meta = await sharp(tiles[0].absolutePath).metadata();
  const tileWidth  = meta.width  || 1920;
  const tileHeight = meta.height || 1200;

  let cols = 0, rows = 0;
  for (const t of tiles) {
    cols = Math.max(cols, t.x + 1);
    rows = Math.max(rows, t.y + 1);
  }

  return {
    region,
    fw,
    tileWidth,
    tileHeight,
    cols,
    rows,
    atlasWidth:  cols * tileWidth,
    atlasHeight: rows * tileHeight,
    tiles: tiles.map(t => ({ x: t.x, y: t.y, urlPath: t.urlPath })),
  };
}

// Load the tile that was active before atlas mode (called on atlas exit)
async function _reloadCurrentTile() {
  const s = state.getState();
  if (!s.tileGrid.loaded) return;
  const { currentRegion, currentFw, currentX, currentY } = s.tileGrid;
  const tile = tileGrid.getTile(currentRegion, currentFw, currentY, currentX);
  if (!tile) return;

  let imgWidth  = s.canvas.width  || 1920;
  let imgHeight = s.canvas.height || 1200;
  try {
    const meta = await sharp(tile.absolutePath).metadata();
    imgWidth  = meta.width  || imgWidth;
    imgHeight = meta.height || imgHeight;
  } catch (_) {}

  state.clearObjects();                                     // broadcasts (uiMode=grid already)
  state.setBackground(tile.urlPath, imgWidth, imgHeight);   // broadcasts
  state.resetViewport();                                    // broadcasts
}

// ---- GET /api/atlas/manifest?region=R&fw=F ----
router.get('/manifest', async (req, res) => {
  try {
    if (!tileGrid.isScanned()) {
      return res.status(400).json({ error: 'Tile dataset not yet scanned. Call POST /api/camera/init first.' });
    }
    const s = state.getState();
    const region = req.query.region != null ? parseInt(req.query.region, 10) : s.tileGrid.currentRegion;
    const fw     = req.query.fw     != null ? parseInt(req.query.fw,     10) : s.tileGrid.currentFw;
    if (region == null || isNaN(region)) return res.status(400).json({ error: 'region parameter required' });
    if (fw     == null || isNaN(fw))     return res.status(400).json({ error: 'fw parameter required' });

    const manifest = await _getManifest(region, fw);
    if (!manifest) return res.status(404).json({ error: `No tiles for Region${region} fw=${fw}um` });
    res.json(manifest);
  } catch (err) {
    console.error('[atlas/manifest]', err);
    res.status(500).json({ error: err.message });
  }
});

// ---- POST /api/atlas/enter  body: { region?, fw? } ----
router.post('/enter', async (req, res) => {
  try {
    if (!tileGrid.isScanned()) {
      return res.status(400).json({ error: 'Tile dataset not yet scanned. Call POST /api/camera/init first.' });
    }
    const s = state.getState();
    const region = req.body.region != null ? parseInt(req.body.region, 10) : s.tileGrid.currentRegion;
    const fw     = req.body.fw     != null ? parseInt(req.body.fw,     10) : s.tileGrid.currentFw;
    if (region == null || isNaN(region)) return res.status(400).json({ error: 'Must be in grid mode or provide region+fw' });
    if (fw     == null || isNaN(fw))     return res.status(400).json({ error: 'Must be in grid mode or provide fw' });

    const manifest = await _getManifest(region, fw);
    if (!manifest) return res.status(404).json({ error: `No tiles for Region${region} fw=${fw}um` });

    state.enterAtlasMode(region, fw, {
      tileWidth:   manifest.tileWidth,
      tileHeight:  manifest.tileHeight,
      cols:        manifest.cols,
      rows:        manifest.rows,
      atlasWidth:  manifest.atlasWidth,
      atlasHeight: manifest.atlasHeight,
    });

    res.json({ ok: true, ...manifest });
  } catch (err) {
    console.error('[atlas/enter]', err);
    res.status(500).json({ error: err.message });
  }
});

// ---- POST /api/atlas/exit ----
router.post('/exit', async (req, res) => {
  try {
    const s = state.getState();
    if (s.uiMode !== 'atlas') {
      return res.json({ ok: true, uiMode: s.uiMode });
    }
    state.exitAtlasMode();   // saves annotations, sets uiMode=grid, no broadcast
    await _reloadCurrentTile();
    res.json({ ok: true, uiMode: 'grid' });
  } catch (err) {
    console.error('[atlas/exit]', err);
    res.status(500).json({ error: err.message });
  }
});

// ---- POST /api/atlas/viewport  { zoom, panX, panY } ----
// Silent viewport sync from client — used so server-side export knows the current atlas view.
router.post('/viewport', (req, res) => {
  const { zoom, panX, panY } = req.body || {};
  if (zoom == null || panX == null || panY == null) {
    return res.status(400).json({ error: 'zoom, panX, panY required' });
  }
  state.setViewportSilent(parseFloat(zoom), parseFloat(panX), parseFloat(panY));
  res.json({ ok: true });
});

module.exports = router;
