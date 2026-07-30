// server/routes/atlas.js - Atlas mode: stitch all tiles of a region/fw into a virtual explorable canvas
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
// Silent viewport sync from client - used so server-side export knows the current atlas view.
router.post('/viewport', (req, res) => {
  const { zoom, panX, panY } = req.body || {};
  if (zoom == null || panX == null || panY == null) {
    return res.status(400).json({ error: 'zoom, panX, panY required' });
  }
  state.setViewportSilent(parseFloat(zoom), parseFloat(panX), parseFloat(panY));
  res.json({ ok: true });
});

// ---- POST /api/atlas/overlay  { grid?, labels?, gridLevel?, subdivision?, gridLineWidth?, gridLineAlpha?, labelFontSize?, labelBoxPadding?, labelBoxAlpha? } ----
// Toggle grid overlay and/or analysis-grid coordinate labels in atlas mode.
// gridLevel/subdivision are consumed by server-side PNG export rendering:
//   L0 => subdivision 1, original SEM acquisition-tile grid
//   L1 => subdivision 2, each original tile split into 2 x 2 cells
//   L2 => subdivision 4, each original tile split into 4 x 4 cells
// Style fields are optional and are consumed by server-side PNG export rendering.
router.post('/overlay', (req, res) => {
  const s = state.getState();
  if (s.uiMode !== 'atlas') {
    return res.status(400).json({ error: 'Not in atlas mode' });
  }

  const {
    grid,
    labels,
    gridLevel,
    subdivision,
    gridLineWidth,
    gridLineAlpha,
    labelFontSize,
    labelBoxPadding,
    labelBoxAlpha,
  } = req.body || {};

  if (
    grid === undefined &&
    labels === undefined &&
    gridLevel === undefined &&
    subdivision === undefined &&
    gridLineWidth === undefined &&
    gridLineAlpha === undefined &&
    labelFontSize === undefined &&
    labelBoxPadding === undefined &&
    labelBoxAlpha === undefined
  ) {
    return res.status(400).json({ error: 'At least one atlas overlay field required' });
  }

  const overlay = state.setAtlasOverlay({
    grid,
    labels,
    gridLevel,
    subdivision,
    gridLineWidth,
    gridLineAlpha,
    labelFontSize,
    labelBoxPadding,
    labelBoxAlpha,
  });
  res.json({ ok: true, atlasOverlay: overlay });
});

// ---- GET /api/atlas/list-regions ----
// Returns all available (region, fw) pairs with tile counts and labels.
router.get('/list-regions', (req, res) => {
  if (!tileGrid.isScanned()) {
    return res.status(400).json({ error: 'Tile dataset not yet scanned.' });
  }
  res.json(tileGrid.listRegions());
});

// ---- GET /api/atlas/stitch?region=R&fw=F&maxWidth=2048 ----
// Stitches all tiles for a (region, fw) pair into a single PNG and returns it
// as base64.  The atlas is downscaled so its width does not exceed maxWidth.
router.get('/stitch', async (req, res) => {
  if (!tileGrid.isScanned()) {
    return res.status(400).json({ error: 'Tile dataset not yet scanned.' });
  }
  const region   = parseInt(req.query.region,   10);
  const fw       = parseInt(req.query.fw,       10);
  const maxWidth = parseInt(req.query.maxWidth || '2048', 10);
  if (isNaN(region) || isNaN(fw)) {
    return res.status(400).json({ error: 'region and fw are required' });
  }

  try {
    const tiles = tileGrid.getTilesForRegion(region, fw);
    if (tiles.length === 0) {
      return res.status(404).json({ error: `No tiles for Region${region} fw=${fw}` });
    }

    // Use the first tile to get canonical tile dimensions
    const firstMeta = await sharp(tiles[0].absolutePath).metadata();
    const tileW = firstMeta.width  || 1920;
    const tileH = firstMeta.height || 1200;

    let cols = 0, rows = 0;
    for (const t of tiles) {
      cols = Math.max(cols, t.x + 1);
      rows = Math.max(rows, t.y + 1);
    }
    const atlasW = cols * tileW;
    const atlasH = rows * tileH;

    // Compute scale so the output width <= maxWidth
    const scale  = Math.min(1, maxWidth / atlasW);
    const outW   = Math.max(1, Math.round(atlasW * scale));
    const outH   = Math.max(1, Math.round(atlasH * scale));
    const cellW  = Math.max(1, Math.round(tileW  * scale));
    const cellH  = Math.max(1, Math.round(tileH  * scale));

    // Build composite list: resize each tile then place at grid position
    const composites = await Promise.all(tiles.map(async t => {
      const buf = await sharp(t.absolutePath)
        .resize(cellW, cellH, { fit: 'fill' })
        .toBuffer();
      return {
        input: buf,
        left:  Math.round(t.x * cellW),
        top:   Math.round(t.y * cellH),
      };
    }));

    const png = await sharp({
      create: { width: outW, height: outH, channels: 3, background: { r: 0, g: 0, b: 0 } },
    })
      .composite(composites)
      .png({ compressionLevel: 6 })
      .toBuffer();

    res.json({ ok: true, imageBase64: png.toString('base64'), width: outW, height: outH, region, fw });
  } catch (err) {
    console.error('[atlas/stitch]', err);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
