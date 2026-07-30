// server/routes/camera.js — tile-grid camera navigation endpoints
//
// POST /api/camera/init           load tile dataset + enter grid mode
// GET  /api/camera/state          get current tile position + mode
// GET  /api/camera/regions        list all available regions (works in any mode)
// POST /api/camera/move           move camera in a direction (left/right/up/down)
// POST /api/camera/goto           jump to explicit (x, y, region?, fw?)
// POST /api/camera/mode           switch uiMode ("image" | "grid")

'use strict';

const path = require('path');
const router = require('express').Router();
const sharp = require('sharp');
const tileGrid = require('../tileGrid');
const state = require('../state');

// Dataset directory inside the container (mounted via Docker volume)
const TILE_DIRS = [
  { dir: '/app/tile-datasets',    split: 'tiles', urlPrefix: '/tile-assets/' },
  { dir: '/app/grid-paper-tiles', split: 'paper', urlPrefix: '/grid-paper-assets/' },
];

// Ensure dataset is scanned; lazy-init on first call to /init
function _ensureScanned() {
  if (!tileGrid.isScanned()) {
    tileGrid.scanDataset(TILE_DIRS);
  }
}

// Load a tile's image file as the canvas background (clear annotations + reset viewport)
async function _loadTile(tile) {
  let imgWidth = state.getState().canvas.width;
  let imgHeight = state.getState().canvas.height;
  try {
    const meta = await sharp(tile.absolutePath).metadata();
    imgWidth = meta.width || imgWidth;
    imgHeight = meta.height || imgHeight;
  } catch (_) {}

  state.clearObjects();
  state.setBackground(tile.urlPath, imgWidth, imgHeight);
  state.resetViewport();
}

// ---- POST /api/camera/init ----
// Body: { region?: number, fw?: number }
router.post('/init', async (req, res) => {
  try {
    _ensureScanned();

    if (tileGrid.getTileCount() === 0) {
      return res.status(500).json({ error: 'No tiles found. Check that the dataset volumes are mounted correctly.' });
    }

    const { region, fw } = req.body || {};
    let tile;

    if (region != null && fw != null) {
      tile = tileGrid.getFirstTileForRegion(parseInt(region, 10), parseInt(fw, 10));
      if (!tile) {
        return res.status(404).json({
          error: `No tiles found for Region ${region} fw=${fw}um`,
          availableRegions: tileGrid.listRegions(),
        });
      }
    } else {
      tile = tileGrid.getDefaultTile();
    }

    // Update state first so every subsequent broadcast (clearObjects/setBackground/resetViewport) has the correct position
    state.setUiMode('grid');
    state.setTileGridState({
      loaded: true,
      datasetName: tileGrid.getDatasetName(),
      currentRegion: tile.region,
      currentFw: tile.fw,
      currentX: tile.x,
      currentY: tile.y,
      tileCount: tileGrid.getTileCount(),
      regions: tileGrid.listRegions(),
    });
    await _loadTile(tile);

    return res.json({
      ok: true,
      tile: { region: tile.region, fw: tile.fw, x: tile.x, y: tile.y, filename: tile.filename },
      tileCount: tileGrid.getTileCount(),
      regions: tileGrid.listRegions(),
    });
  } catch (err) {
    console.error('[camera/init]', err);
    return res.status(500).json({ error: err.message });
  }
});

// ---- GET /api/camera/regions ----
// Returns the full region list from the tile dataset.
// Scans the dataset if needed — works in any UI mode (image, grid, atlas).
router.get('/regions', (req, res) => {
  try {
    _ensureScanned();
    return res.json({ regions: tileGrid.listRegions() });
  } catch (err) {
    console.error('[camera/regions]', err);
    return res.status(500).json({ error: err.message });
  }
});

// ---- GET /api/camera/tiles?region=X&fw=Y ----
// Returns all (x,y) tile coordinates for a given region+fw pair.
router.get('/tiles', (req, res) => {
  try {
    _ensureScanned();
    const region = parseInt(req.query.region, 10);
    const fw     = parseInt(req.query.fw,     10);
    if (isNaN(region) || isNaN(fw)) {
      return res.status(400).json({ error: 'region and fw query params are required' });
    }
    const tiles = tileGrid.getTilesForRegion(region, fw).map(t => ({ x: t.x, y: t.y }));
    return res.json({ region, fw, tiles });
  } catch (err) {
    console.error('[camera/tiles]', err);
    return res.status(500).json({ error: err.message });
  }
});

// ---- GET /api/camera/tile-image?region=X&fw=Y&x=A&y=B ----
// Returns the raw tile image file for the given position.
router.get('/tile-image', (req, res) => {
  try {
    _ensureScanned();
    const region = parseInt(req.query.region, 10);
    const fw     = parseInt(req.query.fw,     10);
    const x      = parseInt(req.query.x,      10);
    const y      = parseInt(req.query.y,      10);
    if ([region, fw, x, y].some(isNaN)) {
      return res.status(400).json({ error: 'region, fw, x, y query params are required' });
    }
    const tile = tileGrid.getTile(region, fw, y, x);
    if (!tile) {
      return res.status(404).json({ error: `Tile not found: region=${region} fw=${fw} x=${x} y=${y}` });
    }
    return res.sendFile(tile.absolutePath);
  } catch (err) {
    console.error('[camera/tile-image]', err);
    return res.status(500).json({ error: err.message });
  }
});

// ---- GET /api/camera/state ----
router.get('/state', (req, res) => {
  const s = state.getState();
  return res.json({
    uiMode:  s.uiMode,
    tileGrid: s.tileGrid,
    atlas:   s.atlas,
  });
});

// ---- POST /api/camera/move ----
// Body: { direction: "left" | "right" | "up" | "down" }
router.post('/move', async (req, res) => {
  const { direction } = req.body || {};
  if (!['left', 'right', 'up', 'down'].includes(direction)) {
    return res.status(400).json({ error: 'direction must be one of: left, right, up, down' });
  }

  const s = state.getState();
  if (s.uiMode !== 'grid' || !s.tileGrid.loaded) {
    return res.status(400).json({ error: 'Not in grid mode. Call POST /api/camera/init first.' });
  }

  const { currentRegion, currentFw, currentX, currentY } = s.tileGrid;
  const deltas = { left: [0, -1], right: [0, 1], up: [-1, 0], down: [1, 0] };
  const [dy, dx] = deltas[direction];

  const neighbor = tileGrid.getNeighbor(currentRegion, currentFw, currentY, currentX, dy, dx);
  if (!neighbor) {
    return res.status(404).json({
      error: `No tile at (x=${currentX + dx}, y=${currentY + dy}) — boundary reached or tile missing.`,
      currentTile: { region: currentRegion, fw: currentFw, x: currentX, y: currentY },
    });
  }

  try {
    // Update coords first so every subsequent broadcast (from clearObjects/setBackground/resetViewport) has the correct position
    state.setTileGridState({ currentX: neighbor.x, currentY: neighbor.y });
    await _loadTile(neighbor);
    return res.json({
      ok: true,
      tile: { region: neighbor.region, fw: neighbor.fw, x: neighbor.x, y: neighbor.y, filename: neighbor.filename },
    });
  } catch (err) {
    console.error('[camera/move]', err);
    return res.status(500).json({ error: err.message });
  }
});

// ---- POST /api/camera/goto ----
// Body: { x: number, y: number, region?: number, fw?: number }
router.post('/goto', async (req, res) => {
  const { x, y, region, fw } = req.body || {};
  if (x == null || y == null) {
    return res.status(400).json({ error: 'goto requires x and y' });
  }

  _ensureScanned();

  const s = state.getState();
  const targetRegion = region != null ? parseInt(region, 10) : s.tileGrid.currentRegion;
  const targetFw     = fw     != null ? parseInt(fw,     10) : s.tileGrid.currentFw;

  if (targetRegion == null || targetFw == null) {
    return res.status(400).json({ error: 'No active region/fw. Call POST /api/camera/init first, or provide region and fw.' });
  }

  const tile = tileGrid.getTile(targetRegion, targetFw, parseInt(y, 10), parseInt(x, 10));
  if (!tile) {
    return res.status(404).json({
      error: `No tile at Region${targetRegion} fw=${targetFw}um x=${x} y=${y}`,
      currentTile: s.tileGrid.loaded
        ? { region: s.tileGrid.currentRegion, fw: s.tileGrid.currentFw, x: s.tileGrid.currentX, y: s.tileGrid.currentY }
        : null,
    });
  }

  try {
    // Update state first so every subsequent broadcast has the correct position
    state.setUiMode('grid');
    state.setTileGridState({
      loaded: true,
      datasetName: tileGrid.getDatasetName(),
      currentRegion: tile.region,
      currentFw: tile.fw,
      currentX: tile.x,
      currentY: tile.y,
      tileCount: tileGrid.getTileCount(),
      regions: tileGrid.listRegions(),
    });
    await _loadTile(tile);

    return res.json({
      ok: true,
      tile: { region: tile.region, fw: tile.fw, x: tile.x, y: tile.y, filename: tile.filename },
    });
  } catch (err) {
    console.error('[camera/goto]', err);
    return res.status(500).json({ error: err.message });
  }
});

// ---- POST /api/camera/mode ----
// Body: { mode: "image" | "grid" }
router.post('/mode', async (req, res) => {
  const { mode } = req.body || {};
  if (mode !== 'image' && mode !== 'grid') {
    return res.status(400).json({ error: 'mode must be "image" or "grid"' });
  }
  try {
    if (mode === 'grid') {
      // Ensure dataset is scanned
      _ensureScanned();
      if (tileGrid.getTileCount() === 0) {
        return res.status(500).json({ error: 'No tiles found. Call POST /api/camera/init first.' });
      }
      // Restore saved grid position, or fall back to default
      const saved = state.getSavedGridTile();
      const s     = state.getState();
      let tile;
      if (saved) {
        tile = tileGrid.getTile(saved.region, saved.fw, saved.y, saved.x) || tileGrid.getDefaultTile();
      } else if (s.tileGrid.loaded) {
        tile = tileGrid.getTile(s.tileGrid.currentRegion, s.tileGrid.currentFw, s.tileGrid.currentY, s.tileGrid.currentX) || tileGrid.getDefaultTile();
      } else {
        tile = tileGrid.getDefaultTile();
      }
      // Update state first, then load tile
      state.setUiMode('grid');
      state.setTileGridState({
        loaded: true,
        datasetName: tileGrid.getDatasetName(),
        currentRegion: tile.region,
        currentFw: tile.fw,
        currentX: tile.x,
        currentY: tile.y,
        tileCount: tileGrid.getTileCount(),
        regions: tileGrid.listRegions(),
      });
      await _loadTile(tile);
      return res.json({ ok: true, uiMode: 'grid', tile: { region: tile.region, fw: tile.fw, x: tile.x, y: tile.y, filename: tile.filename } });
    } else {
      state.setUiMode('image');
      return res.json({ ok: true, uiMode: 'image' });
    }
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

// ---- GET /api/camera/neighbors ----
// Returns a 3x3 grid of neighbor tiles centered on the current position
router.get('/neighbors', (req, res) => {
  const s = state.getState();
  if (!s.tileGrid.loaded) {
    return res.status(400).json({ error: 'Not in grid mode.' });
  }
  const { currentRegion, currentFw, currentX, currentY } = s.tileGrid;
  const grid = [];
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const tile = tileGrid.getTile(currentRegion, currentFw, currentY + dy, currentX + dx);
      grid.push(tile
        ? { exists: true, x: tile.x, y: tile.y, urlPath: tile.urlPath, filename: tile.filename, current: dx === 0 && dy === 0 }
        : { exists: false, x: currentX + dx, y: currentY + dy, current: dx === 0 && dy === 0 }
      );
    }
  }
  return res.json({ grid, region: currentRegion, fw: currentFw });
});

module.exports = router;
