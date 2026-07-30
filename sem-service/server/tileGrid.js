// server/tileGrid.js — tile index for grid navigation
//
// Supported filename patterns:
//   Combined_New_Scans_Andrea: Region{NNN}_y{YY}_x{XX}_fw{FW}um[_1|_2].tiff
//   Grid_Scan_Paper:           Tile_r{ROW}_c{COL}.tiff
// Multiple regions, multiple FOVs per region, sparse grids.
// Duplicate acquisitions (_1 suffix) exist; we prefer the base file.
//
// Grid_Scan_Paper tiles are assigned virtual region=999, fw=0.
// REGION_LABELS maps virtual region IDs to human-readable names.

'use strict';

const fs = require('fs');
const path = require('path');

// --- In-memory tile index ---
// Map key: `${region}:${fw}:${y},${x}` → TileInfo
let tileMap = new Map();
let scanned = false;
let datasetName = 'Combined_New_Scans_Andrea';

// Supported image extensions (lowercase)
const IMG_EXTS = new Set(['.tiff', '.tif', '.png', '.jpg', '.jpeg']);

// Human-readable labels for virtual region IDs
const REGION_LABELS = {
  999: 'Grid_paper',
};

// Parse Combined_New_Scans_Andrea filename: Region{NNN}_y{YY}_x{XX}_fw{FW}um[_N]
// Returns { region, y, x, fw, suffix } or null
function parseTileFilename(filename) {
  const stem = filename.replace(/\.[^.]+$/, ''); // strip extension
  // Match: Region011_y14_x03_fw120um  or  Region011_y14_x03_fw120um_1
  const m = stem.match(/^Region(\d+)_y(\d+)_x(\d+)_fw(\d+)um(?:_(\d+))?$/i);
  if (!m) return null;
  return {
    region:  parseInt(m[1], 10),
    y:       parseInt(m[2], 10),
    x:       parseInt(m[3], 10),
    fw:      parseInt(m[4], 10),
    suffix:  m[5] ? parseInt(m[5], 10) : 0,  // 0 = base (preferred), 1+ = duplicate
  };
}

// Parse Grid_Scan_Paper filename: Tile_r{ROW}_c{COL}.tiff
// Returns { region: 999, y: row, x: col, fw: 0, suffix: 0 } or null
function parsePaperTileFilename(filename) {
  const stem = filename.replace(/\.[^.]+$/, '');
  const m = stem.match(/^Tile_r(\d+)_c(\d+)$/i);
  if (!m) return null;
  return {
    region: 999,
    y:      parseInt(m[1], 10),
    x:      parseInt(m[2], 10),
    fw:     0,
    suffix: 0,
  };
}

function tileKey(region, fw, y, x) {
  return `${region}:${fw}:${y},${x}`;
}

// Scan one directory and add tiles to tileMap
// urlPrefix: the URL prefix used to serve tiles from this directory (e.g. '/tile-assets/')
function _scanDir(dir, split, urlPrefix) {
  if (!fs.existsSync(dir)) return;
  const prefix = urlPrefix || '/tile-assets/';
  const entries = fs.readdirSync(dir);
  for (const filename of entries) {
    const ext = path.extname(filename).toLowerCase();
    if (!IMG_EXTS.has(ext)) continue;

    // Try both filename parsers
    const parsed = parseTileFilename(filename) || parsePaperTileFilename(filename);
    if (!parsed) continue;

    const { region, y, x, fw, suffix } = parsed;
    const key = tileKey(region, fw, y, x);
    const existing = tileMap.get(key);

    // Prefer lower suffix (0 = no suffix = base acquisition)
    if (!existing || suffix < existing.suffix) {
      tileMap.set(key, {
        filename,
        absolutePath: path.join(dir, filename),
        urlPath: `${prefix}${filename}`,
        region, fw, y, x,
        suffix,
        split,
      });
    }
  }
}

// Scan all dataset directories, build index
// Each entry: { dir, split, urlPrefix? }
function scanDataset(dirs) {
  tileMap = new Map();
  for (const { dir, split, urlPrefix } of dirs) {
    _scanDir(dir, split, urlPrefix);
  }
  scanned = true;
  console.log(`[tileGrid] Scanned ${tileMap.size} tiles from ${dirs.map(d => d.split).join(', ')}`);
}

// Get TileInfo or null
function getTile(region, fw, y, x) {
  return tileMap.get(tileKey(region, fw, y, x)) || null;
}

// Get neighbour tile: dy/dx are deltas (+1 = right/down, -1 = left/up)
function getNeighbor(region, fw, y, x, dy, dx) {
  return getTile(region, fw, y + dy, x + dx);
}

// List all unique (region, fw) groups with tile counts, sorted.
// Each entry includes an optional `label` field for human-readable region names.
function listRegions() {
  const groups = new Map();
  for (const tile of tileMap.values()) {
    const k = `${tile.region}:${tile.fw}`;
    if (!groups.has(k)) {
      groups.set(k, { region: tile.region, fw: tile.fw, tileCount: 0 });
    }
    groups.get(k).tileCount++;
  }
  return Array.from(groups.values())
    .sort((a, b) => a.region !== b.region ? a.region - b.region : a.fw - b.fw)
    .map(g => ({ ...g, label: REGION_LABELS[g.region] || null }));
}

// Get the default starting tile: smallest (region asc, fw asc, y asc, x asc)
function getDefaultTile() {
  let best = null;
  for (const tile of tileMap.values()) {
    if (!best ||
        tile.region < best.region ||
        (tile.region === best.region && tile.fw < best.fw) ||
        (tile.region === best.region && tile.fw === best.fw && tile.y < best.y) ||
        (tile.region === best.region && tile.fw === best.fw && tile.y === best.y && tile.x < best.x)) {
      best = tile;
    }
  }
  return best;
}

// Get all tiles for a given (region, fw) pair, sorted row-first
function getTilesForRegion(region, fw) {
  const result = [];
  for (const tile of tileMap.values()) {
    if (tile.region === region && tile.fw === fw) result.push(tile);
  }
  return result.sort((a, b) => a.y !== b.y ? a.y - b.y : a.x - b.x);
}

// Get a tile for a given region+fw, defaulting to lowest y,x if not specified
function getFirstTileForRegion(region, fw) {
  let best = null;
  for (const tile of tileMap.values()) {
    if (tile.region !== region || tile.fw !== fw) continue;
    if (!best ||
        tile.y < best.y ||
        (tile.y === best.y && tile.x < best.x)) {
      best = tile;
    }
  }
  return best;
}

function getDatasetName() {
  return datasetName;
}

function isScanned() {
  return scanned;
}

function getTileCount() {
  return tileMap.size;
}

module.exports = {
  scanDataset,
  getTile,
  getNeighbor,
  listRegions,
  getDefaultTile,
  getFirstTileForRegion,
  getTilesForRegion,
  getDatasetName,
  isScanned,
  getTileCount,
  parseTileFilename,
  parsePaperTileFilename,
  REGION_LABELS,
};
