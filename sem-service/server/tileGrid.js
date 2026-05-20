// server/tileGrid.js — tile index for Combined_New_Scans_Andrea grid navigation
//
// Filename pattern: Region{NNN}_y{YY}_x{XX}_fw{FW}um[_1|_2].tiff
// Multiple regions, multiple FOVs per region, sparse grids.
// Duplicate acquisitions (_1 suffix) exist; we prefer the base file.

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

// Parse filename stem: returns { region, y, x, fw, suffix } or null
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

function tileKey(region, fw, y, x) {
  return `${region}:${fw}:${y},${x}`;
}

// Scan one directory and add tiles to tileMap
function _scanDir(dir, split) {
  if (!fs.existsSync(dir)) return;
  const entries = fs.readdirSync(dir);
  for (const filename of entries) {
    const ext = path.extname(filename).toLowerCase();
    if (!IMG_EXTS.has(ext)) continue;

    const parsed = parseTileFilename(filename);
    if (!parsed) continue;

    const { region, y, x, fw, suffix } = parsed;
    const key = tileKey(region, fw, y, x);
    const existing = tileMap.get(key);

    // Prefer lower suffix (0 = no suffix = base acquisition)
    if (!existing || suffix < existing.suffix) {
      tileMap.set(key, {
        filename,
        absolutePath: path.join(dir, filename),
        // URL path under /tile-assets/ for browser access
        urlPath: `/tile-assets/${filename}`,
        region, fw, y, x,
        suffix,
        split,
      });
    }
  }
}

// Scan all dataset directories, build index
function scanDataset(dirs) {
  tileMap = new Map();
  for (const { dir, split } of dirs) {
    _scanDir(dir, split);
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

// List all unique (region, fw) groups with tile counts, sorted
function listRegions() {
  const groups = new Map();
  for (const tile of tileMap.values()) {
    const k = `${tile.region}:${tile.fw}`;
    if (!groups.has(k)) {
      groups.set(k, { region: tile.region, fw: tile.fw, tileCount: 0 });
    }
    groups.get(k).tileCount++;
  }
  return Array.from(groups.values()).sort((a, b) =>
    a.region !== b.region ? a.region - b.region : a.fw - b.fw
  );
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
};
