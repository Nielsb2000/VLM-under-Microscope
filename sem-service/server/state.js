// server/state.js - single-session in-memory canvas state + SSE broadcast
const { v4: uuidv4 } = require('uuid');

let state = {
  canvas: { width: 1200, height: 800, backgroundImage: null },
  objects: [],
  viewport: { zoom: 1, panX: 0, panY: 0 },
  cursor: { x: null, y: null, visible: false, label: '' },
  filters: { brightness: 100, contrast: 100 },
  session: { filterAdjustments: 0, vlmSnapshots: 0, filterLog: [], vlmSnapshotLog: [] },
  uiMode: 'image',   // 'image' | 'grid' | 'atlas'
  atlas: null,        // { region, fw, tileWidth, tileHeight, cols, rows, atlasWidth, atlasHeight }
  atlasOverlay: {
    grid: false,
    labels: false,
    gridLevel: 0,
    subdivision: 1,
    gridLineWidth: 10,
    gridLineAlpha: 0.85,
    labelFontSize: 64,
    labelBoxPadding: 14,
    labelBoxAlpha: 0.70,
  }, // visual overlays in atlas mode
  segmentation: null, // { mask_png?, centroids?, bboxes? } - stored for server-side export
  segmentationEnabled: false, // true while the Segment panel is open in the UI
  segmentTextEnabled: true,   // when false, agent tool omits coordinate text output
  atlasCoordEnabled: true,    // when false, atlas_state skips coord computation - agent must use VLM
  tileGrid: {
    loaded: false,
    datasetName: null,
    currentRegion: null,
    currentFw: null,
    currentX: null,
    currentY: null,
    tileCount: 0,
    regions: [],
  },
};

// Saved per-mode snapshots - restored when switching back
let _savedImageState = null;  // { backgroundImage, width, height, objects }
let _savedGridTile   = null;  // { region, fw, x, y } - restored on switch back to grid
let _atlasAnnotations = {};   // key: "region:fw" -> objects[] - persists until region switch

const sseClients = new Set();

// ---- Readers ----

function getState() {
  return state;
}

function getObject(id) {
  return state.objects.find((o) => o.id === id) || null;
}

// ---- Writers ----

function resetCanvas(width = 1200, height = 800) {
  state = {
    canvas: { width, height, backgroundImage: null },
    objects: [],
    viewport: { zoom: 1, panX: 0, panY: 0 },
    cursor: { x: null, y: null, visible: false, label: '' },
    filters: { brightness: 100, contrast: 100 },
    session: { filterAdjustments: 0, vlmSnapshots: 0, filterLog: [], vlmSnapshotLog: [] },
    uiMode: 'image',
    atlas: null,
    atlasOverlay: {
      grid: false,
      labels: false,
      gridLevel: 0,
      subdivision: 1,
      gridLineWidth: 10,
      gridLineAlpha: 0.85,
      labelFontSize: 64,
      labelBoxPadding: 14,
      labelBoxAlpha: 0.70,
    },
    segmentation: null,
    segmentationEnabled: false,
    segmentTextEnabled: true,
    atlasCoordEnabled: true,
    tileGrid: {
      loaded: false,
      datasetName: null,
      currentRegion: null,
      currentFw: null,
      currentX: null,
      currentY: null,
      tileCount: 0,
      regions: [],
    },
  };
  broadcast();
}

function clearObjects() {
  state.objects = [];
  broadcast();
}

function replaceObjects(newObjects) {
  state.objects = newObjects;
  broadcast();
}

function setBackground(filename, width, height, { switchToImageMode = false } = {}) {
  if (width) state.canvas.width = width;
  if (height) state.canvas.height = height;
  state.canvas.backgroundImage = filename;
  if (switchToImageMode) state.uiMode = 'image';
  state.segmentation = null;
  broadcast();
}

function addObject(raw) {
  const obj = {
    ...raw,
    id: raw.id || `obj_${uuidv4().slice(0, 8)}`,
    createdAt: raw.createdAt || new Date().toISOString(),
    createdBy: raw.createdBy || 'model',
  };
  state.objects.push(obj);
  broadcast();
  return obj;
}

function addObjectsBatch(raws) {
  const created = raws.map(raw => ({
    ...raw,
    id: raw.id || `obj_${uuidv4().slice(0, 8)}`,
    createdAt: raw.createdAt || new Date().toISOString(),
    createdBy: raw.createdBy || 'model',
  }));
  state.objects.push(...created);
  broadcast();
  return created;
}

function updateObject(id, updates) {
  const idx = state.objects.findIndex((o) => o.id === id);
  if (idx === -1) return null;
  const { id: _id, createdAt: _ca, createdBy: _cb, ...safe } = updates;
  state.objects[idx] = { ...state.objects[idx], ...safe };
  broadcast();
  return state.objects[idx];
}

function deleteObject(id) {
  const before = state.objects.length;
  state.objects = state.objects.filter((o) => o.id !== id);
  if (state.objects.length === before) return false;
  broadcast();
  return true;
}

// ---- SSE ----

function addSseClient(res) {
  sseClients.add(res);
}

function removeSseClient(res) {
  sseClients.delete(res);
}

// ---- Viewport & cursor ----

function setViewport(zoom, panX, panY) {
  state.viewport = {
    zoom: Math.max(0.05, Math.min(zoom, 30)),
    panX, panY,
  };
  broadcast();
  return state.viewport;
}

function setCursor(x, y, visible, label = '') {
  state.cursor = { x, y, visible: !!visible, label };
  broadcast();
}

function zoomToRegion(rx, ry, rw, rh, padding = 60) {
  const W = state.canvas.width;
  const H = state.canvas.height;
  const zoom = Math.min(
    (W - padding * 2) / Math.max(rw, 1),
    (H - padding * 2) / Math.max(rh, 1),
    30,
  );
  const cx = rx + rw / 2;
  const cy = ry + rh / 2;
  const panX = cx * zoom - W / 2;
  const panY = cy * zoom - H / 2;
  state.viewport = { zoom, panX, panY };
  broadcast();
  return state.viewport;
}

function resetViewport() {
  state.viewport = { zoom: 1, panX: 0, panY: 0 };
  broadcast();
}

function setFilters({ brightness, contrast } = {}) {
  state.filters = {
    brightness: Math.max(0, Math.min(brightness ?? state.filters.brightness, 300)),
    contrast: Math.max(0, Math.min(contrast ?? state.filters.contrast, 300)),
  };
  state.session.filterAdjustments++;
  state.session.filterLog.push({
    t: new Date().toISOString(),
    brightness: state.filters.brightness,
    contrast: state.filters.contrast,
  });
  broadcast();
  return state.filters;
}

// ---- Session stats ----

function resetSession() {
  state.session = { filterAdjustments: 0, vlmSnapshots: 0, filterLog: [], vlmSnapshotLog: [] };
}

function incrementVlmSnapshots() {
  state.session.vlmSnapshots++;
  state.session.vlmSnapshotLog.push({ t: new Date().toISOString() });
}

function getSessionStats() {
  return { ...state.session };
}

// ---- Atlas mode ----

function enterAtlasMode(region, fw, atlasInfo) {
  if (state.uiMode === 'atlas' && state.atlas) {
    const key = `${state.atlas.region}:${state.atlas.fw}`;
    _atlasAnnotations[key] = [...state.objects];
  }
  state.uiMode = 'atlas';
  state.atlas = { region, fw, ...atlasInfo };
  state.canvas.backgroundImage = null;
  const key = `${region}:${fw}`;
  state.objects = _atlasAnnotations[key] ? [..._atlasAnnotations[key]] : [];
  broadcast();
}

function exitAtlasMode() {
  if (state.uiMode !== 'atlas') return;
  if (state.atlas) {
    const key = `${state.atlas.region}:${state.atlas.fw}`;
    _atlasAnnotations[key] = [...state.objects];
  }
  state.uiMode = 'grid';
  state.atlas = null;
}

function setViewportSilent(zoom, panX, panY) {
  state.viewport = {
    zoom: Math.max(0.001, Math.min(zoom, 50)),
    panX, panY,
  };
}

// ---- UI mode + tile grid state ----

function setUiMode(mode) {
  if (!['image', 'grid', 'atlas'].includes(mode)) throw new Error(`Invalid uiMode: ${mode}`);
  if (mode === state.uiMode) {
    broadcast();
    return;
  }

  if (state.uiMode === 'atlas' && state.atlas) {
    const key = `${state.atlas.region}:${state.atlas.fw}`;
    _atlasAnnotations[key] = [...state.objects];
    state.atlas = null;
  }

  if (mode === 'image') {
    if (state.tileGrid.loaded) {
      _savedGridTile = {
        region: state.tileGrid.currentRegion,
        fw: state.tileGrid.currentFw,
        x: state.tileGrid.currentX,
        y: state.tileGrid.currentY,
      };
    }
    if (_savedImageState) {
      state.canvas.backgroundImage = _savedImageState.backgroundImage;
      state.canvas.width = _savedImageState.width;
      state.canvas.height = _savedImageState.height;
      state.objects = _savedImageState.objects || [];
    }
    state.uiMode = 'image';
  } else if (mode === 'grid') {
    _savedImageState = {
      backgroundImage: state.canvas.backgroundImage,
      width: state.canvas.width,
      height: state.canvas.height,
      objects: [...state.objects],
    };
    state.uiMode = 'grid';
  } else {
    state.uiMode = 'atlas';
  }
  broadcast();
}

function getSavedGridTile() {
  return _savedGridTile;
}

function setUiModeOnly(mode) {
  if (mode !== 'image' && mode !== 'grid') throw new Error(`Invalid uiMode: ${mode}`);
  if (mode === 'grid' && state.tileGrid.loaded) {
    _savedGridTile = {
      region: state.tileGrid.currentRegion,
      fw: state.tileGrid.currentFw,
      x: state.tileGrid.currentX,
      y: state.tileGrid.currentY,
    };
  }
  state.uiMode = mode;
  broadcast();
}

function setSegmentation(data) {
  state.segmentation = data || null;
  broadcast();
}

function clearSegmentation() {
  if (state.segmentation === null) return;
  state.segmentation = null;
  broadcast();
}

function setSegmentationEnabled(enabled) {
  state.segmentationEnabled = !!enabled;
  if (!enabled) clearSegmentation();
}

function setSegmentTextEnabled(enabled) {
  state.segmentTextEnabled = !!enabled;
}

function setAtlasCoordEnabled(enabled) {
  state.atlasCoordEnabled = !!enabled;
}

// Set atlas overlay visual modes (grid lines, coordinate labels, and style).
function setAtlasOverlay({
  grid,
  labels,
  gridLevel,
  subdivision,
  gridLineWidth,
  gridLineAlpha,
  labelFontSize,
  labelBoxPadding,
  labelBoxAlpha,
} = {}) {
  const current = state.atlasOverlay || {};

  const nextGridLevel =
    gridLevel !== undefined
      ? Math.max(0, Math.floor(Number(gridLevel) || 0))
      : Math.max(0, Math.floor(Number(current.gridLevel) || 0));

  const inferredSubdivision = Math.pow(2, nextGridLevel);

  const nextSubdivision =
    subdivision !== undefined
      ? Math.max(1, Math.floor(Number(subdivision) || inferredSubdivision || 1))
      : Math.max(1, Math.floor(Number(current.subdivision) || inferredSubdivision || 1));

  state.atlasOverlay = {
    ...current,
    grid: grid !== undefined ? !!grid : !!current.grid,
    labels: labels !== undefined ? !!labels : !!current.labels,
    gridLevel: nextGridLevel,
    subdivision: nextSubdivision,
    gridLineWidth:
      gridLineWidth !== undefined
        ? Math.max(1, Number(gridLineWidth) || current.gridLineWidth || 10)
        : current.gridLineWidth || 10,
    gridLineAlpha:
      gridLineAlpha !== undefined
        ? Math.max(0, Math.min(1, Number(gridLineAlpha)))
        : (current.gridLineAlpha ?? 0.85),
    labelFontSize:
      labelFontSize !== undefined
        ? Math.max(8, Number(labelFontSize) || current.labelFontSize || 64)
        : current.labelFontSize || 64,
    labelBoxPadding:
      labelBoxPadding !== undefined
        ? Math.max(0, Number(labelBoxPadding) || current.labelBoxPadding || 14)
        : current.labelBoxPadding || 14,
    labelBoxAlpha:
      labelBoxAlpha !== undefined
        ? Math.max(0, Math.min(1, Number(labelBoxAlpha)))
        : (current.labelBoxAlpha ?? 0.70),
  };

  broadcast();
  return state.atlasOverlay;
}

function setTileGridState(patch) {
  state.tileGrid = { ...state.tileGrid, ...patch };
  broadcast();
}

function broadcast() {
  const data = JSON.stringify(state);
  for (const client of sseClients) {
    try {
      client.write(`data: ${data}\n\n`);
    } catch (_) {
      sseClients.delete(client);
    }
  }
}

module.exports = {
  getState,
  getObject,
  resetCanvas,
  clearObjects,
  replaceObjects,
  setBackground,
  addObject,
  updateObject,
  deleteObject,
  addSseClient,
  removeSseClient,
  setViewport,
  setViewportSilent,
  setCursor,
  zoomToRegion,
  resetViewport,
  setFilters,
  resetSession,
  incrementVlmSnapshots,
  getSessionStats,
  setUiMode,
  setUiModeOnly,
  setTileGridState,
  getSavedGridTile,
  enterAtlasMode,
  exitAtlasMode,
  setSegmentation,
  clearSegmentation,
  setSegmentationEnabled,
  setSegmentTextEnabled,
  setAtlasCoordEnabled,
  setAtlasOverlay,
  addObjectsBatch,
};