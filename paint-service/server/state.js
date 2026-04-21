// server/state.js — single-session in-memory canvas state + SSE broadcast
const { v4: uuidv4 } = require('uuid');

let state = {
  canvas: { width: 1200, height: 800, backgroundImage: null },
  objects: [],
  viewport: { zoom: 1, panX: 0, panY: 0 },
  cursor: { x: null, y: null, visible: false, label: '' },
  filters: { brightness: 100, contrast: 100, saturation: 100 },
};

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

function setBackground(filename, width, height) {
  if (width) state.canvas.width = width;
  if (height) state.canvas.height = height;
  state.canvas.backgroundImage = filename;
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

function updateObject(id, updates) {
  const idx = state.objects.findIndex((o) => o.id === id);
  if (idx === -1) return null;
  // Never allow overwriting id / timestamps
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

// Compute viewport so a bounding box fills the display with optional padding.
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

function setFilters({ brightness, contrast, saturation } = {}) {
  state.filters = {
    brightness: Math.max(0, Math.min(brightness ?? state.filters.brightness, 300)),
    contrast:   Math.max(0, Math.min(contrast   ?? state.filters.contrast,   300)),
    saturation: Math.max(0, Math.min(saturation ?? state.filters.saturation, 300)),
  };
  broadcast();
  return state.filters;
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
  setCursor,
  zoomToRegion,
  resetViewport,
  setFilters,
};
