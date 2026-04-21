// client/app.js
'use strict';

const API = '/api';

let fc;            // fabric.Canvas instance
let currentTool = 'select';
let isApplyingState = false;   // prevent re-entrant SSE cycles
let drawStart = null;
let previewObj = null;

// ---- Bootstrap ----

document.addEventListener('DOMContentLoaded', async () => {
  fc = new fabric.Canvas('main-canvas', {
    selection: true,
    preserveObjectStacking: true,
    backgroundColor: null,
    stopContextMenu: true,
  });

  _resizeCanvas(1200, 800);
  _setupToolbar();
  _setupCanvasEvents();
  _setupFileUpload();
  _setupKeyboard();
  _setupSSE();

  // Fetch initial state
  const s = await fetch(`${API}/canvas/state`).then(r => r.json());
  await _applyState(s);
});

// ---- Tool bar wiring ----

function _setupToolbar() {
  ['select', 'rect', 'ellipse', 'arrow', 'dot', 'pen', 'text'].forEach(t => {
    document.getElementById(`tool-${t}`)?.addEventListener('click', () => setTool(t));
  });
  document.getElementById('btn-delete')?.addEventListener('click', deleteSelected);
  document.getElementById('btn-clear')?.addEventListener('click', clearCanvas);
  document.getElementById('btn-export-png')?.addEventListener('click', exportPng);
  document.getElementById('btn-export-json')?.addEventListener('click', exportJson);
}

function setTool(tool) {
  currentTool = tool;
  document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`tool-${tool}`)?.classList.add('active');

  if (tool === 'pen') {
    fc.isDrawingMode = true;
    fc.freeDrawingBrush.color = _color();
    fc.freeDrawingBrush.width = _strokeWidth();
    fc.selection = false;
  } else {
    fc.isDrawingMode = false;
    fc.selection = tool === 'select';
    fc.forEachObject(o => {
      o.selectable = tool === 'select';
      o.evented = tool === 'select';
    });
    fc.discardActiveObject();
    fc.requestRenderAll();
  }
}

// ---- Canvas mouse events ----

function _setupCanvasEvents() {
  fc.on('mouse:down', _onDown);
  fc.on('mouse:move', _onMove);
  fc.on('mouse:up', _onUp);
  fc.on('path:created', _onPathCreated);
  fc.on('object:modified', _onModified);
}

function _ptr(e) { return fc.getPointer(e.e); }
function _color() { return document.getElementById('color-picker')?.value || '#ff0000'; }
function _strokeWidth() { return parseInt(document.getElementById('stroke-width')?.value || '2', 10); }

function _onDown(e) {
  if (currentTool === 'select' || currentTool === 'pen') return;
  const { x, y } = _ptr(e);
  drawStart = { x, y };
  if (currentTool === 'dot')  { _addDot(x, y); drawStart = null; return; }
  if (currentTool === 'text') { _addText(x, y); drawStart = null; return; }
  _createPreview(x, y);
}

function _onMove(e) {
  if (!drawStart || !previewObj) return;
  const { x, y } = _ptr(e);
  _updatePreview(drawStart.x, drawStart.y, x, y);
  fc.requestRenderAll();
}

function _onUp(e) {
  if (!drawStart) return;
  const { x, y } = _ptr(e);
  if (previewObj) { fc.remove(previewObj); previewObj = null; }
  const { x: sx, y: sy } = drawStart;
  drawStart = null;
  if (Math.abs(x - sx) < 4 && Math.abs(y - sy) < 4) return; // too small
  switch (currentTool) {
    case 'rect':    _addRect(sx, sy, x - sx, y - sy); break;
    case 'ellipse': _addEllipse(sx, sy, x, y);        break;
    case 'arrow':   _addArrow(sx, sy, x, y);           break;
  }
}

// ---- Preview helpers (dashed ghost while dragging) ----

function _createPreview(x, y) {
  const opts = {
    stroke: _color(), fill: 'transparent', strokeWidth: _strokeWidth(),
    selectable: false, evented: false, opacity: 0.55,
    strokeDashArray: [5, 5],
  };
  switch (currentTool) {
    case 'rect':
      previewObj = new fabric.Rect({ left: x, top: y, width: 0, height: 0, ...opts });
      break;
    case 'ellipse':
      previewObj = new fabric.Ellipse({ left: x, top: y, rx: 0, ry: 0, ...opts });
      break;
    case 'arrow':
      previewObj = new fabric.Line([x, y, x, y], {
        stroke: _color(), strokeWidth: _strokeWidth(),
        selectable: false, evented: false, opacity: 0.55,
      });
      break;
  }
  if (previewObj) fc.add(previewObj);
}

function _updatePreview(sx, sy, ex, ey) {
  if (!previewObj) return;
  switch (currentTool) {
    case 'rect': {
      const left = Math.min(sx, ex), top = Math.min(sy, ey);
      previewObj.set({ left, top, width: Math.abs(ex - sx), height: Math.abs(ey - sy) });
      break;
    }
    case 'ellipse': {
      const rx = Math.abs(ex - sx) / 2, ry = Math.abs(ey - sy) / 2;
      previewObj.set({ left: Math.min(sx, ex), top: Math.min(sy, ey), rx, ry });
      break;
    }
    case 'arrow':
      previewObj.set({ x2: ex, y2: ey });
      break;
  }
}

// ---- Shape creators (send to server, then add to canvas) ----

function _genId() { return `obj_${Math.random().toString(36).slice(2, 10)}`; }

async function _addRect(x, y, w, h) {
  const left = w >= 0 ? x : x + w, top = h >= 0 ? y : y + h;
  const payload = {
    id: _genId(), type: 'rect',
    x: left, y: top, width: Math.abs(w), height: Math.abs(h),
    stroke: _color(), fill: 'transparent', strokeWidth: _strokeWidth(),
    createdBy: 'human',
  };
  await _postDraw('rect', payload);
  setTool('select');
}

async function _addEllipse(sx, sy, ex, ey) {
  const cx = (sx + ex) / 2, cy = (sy + ey) / 2;
  const rx = Math.abs(ex - sx) / 2, ry = Math.abs(ey - sy) / 2;
  const payload = {
    id: _genId(), type: 'ellipse', cx, cy, rx, ry,
    stroke: _color(), fill: 'transparent', strokeWidth: _strokeWidth(),
    createdBy: 'human',
  };
  await _postDraw('ellipse', payload);
  setTool('select');
}

async function _addArrow(x1, y1, x2, y2) {
  const payload = {
    id: _genId(), type: 'arrow', x1, y1, x2, y2,
    stroke: _color(), strokeWidth: _strokeWidth(),
    createdBy: 'human',
  };
  await _postDraw('arrow', payload);
  setTool('select');
}

async function _addDot(cx, cy) {
  const payload = {
    id: _genId(), type: 'dot', cx, cy, radius: 6,
    fill: _color(), stroke: _color(), strokeWidth: 1,
    createdBy: 'human',
  };
  await _postDraw('dot', payload);
}

async function _addText(x, y) {
  const id = _genId();
  const itext = new fabric.IText('Text', {
    left: x, top: y, fontSize: 20, fill: _color(),
    fontFamily: 'Arial', selectable: true, evented: true,
  });
  itext._sid = id;
  fc.add(itext);
  fc.setActiveObject(itext);
  itext.enterEditing();

  itext.on('editing:exited', async () => {
    if (itext._synced) return;
    itext._synced = true;
    const payload = {
      id, type: 'text',
      x: Math.round(itext.left), y: Math.round(itext.top),
      text: itext.text,
      fontSize: itext.fontSize,
      fill: itext.fill,
      fontFamily: itext.fontFamily,
      createdBy: 'human',
    };
    const saved = await _postDraw('text', payload);
    itext._sid = saved.id;
  });

  setTool('select');
}

async function _onPathCreated(e) {
  const p = e.path;
  if (p._sid) return; // already synced
  const id = _genId();
  p._sid = id;
  const pathStr = p.path ? fabric.util.joinPath(p.path) : '';
  const payload = {
    id, type: 'freehand',
    path: pathStr,
    left: Math.round(p.left), top: Math.round(p.top),
    stroke: p.stroke, strokeWidth: p.strokeWidth,
    createdBy: 'human',
  };
  const saved = await _postDraw('freehand', payload);
  p._sid = saved.id;
  p._synced = true;
}

// ---- Fabric object builders (server → canvas) ----

function _fabricRect(obj) {
  const r = new fabric.Rect({
    left: obj.x, top: obj.y, width: obj.width, height: obj.height,
    stroke: obj.stroke || '#ff0000',
    fill: obj.fill === 'transparent' ? 'transparent' : (obj.fill || 'transparent'),
    strokeWidth: obj.strokeWidth || 2,
    selectable: true, evented: true,
  });
  r._sid = obj.id; r._type = 'rect';
  fc.add(r); fc.requestRenderAll(); return r;
}

function _fabricEllipse(obj) {
  const e = new fabric.Ellipse({
    left: (obj.cx || 0) - (obj.rx || 50),
    top: (obj.cy || 0) - (obj.ry || 30),
    rx: obj.rx || 50, ry: obj.ry || 30,
    stroke: obj.stroke || '#ff0000',
    fill: obj.fill === 'transparent' ? 'transparent' : (obj.fill || 'transparent'),
    strokeWidth: obj.strokeWidth || 2,
    selectable: true, evented: true,
  });
  e._sid = obj.id; e._type = 'ellipse';
  fc.add(e); fc.requestRenderAll(); return e;
}

function _fabricArrow(obj) {
  const g = _buildArrowGroup(obj.x1, obj.y1, obj.x2, obj.y2, obj.stroke, obj.strokeWidth);
  g._sid = obj.id; g._type = 'arrow';
  g._arrowCoords = { x1: obj.x1, y1: obj.y1, x2: obj.x2, y2: obj.y2 };
  fc.add(g); fc.requestRenderAll(); return g;
}

function _buildArrowGroup(x1, y1, x2, y2, color = '#ff0000', lw = 2) {
  const headLen = Math.max(12, lw * 5);
  const angle = Math.atan2(y2 - y1, x2 - x1);

  const line = new fabric.Line([x1, y1, x2, y2], {
    stroke: color, strokeWidth: lw, selectable: false, evented: false,
  });

  const px = x2, py = y2;
  const qx = x2 - headLen * Math.cos(angle - Math.PI / 6);
  const qy = y2 - headLen * Math.sin(angle - Math.PI / 6);
  const rx = x2 - headLen * Math.cos(angle + Math.PI / 6);
  const ry = y2 - headLen * Math.sin(angle + Math.PI / 6);

  const head = new fabric.Polygon(
    [{ x: px, y: py }, { x: qx, y: qy }, { x: rx, y: ry }],
    { fill: color, stroke: color, strokeWidth: 1, selectable: false, evented: false },
  );

  return new fabric.Group([line, head], { selectable: true, evented: true });
}

function _fabricDot(obj) {
  const r = obj.radius || 6;
  const c = new fabric.Circle({
    left: (obj.cx || 0) - r, top: (obj.cy || 0) - r, radius: r,
    fill: obj.fill || obj.stroke || '#ff0000',
    stroke: obj.stroke || '#ff0000', strokeWidth: obj.strokeWidth || 1,
    selectable: true, evented: true,
  });
  c._sid = obj.id; c._type = 'dot';
  fc.add(c); fc.requestRenderAll(); return c;
}

function _fabricLine(obj) {
  const l = new fabric.Line([obj.x1, obj.y1, obj.x2, obj.y2], {
    stroke: obj.stroke || '#ff0000', strokeWidth: obj.strokeWidth || 2,
    selectable: true, evented: true,
  });
  l._sid = obj.id; l._type = 'line';
  fc.add(l); fc.requestRenderAll(); return l;
}

function _fabricText(obj) {
  const t = new fabric.Text(obj.text || '', {
    left: obj.x, top: obj.y,
    fontSize: obj.fontSize || 18,
    fill: obj.fill || '#000000',
    fontFamily: obj.fontFamily || 'Arial',
    selectable: true, evented: true,
  });
  t._sid = obj.id; t._type = 'text';
  fc.add(t); fc.requestRenderAll(); return t;
}

function _fabricFreehand(obj) {
  if (!obj.path) return;
  const p = new fabric.Path(obj.path, {
    left: obj.left || 0, top: obj.top || 0,
    stroke: obj.stroke || '#000000',
    fill: 'transparent',
    strokeWidth: obj.strokeWidth || 2,
    selectable: true, evented: true,
  });
  p._sid = obj.id; p._type = 'freehand'; p._synced = true;
  fc.add(p); fc.requestRenderAll(); return p;
}

function _addServerObject(obj) {
  const map = {
    rect: _fabricRect, ellipse: _fabricEllipse, arrow: _fabricArrow,
    dot: _fabricDot, line: _fabricLine, text: _fabricText, freehand: _fabricFreehand,
  };
  map[obj.type]?.(obj);
}

// ---- Object-modified → PATCH to server ----

async function _onModified(e) {
  const obj = e.target;
  if (!obj._sid) return;
  const patch = _toServerPatch(obj);
  if (!patch) return;
  try {
    await fetch(`${API}/objects/${obj._sid}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
  } catch (_) { /* best effort */ }
}

function _toServerPatch(obj) {
  const type = obj._type || obj.type;
  const base = { stroke: obj.stroke, fill: obj.fill, strokeWidth: obj.strokeWidth };
  const sx = obj.scaleX || 1, sy = obj.scaleY || 1;

  switch (type) {
    case 'rect':
      return { ...base, x: Math.round(obj.left), y: Math.round(obj.top),
               width: Math.round(obj.width * sx), height: Math.round(obj.height * sy) };
    case 'ellipse':
      return { ...base,
               cx: Math.round(obj.left + obj.rx * sx),
               cy: Math.round(obj.top + obj.ry * sy),
               rx: Math.round(obj.rx * sx), ry: Math.round(obj.ry * sy) };
    case 'dot':
      return { ...base,
               cx: Math.round(obj.left + obj.radius * sx),
               cy: Math.round(obj.top + obj.radius * sy),
               radius: Math.round(obj.radius * sx) };
    case 'text':
      return { ...base, x: Math.round(obj.left), y: Math.round(obj.top),
               text: obj.text, fontSize: Math.round(obj.fontSize * sx) };
    case 'arrow':
      if (obj._arrowCoords) {
        // Compute displacement from original group bounding box origin
        const { x1, y1, x2, y2 } = obj._arrowCoords;
        const origLeft = Math.min(x1, x2);
        const origTop  = Math.min(y1, y2);
        const dx = obj.left - origLeft, dy = obj.top - origTop;
        return { ...base, x1: Math.round(x1+dx), y1: Math.round(y1+dy),
                          x2: Math.round(x2+dx), y2: Math.round(y2+dy) };
      }
      return null;
    case 'freehand':
      return { ...base, left: Math.round(obj.left), top: Math.round(obj.top) };
    case 'line':
      return { ...base, x1: Math.round(obj.x1+obj.left), y1: Math.round(obj.y1+obj.top),
               x2: Math.round(obj.x2+obj.left), y2: Math.round(obj.y2+obj.top) };
    default:
      return null;
  }
}

// ---- SSE ----

let _lastViewport = null;

function _setupSSE() {
  const es = new EventSource(`${API}/canvas/events`);
  es.onmessage = async (e) => {
    if (isApplyingState) return;
    isApplyingState = true;
    try {
      await _applyState(JSON.parse(e.data));
    } finally {
      isApplyingState = false;
    }
  };
}

// ---- Apply full server state to canvas (incremental) ----

let _lastBgFilename = null;

async function _applyState(state) {
  const { canvas: meta, objects, viewport, cursor } = state;

  _resizeCanvas(meta.width, meta.height);

  // Background image — only reload when filename changes
  if (meta.backgroundImage !== _lastBgFilename) {
    _lastBgFilename = meta.backgroundImage;
    if (meta.backgroundImage) {
      // Cache-bust with timestamp so the browser always fetches the new crop file
      const url = `/uploads/${meta.backgroundImage}?t=${Date.now()}`;
      await new Promise(resolve => {
        fc.setBackgroundImage(url, () => {
          fc.requestRenderAll();
          resolve();
        }, { crossOrigin: 'anonymous' });
      });
    } else {
      await new Promise(resolve => fc.setBackgroundImage(null, resolve));
    }
  }

  // Build lookup of what's already on canvas
  const onCanvas = new Map();
  fc.getObjects().forEach(o => { if (o._sid) onCanvas.set(o._sid, o); });

  const serverIds = new Set(objects.map(o => o.id));

  // Remove objects no longer in server state
  for (const [id, obj] of onCanvas) {
    if (!serverIds.has(id)) fc.remove(obj);
  }

  // Add new objects
  for (const obj of objects) {
    if (!onCanvas.has(obj.id)) _addServerObject(obj);
  }

  fc.requestRenderAll();
  _updateSidebar(objects);

  // Viewport (zoom + pan)
  if (viewport) {
    const vpKey = `${viewport.zoom},${viewport.panX},${viewport.panY}`;
    if (vpKey !== _lastViewport) {
      _lastViewport = vpKey;
      _applyViewport(viewport);
    }
  }

  // Model cursor
  if (cursor) _applyCursor(cursor);
}

// ---- Delete & Clear ----

async function deleteSelected() {
  const active = fc.getActiveObjects();
  if (!active.length) return;
  for (const obj of active) {
    if (obj._sid) {
      await fetch(`${API}/objects/${obj._sid}`, { method: 'DELETE' }).catch(() => {});
    }
    fc.remove(obj);
  }
  fc.discardActiveObject();
  fc.requestRenderAll();
}

async function clearCanvas() {
  if (!confirm('Clear all annotations? This cannot be undone.')) return;
  await fetch(`${API}/canvas/clear`, { method: 'POST' });
  fc.getObjects().slice().forEach(o => fc.remove(o));
  fc.requestRenderAll();
  _updateSidebar([]);
}

// ---- File upload ----

function _setupFileUpload() {
  document.getElementById('file-upload')?.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('image', file);
    const res = await fetch(`${API}/canvas/load-image`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!data.ok) alert('Image load failed: ' + data.error);
    e.target.value = '';
  });
}

// ---- Keyboard shortcuts ----

function _setupKeyboard() {
  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    if (e.key === 'Delete' || e.key === 'Backspace') deleteSelected();
    if (e.key === 'Escape') setTool('select');
    const map = { s: 'select', r: 'rect', e: 'ellipse', a: 'arrow', d: 'dot', p: 'pen', t: 'text' };
    if (map[e.key.toLowerCase()]) setTool(map[e.key.toLowerCase()]);
  });
}

// ---- Export ----

async function exportPng() {
  const res = await fetch(`${API}/export/png`);
  const blob = await res.blob();
  _download(blob, 'annotated.png');
}

async function exportJson() {
  const res = await fetch(`${API}/export/json`);
  const data = await res.json();
  _download(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), 'annotations.json');
}

function _download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---- Sidebar ----

function _updateSidebar(objects) {
  const list = document.getElementById('object-list');
  const count = document.getElementById('obj-count');
  if (!list) return;
  count.textContent = `(${objects.length})`;
  list.innerHTML = objects.map(obj => `
    <div class="obj-item" data-id="${obj.id}" onclick="_selectById('${obj.id}')">
      <span class="obj-type">${obj.type}</span>
      <span class="obj-by-${obj.createdBy}">[${obj.createdBy}]</span>
      <span class="obj-id">${obj.id}</span>
      ${obj.label ? `<span class="obj-label">${obj.label}</span>` : ''}
    </div>
  `).join('');
}

window._selectById = function(id) {
  const obj = fc.getObjects().find(o => o._sid === id);
  if (!obj) return;
  fc.setActiveObject(obj);
  fc.requestRenderAll();
};

// ---- Viewport ----

function _applyViewport(vp) {
  // Fabric.js: setZoom then absolutePan
  // absolutePan({x, y}) makes canvas point (x/zoom, y/zoom) appear at screen top-left
  fc.setZoom(vp.zoom);
  fc.absolutePan(new fabric.Point(vp.panX, vp.panY));
  fc.requestRenderAll();
}

// ---- Model cursor ----

function _applyCursor(cursor) {
  const el = document.getElementById('model-cursor');
  const lbl = document.getElementById('model-cursor-label');
  if (!el) return;

  if (!cursor.visible || cursor.x == null || cursor.y == null) {
    el.style.display = 'none';
    return;
  }

  // Convert canvas coordinates → screen coordinates using current viewport transform
  // Fabric viewportTransform: [zoom, 0, 0, zoom, tx, ty]
  // screen = canvas * zoom + translate
  const vt = fc.viewportTransform;        // [a, b, c, d, e, f]
  const zoom = vt[0];
  const tx   = vt[4];
  const ty   = vt[5];

  const wrap = document.getElementById('canvas-wrap');
  const canvasEl = document.getElementById('main-canvas');
  const wrapRect  = wrap.getBoundingClientRect();
  const canvasRect = canvasEl.getBoundingClientRect();

  // Position of canvas element relative to canvas-wrap
  const canvasOffX = canvasRect.left - wrapRect.left + wrap.scrollLeft;
  const canvasOffY = canvasRect.top  - wrapRect.top  + wrap.scrollTop;

  const screenX = cursor.x * zoom + tx + canvasOffX;
  const screenY = cursor.y * zoom + ty + canvasOffY;

  el.style.display = 'flex';
  el.style.left = `${screenX}px`;
  el.style.top  = `${screenY}px`;
  lbl.textContent = cursor.label || '';
  lbl.style.display = cursor.label ? 'block' : 'none';
}

// ---- Helpers ----

function _resizeCanvas(w, h) {
  if (fc.width !== w || fc.height !== h) {
    fc.setWidth(w);
    fc.setHeight(h);
  }
}

async function _postDraw(type, payload) {
  const res = await fetch(`${API}/draw/${type}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}
