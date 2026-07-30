// client/app.js
'use strict';

const API = '/api';

let fc;            // fabric.Canvas instance
let currentTool = 'select';
let isApplyingState = false;   // prevent re-entrant SSE cycles
let _lastServerSegKey = null;  // tracks last rendered segmentation to skip SSE bounce
let drawStart = null;
let previewObj = null;

// ---- Atlas mode state ----
let _atlasManifest     = null;    // full manifest from server (tiles array + dimensions)
let _atlasTileObjects  = new Map(); // key "tx,ty" → fabric.Image (loaded tiles)
let _atlasTilesLoading = new Set(); // keys currently being fetched
let _atlasLoading      = false;     // guard to prevent duplicate manifest fetches
let _atlasPanning      = false;     // true while user drags in select mode
let _atlasPanLastX     = 0;
let _atlasPanLastY     = 0;
let _atlasLoadTimer    = null;      // debounce handle for lazy tile loader
let _atlasSyncTimer    = null;      // debounce handle for viewport sync
let _canvasHovered     = false;     // true while pointer is over #canvas-wrap

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
  _setupDatasetBrowser();
  _setupKeyboard();
  _setupFilterSliders();
  _setupSSE();
  _setupResizeHandles();
  _setupTileNav();
  _setupAtlasButtons();
  _setupSegmentation();
  _setupCs4Panel();
  _setupSavePatternPanel();

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
  document.getElementById('btn-save-gt')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-save-gt');
    if (btn) btn.disabled = true;
    try {
      const res  = await fetch(`${API}/gt/annotate`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        alert('Save GT failed: ' + (data.error || res.statusText));
        return;
      }
      const prev = data.prev ? ` (was ${data.prev.gt_count})` : '';
      alert(`Saved GT: ${data.sample_id} → ${data.gt_count} particles${prev}`);
    } catch (err) {
      alert('Save GT error: ' + err.message);
    } finally {
      if (btn) btn.disabled = false;
    }
  });
  document.getElementById('btn-reset-filters')?.addEventListener('click', () => {
    fetch(`${API}/viewport/filters`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ brightness: 100, contrast: 100 }),
    }).catch(() => {});
  });

  document.getElementById('btn-randomize')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-randomize');
    if (btn) btn.disabled = true;
    try {
      const res  = await fetch(`${API}/randomize`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        alert('Randomize failed: ' + (data.error || res.statusText));
        return;
      }
      // Filters are broadcast via SSE; no manual DOM update needed.
      // Brief visual confirmation in the button text.
      if (btn) {
        btn.textContent = '🎲 Randomized!';
        setTimeout(() => { btn.textContent = '🎲 Randomize'; }, 2000);
      }
    } catch (e) {
      alert('Randomize error: ' + e.message);
    } finally {
      if (btn) btn.disabled = false;
    }
  });

  // Mode toggle
  document.getElementById('btn-zoom-in')?.addEventListener('click',  () => _zoomCanvas(1.3));
  document.getElementById('btn-zoom-out')?.addEventListener('click', () => _zoomCanvas(1 / 1.3));

  document.getElementById('btn-fit-image')?.addEventListener('click', () => {
    const wrap = document.getElementById('canvas-wrap');
    if (!wrap) return;
    const availW = wrap.clientWidth  - 40;
    const availH = wrap.clientHeight - 40;
    const fitZoom = Math.min(availW / fc.getWidth(), availH / fc.getHeight(), 1);
    fc.setZoom(fitZoom);
    fc.absolutePan(new fabric.Point(0, 0));
  });

  document.getElementById('btn-mode-image')?.addEventListener('click', async () => {
    await fetch(`${API}/camera/mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'image' }),
    }).catch(() => {});
  });

  document.getElementById('btn-mode-grid')?.addEventListener('click', async () => {
    const res = await fetch(`${API}/camera/init`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    }).catch(() => null);
    if (res && !res.ok) {
      const data = await res.json().catch(() => ({}));
      alert('Grid mode error: ' + (data.error || res.statusText));
    }
  });
}

function _setupAtlasButtons() {
  // Atlas button: enter atlas mode for current region/fw
  document.getElementById('btn-atlas')?.addEventListener('click', async () => {
    const res = await fetch(`${API}/atlas/enter`, { method: 'POST' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      alert('Atlas error: ' + (data.error || res.statusText));
    }
    // SSE broadcast drives the UI update
  });

  // Back button: exit atlas mode, return to grid
  document.getElementById('btn-atlas-back')?.addEventListener('click', async () => {
    _exitAtlasClientState();
    await fetch(`${API}/atlas/exit`, { method: 'POST' }).catch(() => {});
    // SSE broadcast drives the UI update (tile reload)
  });

  // Fit button: zoom to show whole atlas
  document.getElementById('btn-atlas-fit')?.addEventListener('click', () => {
    if (!_atlasManifest) return;
    const fitZoom = Math.min(
      fc.getWidth()  / _atlasManifest.atlasWidth,
      fc.getHeight() / _atlasManifest.atlasHeight,
    ) * 0.92;
    fc.setZoom(fitZoom);
    fc.absolutePan(new fabric.Point(0, 0));
    _atlasLoadVisibleDebounced();
    _syncAtlasViewport();
  });

  // Grid overlay toggle
  document.getElementById('btn-atlas-grid')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-atlas-grid');
    const active = btn?.classList.contains('atlas-overlay-active');
    const newGrid = !active;
    const res = await fetch(`${API}/atlas/overlay`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ grid: newGrid }),
    }).catch(() => null);
    if (!res || !res.ok) return;
    if (btn) btn.classList.toggle('atlas-overlay-active', newGrid);
    // If grid is turned off, labels must also go off
    if (!newGrid) {
      await fetch(`${API}/atlas/overlay`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ labels: false }),
      }).catch(() => {});
      const lblBtn = document.getElementById('btn-atlas-labels');
      if (lblBtn) lblBtn.classList.remove('atlas-overlay-active');
    }
    // Trigger re-export of server-side PNG (force viewport re-sync so export is current)
    _syncAtlasViewport();
  });

  // Coordinate-label overlay toggle (only meaningful when grid is on)
  document.getElementById('btn-atlas-labels')?.addEventListener('click', async () => {
    const btn    = document.getElementById('btn-atlas-labels');
    const active = btn?.classList.contains('atlas-overlay-active');
    const newLabels = !active;
    // If enabling labels, ensure grid is also on
    const body = newLabels ? { grid: true, labels: true } : { labels: false };
    const res = await fetch(`${API}/atlas/overlay`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    }).catch(() => null);
    if (!res || !res.ok) return;
    if (btn) btn.classList.toggle('atlas-overlay-active', newLabels);
    if (newLabels) {
      const gridBtn = document.getElementById('btn-atlas-grid');
      if (gridBtn) gridBtn.classList.add('atlas-overlay-active');
    }
    _syncAtlasViewport();
  });

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
      if (o._isTile || o._isSegment) return;  // atlas tiles + seg overlays are never interactive
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

  // Zoom with mouse wheel — atlas mode: all scroll; other modes: only Ctrl+scroll (trackpad pinch)
  fc.lowerCanvasEl.addEventListener('wheel', (e) => {
    if (_currentUiMode === 'atlas') {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.85 : 1 / 0.85;
      const newZoom = Math.max(0.02, Math.min(fc.getZoom() * delta, 15));
      const rect = fc.lowerCanvasEl.getBoundingClientRect();
      fc.zoomToPoint(new fabric.Point(e.clientX - rect.left, e.clientY - rect.top), newZoom);
      _atlasLoadVisibleDebounced();
      _syncAtlasViewport();
    } else if (e.ctrlKey) {
      // Prevent browser zoom; zoom canvas to mouse point instead (trackpad pinch or Ctrl+scroll)
      e.preventDefault();
      const delta   = e.deltaY > 0 ? 0.85 : 1 / 0.85;
      const newZoom = Math.max(0.05, Math.min(fc.getZoom() * delta, 20));
      const rect    = fc.lowerCanvasEl.getBoundingClientRect();
      fc.zoomToPoint(new fabric.Point(e.clientX - rect.left, e.clientY - rect.top), newZoom);
    }
  }, { passive: false });

  // Track hover so keyboard shortcuts know whether canvas is active
  const wrap = document.getElementById('canvas-wrap');
  if (wrap) {
    wrap.addEventListener('mouseenter', () => { _canvasHovered = true; });
    wrap.addEventListener('mouseleave', () => { _canvasHovered = false; });
    // Also prevent browser Ctrl+scroll when hovered (fires on wrap, not just lower canvas)
    wrap.addEventListener('wheel', (e) => {
      if (e.ctrlKey) e.preventDefault();
    }, { passive: false });
  }
}

function _ptr(e) { return fc.getPointer(e.e); }
function _color() { return document.getElementById('color-picker')?.value || '#ff0000'; }
function _strokeWidth() { return parseInt(document.getElementById('stroke-width')?.value || '2', 10); }

function _onDown(e) {
  // Pan viewport: drag with select tool when no object is active (all modes)
  if (currentTool === 'select' && !fc.getActiveObject()) {
    _atlasPanning = true;
    _atlasPanLastX = e.e.clientX;
    _atlasPanLastY = e.e.clientY;
    return;
  }
  if (currentTool === 'select' || currentTool === 'pen') return;
  const { x, y } = _ptr(e);
  drawStart = { x, y };
  if (currentTool === 'dot')  { _addDot(x, y); drawStart = null; return; }
  if (currentTool === 'text') { _addText(x, y); drawStart = null; return; }
  _createPreview(x, y);
}

function _onMove(e) {
  if (_atlasPanning) {
    const dx = e.e.clientX - _atlasPanLastX;
    const dy = e.e.clientY - _atlasPanLastY;
    _atlasPanLastX = e.e.clientX;
    _atlasPanLastY = e.e.clientY;
    const vt = [...fc.viewportTransform];
    vt[4] += dx;
    vt[5] += dy;
    fc.setViewportTransform(vt);
    if (_currentUiMode === 'atlas') _atlasLoadVisibleDebounced();
    return;
  }
  if (!drawStart || !previewObj) return;
  const { x, y } = _ptr(e);
  _updatePreview(drawStart.x, drawStart.y, x, y);
  fc.requestRenderAll();
}

function _onUp(e) {
  if (_atlasPanning) {
    _atlasPanning = false;
    _syncAtlasViewport();
    return;
  }
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
    if (p._sid) return;

    const id = _genId();
    p._sid = id;

    const pathStr = p.path ? fabric.util.joinPath(p.path) : '';

    const payload = {
      id,
      type: 'freehand',
      path: pathStr,
      left: Math.round(p.left),
      top: Math.round(p.top),
      stroke: p.stroke,
      strokeWidth: p.strokeWidth,
      createdBy: 'human',
      // no coordMode here
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
  const { canvas: meta, objects, viewport, cursor, filters } = state;

  _resizeCanvas(meta.width, meta.height);

  // Background image — only reload when filename changes
  if (meta.backgroundImage !== _lastBgFilename) {
    _lastBgFilename = meta.backgroundImage;
    _clearSegmentation();
    if (meta.backgroundImage) {
      // If path starts with '/' it's a tile or other absolute URL path (e.g. /tile-assets/...)
      // Otherwise it's an uploads filename, prefix with /uploads/
      const bgPath = meta.backgroundImage.startsWith('/')
        ? meta.backgroundImage
        : `/uploads/${meta.backgroundImage}`;
      const url = `${bgPath}?t=${Date.now()}`;
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

  // Viewport (zoom + pan) — skip in atlas mode (client manages viewport)
  if (viewport && _currentUiMode !== 'atlas') {
    const vpKey = `${viewport.zoom},${viewport.panX},${viewport.panY}`;
    if (vpKey !== _lastViewport) {
      _lastViewport = vpKey;
      _applyViewport(viewport);
    }
  }

  // Model cursor
  if (cursor) _applyCursor(cursor);

  // Filters
  if (filters) _applyFilters(filters);

  // UI mode + tile grid status
  if (state.uiMode !== undefined) _applyUiMode(state.uiMode, state.tileGrid, state.atlas);

  // Atlas overlay button sync (keep visual state consistent with server state)
  if (state.atlasOverlay !== undefined) _applyAtlasOverlayState(state.atlasOverlay);

  // Segmentation overlay — render when server state changes (e.g. agent ran segment tool)
  const incomingSegKey = _segKey(state.segmentation);
  if (incomingSegKey !== _lastServerSegKey) {
    _lastServerSegKey = incomingSegKey;
    _clearSegmentationCanvas();
    if (state.segmentation) await _renderSegmentationData(state.segmentation);
  }

  // Live histogram — refresh on any state change (debounced)
  _scheduleLiveHistogram();
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
    if (!data.ok) { alert('Image load failed: ' + data.error); }
    e.target.value = '';
  });
}

// ---- Dataset browser ----

let _datasetLabeled   = [];
let _datasetUnlabeled = [];
let _datasetValidationAndrea = [];
let _datasetValidationUncLuca = [];

async function _setupDatasetBrowser() {
  const btn    = document.getElementById('btn-browse-dataset');
  const dialog = document.getElementById('dataset-dialog');
  const catSel = document.getElementById('dataset-category-select');
  const imgSel = document.getElementById('dataset-image-select');
  if (!btn || !dialog || !catSel || !imgSel) return;

  // Load category list once
  try {
    const data = await fetch(`${API}/dataset/list`).then(r => r.json());
    _datasetLabeled = (data.labeled || [])
    .map(c => ({ ...c, source: 'labeled' }));

    _datasetUnlabeled = (data.unlabeled || [])
      .map(c => ({ ...c, source: 'unlabeled' }));

    _datasetValidationAndrea = (data.validation_andrea || [])
      .map(c => ({ ...c, source: 'validation_andrea' }));

    _datasetValidationUncLuca = (data.validation_unc_luca || [])
      .map(c => ({ ...c, source: 'validation_unc_luca' }));
    const allCats = [
      ..._datasetLabeled,
      ..._datasetUnlabeled,
      ..._datasetValidationAndrea,
      ..._datasetValidationUncLuca,
    ];
    catSel.innerHTML = [
      '<optgroup label="Labeled">',
      ..._datasetLabeled.map(c  => `<option value="${c.source}::${c.name}">${c.name} (${c.images.length})</option>`),
      '</optgroup>',

      '<optgroup label="Unlabeled (SSL)">',
      ..._datasetUnlabeled.map(c => `<option value="${c.source}::${c.name}">${c.name} (${c.images.length})</option>`),
      '</optgroup>',

      '<optgroup label="Validation Andrea CS2">',
      ..._datasetValidationAndrea.map(c => `<option value="${c.source}::${c.name}">${c.name} (${c.images.length})</option>`),
      '</optgroup>',

      '<optgroup label="Validation UNC Luca CS2">',
      ..._datasetValidationUncLuca.map(c => `<option value="${c.source}::${c.name}">${c.name} (${c.images.length})</option>`),
      '</optgroup>',
    ].join('');
    _populateDatasetImages();
  } catch (_) {}

  catSel.addEventListener('change', _populateDatasetImages);

  btn.addEventListener('click', () => dialog.showModal());
  document.getElementById('btn-dataset-close')?.addEventListener('click', () => dialog.close());

  document.getElementById('btn-sample-image')?.addEventListener('click', async () => {
    const btnSample = document.getElementById('btn-sample-image');
    btnSample.disabled = true;
    btnSample.textContent = '⏳ Sampling…';
    try {
      const res  = await fetch(`${API}/dataset/sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 'unlabeled' }),
      });
      const data = await res.json();
      if (!data.ok) { alert('Sample failed: ' + (data.error || res.statusText)); }
    } catch (e) {
      alert('Sample failed: ' + e.message);
    } finally {
      btnSample.disabled = false;
      btnSample.textContent = '🎲 Sample';
    }
  });

  document.getElementById('btn-dataset-load')?.addEventListener('click', async () => {
    const [source, ...catParts] = catSel.value.split('::');
    const category = catParts.join('::');
    const filename = imgSel.value;
    if (!source || !category || !filename) return;
    const res  = await fetch(`${API}/dataset/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, category, filename }),
    });
    const data = await res.json();
    if (!data.ok) { alert('Failed to load: ' + (data.error || res.statusText)); return; }
    dialog.close();
  });
}

function _populateDatasetImages() {
  const catSel = document.getElementById('dataset-category-select');
  const imgSel = document.getElementById('dataset-image-select');
  if (!catSel || !imgSel) return;
  const [source, ...catParts] = catSel.value.split('::');
  const catName = catParts.join('::');
  const allCats = [
    ..._datasetLabeled,
    ..._datasetUnlabeled,
    ..._datasetValidationAndrea,
    ..._datasetValidationUncLuca,
  ];
  const cat = allCats.find(c => c.source === source && c.name === catName);
  imgSel.innerHTML = (cat?.images || [])
    .map(img => `<option value="${img.filename}">${img.filename}</option>`)
    .join('');
}

// ---- Keyboard shortcuts ----

function _zoomCanvas(factor) {
  const newZoom = Math.max(0.05, Math.min(fc.getZoom() * factor, 20));
  const center  = new fabric.Point(fc.getWidth() / 2, fc.getHeight() / 2);
  fc.zoomToPoint(center, newZoom);
  if (_currentUiMode === 'atlas') {
    _atlasLoadVisibleDebounced();
    _syncAtlasViewport();
  }
}

function _setupKeyboard() {
  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    // Ctrl+= / Ctrl++ / Ctrl+- zoom the canvas (not browser) when cursor is over it
    if (_canvasHovered && e.ctrlKey) {
      if (e.key === '=' || e.key === '+') { e.preventDefault(); _zoomCanvas(1.3); return; }
      if (e.key === '-')                  { e.preventDefault(); _zoomCanvas(1 / 1.3); return; }
      if (e.key === '0')                  { e.preventDefault(); fc.setZoom(1); fc.absolutePan(new fabric.Point(0, 0)); return; }
    }
    // Arrow keys navigate the tile grid when in grid mode
    if (_currentUiMode === 'grid') {
      const arrowMap = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down' };
      if (arrowMap[e.key]) {
        e.preventDefault();
        _cameraMove(arrowMap[e.key]);
        return;
      }
    }
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

// ---- Filters ----

function _applyFilters(filters) {
  const wrap = document.getElementById('canvas-wrap');
  if (!wrap) return;
  wrap.style.filter =
    `brightness(${filters.brightness}%) contrast(${filters.contrast}%)`;
  // Sync sliders without triggering input events
  ['brightness', 'contrast'].forEach(k => {
    const slider = document.getElementById(`filter-${k}`);
    const label  = document.getElementById(`filter-${k}-val`);
    if (slider && slider.value != filters[k]) slider.value = filters[k];
    if (label && label.value != filters[k]) label.value = filters[k];
  });
}

async function _patchFilter(key, value) {
  await fetch(`${API}/viewport/filters`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ [key]: Number(value) }),
  }).catch(() => {});
}

function _setupFilterSliders() {
  ['brightness', 'contrast'].forEach(k => {
    const slider = document.getElementById(`filter-${k}`);
    const label  = document.getElementById(`filter-${k}-val`);
    if (!slider) return;
    slider.addEventListener('input', () => {
      if (label) label.value = slider.value;
      // Apply locally immediately for responsiveness
      const wrap = document.getElementById('canvas-wrap');
      if (wrap) {
        const b = document.getElementById('filter-brightness')?.value || 100;
        const c = document.getElementById('filter-contrast')?.value   || 100;
        wrap.style.filter = `brightness(${b}%) contrast(${c}%)`;
      }
    });
    slider.addEventListener('change', () => _patchFilter(k, slider.value));

    // Allow direct number entry in the value field
    if (label) {
      label.addEventListener('focus', () => label.select());
      label.addEventListener('input', () => {
        const v = Math.max(0, Math.min(300, Number(label.value) || 0));
        if (slider.value != v) slider.value = v;
        const wrap = document.getElementById('canvas-wrap');
        if (wrap) {
          const b = document.getElementById('filter-brightness')?.value || 100;
          const c = document.getElementById('filter-contrast')?.value   || 100;
          wrap.style.filter = `brightness(${b}%) contrast(${c}%)`;
        }
      });
      label.addEventListener('change', () => {
        const v = Math.max(0, Math.min(300, Number(label.value) || 0));
        label.value = v;
        slider.value = v;
        _patchFilter(k, v);
      });
      label.addEventListener('keydown', e => {
        if (e.key === 'Enter') { label.blur(); }
        if (e.key === 'Escape') { label.value = slider.value; label.blur(); }
      });
    }
  });
}

// ---- Live brightness histogram ----

let _histDebounceTimer = null;

function _scheduleLiveHistogram() {
  clearTimeout(_histDebounceTimer);
  _histDebounceTimer = setTimeout(_fetchAndDrawHistogram, 300);
}

async function _fetchAndDrawHistogram() {
  if (_currentUiMode === 'atlas') return;   // no histogram in atlas mode
  const canvas = document.getElementById('live-hist-canvas');
  if (!canvas) return;
  try {
    const [curRes, neutralRes] = await Promise.all([
      fetch('/api/histogram/current'),
      fetch('/api/histogram/neutral'),
    ]);
    if (!curRes.ok) return;
    const curData     = await curRes.json();
    const neutralData = neutralRes.ok ? await neutralRes.json() : null;

    if (curData.bins) _drawLiveHistogram(canvas, curData.bins, neutralData?.bins || null);

    // Update score badge in the heading (score is vs neutral)
    const scoreEl = document.getElementById('live-hist-score');
    if (scoreEl) {
      if (curData.score != null) {
        const s = curData.score;
        scoreEl.textContent = `score: ${s.toFixed(4)}`;
        scoreEl.className   = s < 0.15 ? 'good' : s < 0.4 ? 'warn' : 'bad';
      } else {
        scoreEl.textContent = '';
        scoreEl.className   = '';
      }
    }
  } catch (_) { /* no image loaded — silently skip */ }
}

function _drawLiveHistogram(canvas, bins, refBins = null) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  // Paddings to fit both axes inside the canvas
  const PL = 58;   // left  — y-axis labels
  const PR = 22;   // right — room for "255" label
  const PT = 8;    // top
  const PB = 44;   // bottom — x-axis labels
  const plotW = W - PL - PR;
  const plotH = H - PT - PB;

  // Background
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, W, H);

  const total  = bins.reduce((s, b) => s + b, 0) || 1;
  const norm   = bins.map(b => b / total);

  // If a reference is provided, use a shared max so both curves share the same y-axis scale
  const refTotal = refBins ? (refBins.reduce((s, b) => s + b, 0) || 1) : 1;
  const refNorm  = refBins ? refBins.map(b => b / refTotal) : null;
  const maxVal   = Math.max(...norm, ...(refNorm || []), 1e-9);
  const barW   = plotW / bins.length;

  // Bars — solid, no sub-pixel gap at this resolution
  for (let i = 0; i < bins.length; i++) {
    const barH = (norm[i] / maxVal) * plotH;
    const x    = PL + i * barW;
    const y    = PT + plotH - barH;
    const grey = Math.round((i / (bins.length - 1)) * 210 + 45);
    ctx.fillStyle = `rgb(${grey},${grey},${grey})`;
    ctx.fillRect(Math.floor(x), Math.floor(y), Math.ceil(barW), Math.ceil(barH));
  }

  // Axes
  ctx.strokeStyle = '#666';
  ctx.lineWidth = 1;
  // X-axis
  ctx.beginPath();
  ctx.moveTo(PL, PT + plotH);
  ctx.lineTo(PL + plotW, PT + plotH);
  ctx.stroke();
  // Y-axis
  ctx.beginPath();
  ctx.moveTo(PL, PT);
  ctx.lineTo(PL, PT + plotH);
  ctx.stroke();

  // Y-axis ticks + labels
  ctx.fillStyle = '#999';
  ctx.font = '14px sans-serif';
  ctx.textAlign = 'right';
  const yTicks = 4;
  for (let t = 0; t <= yTicks; t++) {
    const frac = t / yTicks;
    const yPos = PT + plotH - frac * plotH;
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(PL - 3, yPos);
    ctx.lineTo(PL,     yPos);
    ctx.stroke();
    ctx.fillText((maxVal * frac).toFixed(3), PL - 5, yPos + 5);
  }

  // Y-axis label (rotated)
  ctx.save();
  ctx.fillStyle = '#777';
  ctx.font = '13px sans-serif';
  ctx.textAlign = 'center';
  ctx.translate(12, PT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('density', 0, 0);
  ctx.restore();

  // X-axis ticks + labels at 0, 64, 128, 192, 255
  ctx.fillStyle = '#999';
  ctx.font = '14px sans-serif';
  ctx.textAlign = 'center';
  [0, 64, 128, 192, 255].forEach(v => {
    const x = PL + (v / (bins.length - 1)) * plotW;
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(x, PT + plotH);
    ctx.lineTo(x, PT + plotH + 4);
    ctx.stroke();
    ctx.fillText(String(v), x, PT + plotH + 18);
  });

  // X-axis label
  ctx.fillStyle = '#777';
  ctx.font = '13px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('brightness', PL + plotW / 2, H - 8);

  // Reference histogram overlay (red polyline, same y-scale as bars)
  if (refNorm && refNorm.length > 0) {
    ctx.strokeStyle = '#e53935';
    ctx.lineWidth   = 2;
    ctx.beginPath();
    refNorm.forEach((v, i) => {
      const x = PL + (i + 0.5) * (plotW / refNorm.length);
      const y = PT + plotH - (v / maxVal) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

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

// ---- Atlas mode — client ----

function _atlasLoadVisibleDebounced() {
  clearTimeout(_atlasLoadTimer);
  _atlasLoadTimer = setTimeout(_atlasLoadVisible, 120);
}

function _atlasLoadVisible() {
  if (!_atlasManifest) return;
  const { tileWidth, tileHeight, cols, rows, tiles } = _atlasManifest;
  const zoom = fc.getZoom();
  const vt   = fc.viewportTransform;

  // Atlas pixel coords at the viewport edges (vt[4]=tx, vt[5]=ty; atlas_left = -tx/zoom)
  const left   = -vt[4] / zoom;
  const top    = -vt[5] / zoom;
  const right  = left + fc.getWidth()  / zoom;
  const bottom = top  + fc.getHeight() / zoom;

  // Load ±1 tile around viewport, unload beyond ±3 to save memory
  const buf       = 1;
  const unloadBuf = 3;
  const x0 = Math.max(0,        Math.floor(left   / tileWidth)  - buf);
  const x1 = Math.min(cols - 1, Math.ceil( right  / tileWidth)  + buf);
  const y0 = Math.max(0,        Math.floor(top    / tileHeight) - buf);
  const y1 = Math.min(rows - 1, Math.ceil( bottom / tileHeight) + buf);

  // Unload distant tiles
  for (const [key, img] of _atlasTileObjects) {
    const [tx, ty] = key.split(',').map(Number);
    if (tx < x0 - unloadBuf || tx > x1 + unloadBuf ||
        ty < y0 - unloadBuf || ty > y1 + unloadBuf) {
      fc.remove(img);
      _atlasTileObjects.delete(key);
    }
  }

  // Build fast lookup
  const tileByKey = new Map(tiles.map(t => [`${t.x},${t.y}`, t]));

  // Load visible tiles
  for (let ty = y0; ty <= y1; ty++) {
    for (let tx = x0; tx <= x1; tx++) {
      const key = `${tx},${ty}`;
      if (_atlasTileObjects.has(key) || _atlasTilesLoading.has(key)) continue;
      const tileInfo = tileByKey.get(key);
      if (!tileInfo) continue;

      _atlasTilesLoading.add(key);
      fabric.Image.fromURL(tileInfo.urlPath + '?atlas=1', (img) => {
        if (!_atlasManifest) { _atlasTilesLoading.delete(key); return; }
        _atlasTilesLoading.delete(key);
        img.set({
          left: tx * tileWidth,
          top:  ty * tileHeight,
          selectable:  false,
          evented:     false,
          hoverCursor: 'crosshair',
          _isTile: true,
        });
        _atlasTileObjects.set(key, img);
        fc.add(img);
        fc.sendToBack(img);
        fc.requestRenderAll();
      }, { crossOrigin: 'anonymous' });
    }
  }
}

async function _enterAtlasFromServer(atlas) {
  if (_atlasLoading) return;
  _atlasLoading = true;
  try {
    const manifest = await fetch(`${API}/atlas/manifest?region=${atlas.region}&fw=${atlas.fw}`)
      .then(r => r.json());
    if (manifest.error) { console.error('[atlas] manifest error:', manifest.error); return; }
    _doEnterAtlasMode(manifest);
  } catch (e) {
    console.error('[atlas] enter error:', e);
  } finally {
    _atlasLoading = false;
  }
}

function _doEnterAtlasMode(manifest) {
  _atlasManifest = manifest;
  _atlasTileObjects.clear();
  _atlasTilesLoading.clear();

  // Clear the single-tile background image
  fc.setBackgroundImage(null, () => {});

  // Start at fit-to-screen zoom
  const fitZoom = Math.min(
    fc.getWidth()  / manifest.atlasWidth,
    fc.getHeight() / manifest.atlasHeight,
  ) * 0.92;
  fc.setZoom(fitZoom);
  fc.absolutePan(new fabric.Point(0, 0));

  _syncAtlasViewport();
  _atlasLoadVisible();
}

function _exitAtlasClientState() {
  if (!_atlasManifest) return;
  for (const img of _atlasTileObjects.values()) fc.remove(img);
  _atlasTileObjects.clear();
  _atlasTilesLoading.clear();
  _atlasManifest = null;
  _atlasLoading  = false;
  _lastViewport  = null;  // force re-apply of tile viewport after exit
  // Reset zoom immediately so there is no flash of the atlas fit-zoom
  // while waiting for the server SSE to send the grid viewport.
  fc.setZoom(1);
  fc.absolutePan(new fabric.Point(0, 0));
  fc.requestRenderAll();
}

function _syncAtlasViewport() {
  if (_currentUiMode !== 'atlas') return;
  clearTimeout(_atlasSyncTimer);
  _atlasSyncTimer = setTimeout(async () => {
    const zoom = fc.getZoom();
    const vt   = fc.viewportTransform;
    const panX = -vt[4];
    const panY = -vt[5];
    await fetch(`${API}/atlas/viewport`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zoom, panX, panY }),
    }).catch(() => {});
  }, 300);
}

// ---- UI mode + tile grid status ----

// Sync overlay toggle button visual state to match the server's atlasOverlay state.
function _applyAtlasOverlayState(overlay) {
  const gridBtn   = document.getElementById('btn-atlas-grid');
  const labelsBtn = document.getElementById('btn-atlas-labels');
  if (gridBtn)   gridBtn.classList.toggle('atlas-overlay-active',   !!overlay.grid);
  if (labelsBtn) labelsBtn.classList.toggle('atlas-overlay-active', !!overlay.labels);
}

let _currentUiMode = 'image';

function _applyUiMode(mode, tileGrid, atlas) {
  const prevMode = _currentUiMode;
  _currentUiMode = mode;

  const btnImage   = document.getElementById('btn-mode-image');
  const btnGrid    = document.getElementById('btn-mode-grid');
  const btnAtlas   = document.getElementById('btn-atlas');
  const atlasBar   = document.getElementById('atlas-bar');
  const status     = document.getElementById('grid-status');
  const label      = document.getElementById('grid-status-label');
  const minimap    = document.getElementById('tile-minimap');
  const tileNav    = document.getElementById('tile-nav');

  const btnFit = document.getElementById('btn-fit-image');

  if (mode === 'grid') {
    btnImage?.classList.remove('mode-btn-active');
    btnGrid?.classList.add('mode-btn-active');
    if (btnAtlas) btnAtlas.style.display  = '';
    if (atlasBar) atlasBar.style.display  = 'none';
    if (btnFit)   btnFit.style.display    = '';
    // Restore histogram, remove atlas-nav class
    const histPanel = document.getElementById('live-hist-panel');
    if (histPanel) histPanel.style.display = '';
    if (tileNav) tileNav.classList.remove('atlas-nav-mode');
    if (tileGrid?.loaded) {
      const reg = String(tileGrid.currentRegion).padStart(3, '0');
      const fw  = tileGrid.currentFw;
      const x   = String(tileGrid.currentX).padStart(2, '0');
      const y   = String(tileGrid.currentY).padStart(2, '0');
      const regionLabel = tileGrid.regions?.find(r => r.region === tileGrid.currentRegion)?.label;
      const regDisplay = regionLabel || `Region${reg}`;
      if (label)  label.textContent     = `${regDisplay} · fw${fw}um · x${x} y${y}`;
      if (status) status.style.display  = '';
      if (minimap) minimap.style.display = '';
      if (tileNav) tileNav.style.display = '';
      _refreshMinimap();
      _updateTileNav(tileGrid);
    }
    // Clean up client atlas state when returning from atlas
    if (prevMode === 'atlas') _exitAtlasClientState();

  } else if (mode === 'atlas') {
    btnImage?.classList.remove('mode-btn-active');
    btnGrid?.classList.add('mode-btn-active');
    if (btnAtlas) btnAtlas.style.display  = 'none';
    if (atlasBar) atlasBar.style.display  = '';
    if (btnFit)   btnFit.style.display    = 'none';
    if (minimap)  minimap.style.display   = 'none';
    // Show tile-nav but only with region/fw dropdowns (hide dpad via CSS class)
    if (tileNav) { tileNav.style.display = ''; tileNav.classList.add('atlas-nav-mode'); }
    // Hide live histogram in atlas mode
    const histPanel = document.getElementById('live-hist-panel');
    if (histPanel) histPanel.style.display = 'none';
    if (tileGrid?.loaded) _updateTileNav(tileGrid, { regionOverride: atlas?.region, fwOverride: atlas?.fw });
    if (atlas) {
      const reg = String(atlas.region).padStart(3, '0');
      const regionLabel = tileGrid?.regions?.find(r => r.region === atlas.region)?.label;
      const regDisplay = regionLabel || `Region${reg}`;
      if (label)  label.textContent    = `${regDisplay} · fw${atlas.fw}um · Atlas`;
      if (status) status.style.display = '';
    }
    // Trigger atlas loading if this is a fresh transition
    if (prevMode !== 'atlas' && atlas && !_atlasManifest && !_atlasLoading) {
      _enterAtlasFromServer(atlas);
    }

  } else {
    // image mode
    btnImage?.classList.add('mode-btn-active');
    btnGrid?.classList.remove('mode-btn-active');
    if (status)  status.style.display  = 'none';
    if (minimap) minimap.style.display = 'none';
    if (tileNav) { tileNav.style.display = 'none'; tileNav.classList.remove('atlas-nav-mode'); }
    const histPanel = document.getElementById('live-hist-panel');
    if (histPanel) histPanel.style.display = '';
    if (btnAtlas) btnAtlas.style.display = 'none';
    if (atlasBar) atlasBar.style.display = 'none';
    if (btnFit)   btnFit.style.display   = '';
    if (prevMode === 'atlas') _exitAtlasClientState();
  }
}

async function _refreshMinimap() {
  const grid = document.getElementById('minimap-grid');
  if (!grid) return;
  try {
    const res  = await fetch(`${API}/camera/neighbors`);
    if (!res.ok) return;
    const data = await res.json();
    grid.innerHTML = '';
    for (const cell of data.grid) {
      const div = document.createElement('div');
      div.className = 'minimap-cell' + (cell.current ? ' minimap-current' : '') + (!cell.exists ? ' minimap-missing' : '');
      if (cell.exists) {
        const img = document.createElement('img');
        img.src = cell.urlPath + '?t=' + Date.now();
        img.alt = cell.filename;
        div.appendChild(img);
        // Click to navigate
        if (!cell.current) {
          div.style.cursor = 'pointer';
          div.title = cell.filename;
          div.addEventListener('click', () => {
            fetch(`${API}/camera/goto`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ x: cell.x, y: cell.y }),
            }).catch(() => {});
          });
        }
      }
      const lbl = document.createElement('span');
      lbl.className = 'minimap-label';
      lbl.textContent = `x${String(cell.x).padStart(2,'0')} y${String(cell.y).padStart(2,'0')}`;
      div.appendChild(lbl);
      grid.appendChild(div);
    }
  } catch (_) {}
}

// ---- Helpers ----

function _setupTileNav() {
  // Direction buttons → move API
  document.getElementById('nav-x-left') ?.addEventListener('click', () => _cameraMove('left'));
  document.getElementById('nav-x-right')?.addEventListener('click', () => _cameraMove('right'));
  document.getElementById('nav-y-up')   ?.addEventListener('click', () => _cameraMove('up'));
  document.getElementById('nav-y-down') ?.addEventListener('click', () => _cameraMove('down'));

  // Region dropdown → rebuild fw list then goto x=0 y=0 (or re-enter atlas)
  document.getElementById('nav-region')?.addEventListener('change', async (e) => {
    const regionVal = parseInt(e.target.value, 10);
    if (_currentUiMode === 'atlas') {
      await _atlasNavRegionChanged(regionVal);
    } else {
      await _navRegionChanged(regionVal);
    }
  });

  // fw dropdown → enter first tile of new fw (or re-enter atlas)
  document.getElementById('nav-fw')?.addEventListener('change', async (e) => {
    const fw = parseInt(e.target.value, 10);
    if (_currentUiMode === 'atlas') {
      const s = await fetch(`${API}/camera/state`).then(r => r.json());
      await _atlasReload(s.tileGrid.currentRegion, fw);
    } else {
      const s = await fetch(`${API}/camera/state`).then(r => r.json());
      const region = s.tileGrid.currentRegion;
      // Use /init which calls getFirstTileForRegion — works for non-zero-origin grids
      await fetch(`${API}/camera/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ region, fw }),
      }).catch(() => {});
    }
  });
}

async function _cameraMove(direction) {
  await fetch(`${API}/camera/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ direction }),
  }).catch(() => {});
}

async function _atlasNavRegionChanged(region) {
  // Update fw dropdown for new region, then reload atlas
  const state = await fetch(`${API}/camera/state`).then(r => r.json()).catch(() => null);
  if (!state) return;
  const fwOptions = state.tileGrid.regions.filter(r => r.region === region).map(r => r.fw);
  const fwSel = document.getElementById('nav-fw');
  if (fwSel && fwOptions.length) {
    fwSel.innerHTML = fwOptions.map(fw => `<option value="${fw}">${fw}um</option>`).join('');
    await _atlasReload(region, fwOptions[0]);
  }
}

async function _atlasReload(region, fw) {
  _exitAtlasClientState();
  try {
    const res = await fetch(`${API}/atlas/enter`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ region, fw }),
    });
    if (!res.ok) return;
    const manifest = await fetch(`${API}/atlas/manifest?region=${region}&fw=${fw}`).then(r => r.json());
    _doEnterAtlasMode(manifest);
    // Update status label
    const state = await fetch(`${API}/camera/state`).then(r => r.json()).catch(() => null);
    const regionLabel = state?.tileGrid?.regions?.find(r => r.region === region)?.label;
    const regDisplay = regionLabel || `Region${String(region).padStart(3,'0')}`;
    const label = document.getElementById('grid-status-label');
    if (label) label.textContent = `${regDisplay} · fw${fw}um · Atlas`;
  } catch (_) {}
}

async function _navRegionChanged(region) {
  // Get available fw values for this region from current tileGrid regions list
  const state = await fetch(`${API}/camera/state`).then(r => r.json()).catch(() => null);
  if (!state) return;
  const fwOptions = state.tileGrid.regions.filter(r => r.region === region).map(r => r.fw);
  const fwSel = document.getElementById('nav-fw');
  if (fwSel && fwOptions.length) {
    const curRegionObj = state.tileGrid.regions.find(r => r.region === region);
    fwSel.innerHTML = fwOptions.map(fw => {
      const fwLabel = (curRegionObj?.label && fw === 0) ? '-' : `${fw}um`;
      return `<option value="${fw}">${fwLabel}</option>`;
    }).join('');
    const fw = fwOptions[0];
    // Use /init which calls getFirstTileForRegion — works for non-zero-origin grids
    await fetch(`${API}/camera/init`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ region, fw }),
    }).catch(() => {});
  }
}

function _updateTileNav(tileGrid, { regionOverride, fwOverride } = {}) {
  if (!tileGrid?.loaded) return;

  const curRegion = regionOverride ?? tileGrid.currentRegion;
  const curFw     = fwOverride     ?? tileGrid.currentFw;

  // Populate region dropdown (unique regions, sorted)
  const regionSel = document.getElementById('nav-region');
  const fwSel     = document.getElementById('nav-fw');
  if (regionSel && tileGrid.regions) {
    const uniqueRegions = [...new Set(tileGrid.regions.map(r => r.region))].sort((a, b) => a - b);
    // Build a label map from the regions list (label may be null for standard regions)
    const labelMap = Object.fromEntries(tileGrid.regions.map(r => [r.region, r.label]));
    regionSel.innerHTML = uniqueRegions
      .map(r => {
        const lbl = labelMap[r] || `Region${String(r).padStart(3, '0')}`;
        return `<option value="${r}" ${r === curRegion ? 'selected' : ''}>${lbl}</option>`;
      })
      .join('');
  }

  // Populate fw dropdown for current region
  if (fwSel && tileGrid.regions) {
    const curRegionObj = tileGrid.regions.find(r => r.region === curRegion);
    const fwOptions = tileGrid.regions.filter(r => r.region === curRegion).map(r => r.fw);
    fwSel.innerHTML = fwOptions
      .map(fw => {
        const fwLabel = (curRegionObj?.label && fw === 0) ? '-' : `${fw}um`;
        return `<option value="${fw}" ${fw === curFw ? 'selected' : ''}>${fwLabel}</option>`;
      })
      .join('');
  }

  // Update x/y display
  const xyVal = document.getElementById('nav-xy-val');
  if (xyVal) xyVal.textContent = `${String(tileGrid.currentX).padStart(2,'0')},${String(tileGrid.currentY).padStart(2,'0')}`;
}

function _setupResizeHandles() {
  // Generic horizontal (col-resize) drag: drags between two sibling elements
  function makeHResize(handleId, prevEl, invert = false) {
    const handle = document.getElementById(handleId);
    if (!handle) return;
    handle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      handle.classList.add('dragging');
      const startX    = e.clientX;
      const startW    = prevEl.getBoundingClientRect().width;
      function onMove(e) {
        const delta = (e.clientX - startX) * (invert ? -1 : 1);
        const newW  = Math.max(0, startW + delta);
        prevEl.style.width = newW + 'px';
      }
      function onUp() {
        handle.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }

  // Generic vertical (row-resize) drag: resizes prevEl height inside sidebar
  function makeVResize(handleId, prevEl) {
    const handle = document.getElementById(handleId);
    if (!handle) return;
    handle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      handle.classList.add('dragging');
      const startY  = e.clientY;
      const startH  = prevEl.getBoundingClientRect().height;
      function onMove(e) {
        const delta = e.clientY - startY;
        const newH  = Math.max(60, startH + delta);
        prevEl.style.height = newH + 'px';
        prevEl.style.overflowY = 'auto';
      }
      function onUp() {
        handle.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }

  makeHResize('resize-toolbar', document.getElementById('toolbar'));
  makeHResize('resize-sidebar', document.getElementById('sidebar'), true);
  makeVResize('resize-minimap', document.getElementById('object-list'));

  // Agent panel — resize from top edge
  const agentHandle = document.getElementById('resize-agent');
  const agentPanel  = document.getElementById('agent-panel');
  if (agentHandle && agentPanel) {
    agentHandle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      agentHandle.classList.add('dragging');
      const startY = e.clientY;
      const startH = agentPanel.getBoundingClientRect().height;
      function onMove(e) {
        const delta = startY - e.clientY;
        const newH  = Math.max(44, startH + delta);
        agentPanel.style.height = newH + 'px';
      }
      function onUp() {
        agentHandle.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }
}

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

// ============================================================
// Agent panel — connects to agent_api.py on localhost:3001
// ============================================================
(function initAgentPanel() {
  const AGENT_URL = 'http://localhost:3001';

  const chatEl      = document.getElementById('agent-chat');
  const traceEl     = document.getElementById('agent-trace-body');
  const dotEl       = document.getElementById('agent-status-dot');
  const inputEl     = document.getElementById('agent-input');
  const btnStop     = document.getElementById('btn-agent-stop');
  const btnClear    = document.getElementById('btn-agent-clear');
  const btnKill     = document.getElementById('btn-agent-kill');
  const btnNoCode   = document.getElementById('btn-nocode');

  let _isProcessing = false;

  function _setProcessing(active) {
    _isProcessing = active;
    if (inputEl)  inputEl.disabled  = active;
    if (btnKill)  btnKill.disabled  = !active;
  }
  _setProcessing(false);  // init: kill disabled when idle
  const btnTracePrev     = document.getElementById('btn-trace-prev');
  const btnTraceNext     = document.getElementById('btn-trace-next');
  const btnTraceDownload = document.getElementById('btn-trace-download');
  const traceNavLbl      = document.getElementById('trace-nav-label');

  // ---- Trace history ----
  let _traces  = [];   // [{prompt, steps, reply}]
  let _traceIdx = -1;

  function _showTrace(idx) {
    if (!_traces.length) {
      traceEl.innerHTML = '';
      if (traceNavLbl) traceNavLbl.textContent = '—';
      return;
    }
    _traceIdx = Math.max(0, Math.min(idx, _traces.length - 1));
    if (traceNavLbl) traceNavLbl.textContent = `[${_traceIdx + 1} / ${_traces.length}]`;
    const { prompt, steps, reply } = _traces[_traceIdx];
    _renderTrace(prompt, steps, reply);
  }

  btnTracePrev?.addEventListener('click', () => _showTrace(_traceIdx - 1));
  btnTraceNext?.addEventListener('click', () => _showTrace(_traceIdx + 1));
  btnTraceDownload?.addEventListener('click', () => _downloadTrace(_traceIdx));

  function _formatTraceAsText(idx) {
    if (!_traces.length || idx < 0 || idx >= _traces.length) return null;
    const { prompt, steps, reply } = _traces[idx];
    const SEP  = '='.repeat(72);
    const SEP2 = '-'.repeat(72);
    const lines = [];
    lines.push('REASONING TRACE');
    lines.push(`Prompt ${idx + 1} of ${_traces.length}  |  Exported: ${new Date().toISOString()}`);
    lines.push(SEP);
    lines.push('');
    lines.push('INPUT');
    lines.push(SEP2);
    lines.push(prompt || '(empty)');
    lines.push('');
    if (steps && steps.length) {
      for (const step of steps) {
        lines.push(SEP2);
        lines.push(`STEP ${step.step}`);
        lines.push(SEP2);
        if (step.thinking) {
          lines.push('[THINKING]');
          lines.push(step.thinking);
          lines.push('');
        }
        for (const call of (step.calls || [])) {
          const label = call.action ? `${call.tool} / ${call.action}` : call.tool;
          lines.push(`[TOOL CALL]  ${label}  (${call.category})`);
          if (call.input_summary) lines.push(`  args:    ${call.input_summary}`);
          if (call.result) {
            if (call.result_is_json) {
              try {
                const pretty = JSON.stringify(JSON.parse(call.result), null, 2)
                  .split('\n').map(l => '    ' + l).join('\n');
                lines.push('  result:');
                lines.push(pretty);
              } catch (_) {
                lines.push(`  result:  ${call.result}`);
              }
            } else {
              lines.push(`  result:  ${call.result}`);
            }
          }
          lines.push('');
        }
      }
    }
    lines.push(SEP2);
    lines.push('OUTPUT');
    lines.push(SEP2);
    lines.push(reply || '(no reply)');
    lines.push('');
    lines.push(SEP);
    return lines.join('\n');
  }

  function _downloadTrace(idx) {
    const text = _formatTraceAsText(idx);
    if (!text) return;
    const blob = new Blob([text], { type: 'text/plain; charset=utf-8' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `trace_${idx + 1}_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  let _noCodeMode = false;
  const NO_CODE_PREFIX =
    'IMPORTANT CONSTRAINT: Do not write or execute any custom code to analyze images, ' +
    'extract information unrelated to the user\'s objective, or manipulate experiment ' +
    'results or test conditions.\n\n';

  btnNoCode?.addEventListener('click', () => {
    _noCodeMode = !_noCodeMode;
    btnNoCode.classList.toggle('nocode-on', _noCodeMode);
    btnNoCode.title = _noCodeMode
      ? 'No custom analysis code — click to allow again'
      : 'No custom analysis code: when ON, the agent is instructed not to write code that analyzes images or manipulates experiment results';
  });
  const fontSizeEl  = document.getElementById('agent-font-size');
  const agentBody   = document.getElementById('agent-body');
  const splitHandle = document.getElementById('resize-agent-split');
  const tracePane   = document.getElementById('agent-trace');

  // ---- Font size selector ----
  const _FS_KEY = 'agentFontSize';
  function _applyFontSize(size) {
    if (!agentBody) return;
    const n = Math.max(8, Math.min(32, parseInt(size, 10) || 12));
    agentBody.style.setProperty('--agent-chat-fs',  n + 'px');
    agentBody.style.setProperty('--agent-trace-fs', Math.max(8, n - 2) + 'px');
    agentBody.style.setProperty('--agent-input-fs', (n + 1) + 'px');
    localStorage.setItem(_FS_KEY, n);
  }
  const _savedFs = parseInt(localStorage.getItem(_FS_KEY), 10) || 12;
  if (fontSizeEl) fontSizeEl.value = _savedFs;
  _applyFontSize(_savedFs);
  fontSizeEl?.addEventListener('input', () => _applyFontSize(fontSizeEl.value));

  // ---- Resize: drag the splitter between chat and trace ----
  if (splitHandle && tracePane) {
    splitHandle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      splitHandle.classList.add('dragging');
      const startX = e.clientX;
      const startW = tracePane.getBoundingClientRect().width;
      function onMove(e) {
        const newW = Math.max(120, startW + (startX - e.clientX));
        tracePane.style.width = newW + 'px';
      }
      function onUp() {
        splitHandle.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }

  // ---- Helpers ----
  function _setRunning(running) {
    dotEl.className = 'agent-dot ' + (running ? 'agent-dot-running' : 'agent-dot-off');
    dotEl.title = running ? 'Agent ready' : 'Agent not connected';
    if (inputEl) inputEl.disabled = !running;
  }

  function _appendChat(role, text) {
    const wrap = document.createElement('div');
    wrap.className = 'chat-msg chat-msg-' + role;
    const lbl = document.createElement('div');
    lbl.className = 'chat-msg-label';
    lbl.textContent = role === 'user' ? 'You' : role === 'error' ? 'Error' : 'Agent';
    const body = document.createElement('div');
    body.className = 'chat-msg-text';
    body.textContent = text;
    wrap.appendChild(lbl);
    wrap.appendChild(body);
    chatEl.appendChild(wrap);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function _appendThinking() {
    const el = document.createElement('div');
    el.className = 'chat-thinking';
    el.textContent = '⏳ Thinking…';
    el.id = 'agent-thinking-indicator';
    chatEl.appendChild(el);
    chatEl.scrollTop = chatEl.scrollHeight;
    return el;
  }

  function _renderTrace(prompt, steps, reply) {
    traceEl.innerHTML = '';

    // ---- Input block ----
    if (prompt) {
      const inp = document.createElement('div');
      inp.className = 'trace-io-block trace-input-block';
      const lbl = document.createElement('div');
      lbl.className = 'trace-io-label';
      lbl.textContent = '\u2191 Input';
      const txt = document.createElement('div');
      txt.className = 'trace-io-text';
      txt.textContent = prompt;
      inp.appendChild(lbl);
      inp.appendChild(txt);
      traceEl.appendChild(inp);
    }

    if (!steps || !steps.length) {
      if (reply) _appendOutputBlock(reply);
      return;
    }

    const CATEGORY_LABEL = {
      draw: 'draw', adjust: 'adjust', navigate: 'nav',
      image: 'image', vision: 'vision', exec: 'exec', meta: 'meta',
    };

    for (const step of steps) {
      // Top-level step card
      const stepDiv = document.createElement('div');
      stepDiv.className = 'trace-step';

      // Step number
      const numEl = document.createElement('div');
      numEl.className = 'trace-step-num';
      numEl.textContent = `Step ${step.step}`;
      stepDiv.appendChild(numEl);

      // Thinking block — collapsible <details> with first 90 chars as summary
      if (step.thinking) {
        const details = document.createElement('details');
        details.className = 'trace-thinking';
        const summary = document.createElement('summary');
        const short = step.thinking.length > 90
          ? step.thinking.slice(0, 90) + '…'
          : step.thinking;
        summary.textContent = '💭 ' + short;
        details.appendChild(summary);
        if (step.thinking.length > 90) {
          const full = document.createElement('div');
          full.className = 'trace-thinking-full';
          full.textContent = step.thinking;
          details.appendChild(full);
        }
        stepDiv.appendChild(details);
      }

      // Tool calls
      for (const call of (step.calls || [])) {
        const callDiv = document.createElement('div');
        callDiv.className = 'trace-call';

        // Header: badge + action/tool label
        const hdr = document.createElement('div');
        hdr.className = 'trace-call-hdr';

        const badge = document.createElement('span');
        badge.className = `trace-badge tc-${call.category}`;
        badge.textContent = CATEGORY_LABEL[call.category] || call.category;
        hdr.appendChild(badge);

        const label = document.createElement('span');
        label.className = 'trace-call-label';
        label.textContent = call.action || call.tool;
        hdr.appendChild(label);

        callDiv.appendChild(hdr);

        // Args line
        if (call.input_summary) {
          const args = document.createElement('div');
          args.className = 'trace-call-args';
          args.textContent = call.input_summary;
          callDiv.appendChild(args);
        }

        // Result — collapsible
        if (call.result) {
          const res = document.createElement('details');
          res.className = 'trace-result';

          const resSummary = document.createElement('summary');
          resSummary.className = 'trace-result-summary';
          resSummary.textContent = '↩ result';
          res.appendChild(resSummary);

          const resBody = document.createElement('pre');
          resBody.className = 'trace-result-body';
          if (call.result_is_json) {
            try {
              resBody.textContent = JSON.stringify(JSON.parse(call.result), null, 2);
            } catch (_) {
              resBody.textContent = call.result;
            }
          } else {
            resBody.textContent = call.result;
          }
          res.appendChild(resBody);
          callDiv.appendChild(res);
        }

        stepDiv.appendChild(callDiv);
      }

      traceEl.appendChild(stepDiv);
    }

    // ---- Output block ----
    if (reply) _appendOutputBlock(reply);

    traceEl.scrollTop = traceEl.scrollHeight;
  }

  function _appendOutputBlock(reply) {
    const out = document.createElement('div');
    out.className = 'trace-io-block trace-output-block';
    const lbl = document.createElement('div');
    lbl.className = 'trace-io-label';
    lbl.textContent = '\u2193 Output';
    const txt = document.createElement('div');
    txt.className = 'trace-io-text';
    txt.textContent = reply;
    out.appendChild(lbl);
    out.appendChild(txt);
    traceEl.appendChild(out);
  }

  async function _pollStatus() {
    try {
      const res = await fetch(`${AGENT_URL}/status`);
      const { running } = await res.json();
      _setRunning(running);
    } catch (_) {
      _setRunning(false);
    }
  }

  async function _sendMessage(msg) {
    if (!msg.trim() || _isProcessing) return;
    const fullMsg = _noCodeMode ? NO_CODE_PREFIX + msg : msg;
    _appendChat('user', msg);
    if (inputEl) inputEl.value = '';
    _setProcessing(true);
    const thinking = _appendThinking();
    traceEl.innerHTML = '';

    try {
      const res = await fetch(`${AGENT_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: fullMsg }),
      });
      thinking.remove();
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      const { reply, trace } = await res.json();
      _appendChat('reply', reply);
      _traces.push({ prompt: msg, steps: trace, reply });
      _showTrace(_traces.length - 1);
    } catch (err) {
      thinking.remove();
      _appendChat('error', err.message);
    } finally {
      _setProcessing(false);
      inputEl?.focus();
    }
  }

  btnKill?.addEventListener('click', async () => {
    try {
      await fetch(`${AGENT_URL}/stop`, { method: 'POST' });
    } catch (_) {}
    _setProcessing(false);
    inputEl?.focus();
  });

  btnStop?.addEventListener('click', async () => {
    try {
      _setRunning(false);
      if (inputEl) inputEl.placeholder = 'Resetting…';
      await fetch(`${AGENT_URL}/reset`, { method: 'POST' });
      _appendChat('error', '[memory cleared — new conversation started]');
      traceEl.innerHTML = '';
      _traces  = [];
      _traceIdx = -1;
      if (traceNavLbl) traceNavLbl.textContent = '\u2014';
      _setRunning(true);
      if (inputEl) inputEl.placeholder = 'Ask the agent…';
    } catch (_) {
      _appendChat('error', '[could not reach agent_api — run: docker compose up -d]');
      _setRunning(false);
    }
  });

  btnClear?.addEventListener('click', () => {
    chatEl.innerHTML = '';
    traceEl.innerHTML = '';
    _traces  = [];
    _traceIdx = -1;
    if (traceNavLbl) traceNavLbl.textContent = '\u2014';
  });

  inputEl?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') _sendMessage(inputEl.value);
  });

  _pollStatus();
  setInterval(_pollStatus, 10000);
})();

// ---------------------------------------------------------------------------
// Segmentation (SAM2)
// ---------------------------------------------------------------------------

function _setupSegmentation() {
  const panel  = document.getElementById('segment-panel');
  const toggle = document.getElementById('tool-segment');

  toggle?.addEventListener('click', () => {
    const opening = !panel.classList.contains('open');
    panel.classList.toggle('open', opening);
    toggle.classList.toggle('active', opening);
    if (!opening) _clearSegmentation();
    fetch(`${API}/canvas/segmentation-enabled`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: opening }),
    }).catch(() => {});
  });

  ['seg-opt-centroids', 'seg-opt-bboxes', 'seg-opt-mask'].forEach(id => {
    document.getElementById(id)?.addEventListener('click', e =>
      e.currentTarget.classList.toggle('active'));
  });

  // Text toggle — also syncs state to server so the agent tool can respect it
  const textBtn = document.getElementById('seg-opt-text');
  textBtn?.addEventListener('click', () => {
    textBtn.classList.toggle('active');
    const enabled = textBtn.classList.contains('active');
    fetch(`${API}/canvas/segmentation-text-enabled`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    }).catch(() => {});
  });

  document.getElementById('btn-run-seg')?.addEventListener('click', _runSegmentation);
  document.getElementById('btn-clear-seg')?.addEventListener('click', _clearSegmentation);

  // Atlas coord toggle — controls whether atlas_state computes tile coordinates
  const coordBtn = document.getElementById('tool-atlas-coord');
  coordBtn?.addEventListener('click', () => {
    const enabled = !coordBtn.classList.contains('active');
    coordBtn.classList.toggle('active', enabled);
    fetch(`${API}/canvas/atlas-coord-enabled`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    }).catch(() => {});
  });
}

async function _runSegmentation() {
  const btn = document.getElementById('btn-run-seg');
  if (!btn) return;

  if (!_lastBgFilename) { alert('No image loaded.'); return; }

  const wantCentroids = document.getElementById('seg-opt-centroids')?.classList.contains('active');
  const wantBboxes    = document.getElementById('seg-opt-bboxes')?.classList.contains('active');
  const wantMask      = document.getElementById('seg-opt-mask')?.classList.contains('active');

  if (!wantCentroids && !wantBboxes && !wantMask) {
    alert('Enable at least one output (Centroids, BBoxes, or Mask).');
    return;
  }

  btn.disabled = true;
  const origLabel = btn.textContent;
  btn.textContent = '⏳ Running…';

  try {
    // Fetch the raw background tile and convert to base64
    const bgPath = _lastBgFilename.startsWith('/')
      ? _lastBgFilename
      : `/uploads/${_lastBgFilename}`;
    const imgResp = await fetch(bgPath);
    if (!imgResp.ok) throw new Error(`Could not fetch image: ${imgResp.status}`);
    const blob = await imgResp.blob();
    const image_b64 = await new Promise(res => {
      const r = new FileReader();
      r.onload = () => res(r.result);
      r.readAsDataURL(blob);
    });

    _clearSegmentation();

    const resp = await fetch('http://localhost:3001/segment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_b64, centroids: wantCentroids, bboxes: wantBboxes, mask: wantMask }),
    });
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || 'Segmentation failed');

    await _renderSegmentationData(data);

    btn.textContent = `✓ ${data.count} segments`;
    setTimeout(() => { btn.textContent = origLabel; }, 4000);

    // Sync to server so exports (PNG download + agent VLM) include the overlay
    const serverSeg = {};
    if (wantMask      && data.mask_png)   serverSeg.mask_png   = data.mask_png;
    if (wantBboxes    && data.bboxes)     serverSeg.bboxes     = data.bboxes;
    if (wantCentroids && data.centroids)  serverSeg.centroids  = data.centroids;
    // Pre-set key so the SSE bounce from the PUT doesn't re-render unnecessarily
    _lastServerSegKey = _segKey(serverSeg);
    try {
      const putRes = await fetch(`${API}/canvas/segmentation`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(serverSeg),
      });
      const putJson = await putRes.json();
      console.log('[seg] server sync:', putJson, '| mask_png bytes:', serverSeg.mask_png?.length ?? 0);
    } catch (e) {
      console.warn('[seg] server sync failed:', e.message);
    }

  } catch (err) {
    console.error('Segmentation error:', err);
    btn.textContent = '✗ Error';
    alert('Segmentation error: ' + err.message);
    setTimeout(() => { btn.textContent = origLabel; }, 3000);
  } finally {
    btn.disabled = false;
  }
}

// ---- Segmentation helpers ----

function _segKey(seg) {
  if (!seg) return null;
  return `${seg.mask_png?.length ?? 0}:${seg.bboxes?.length ?? 0}:${seg.centroids?.length ?? 0}`;
}

async function _renderSegmentationData(segData) {
  if (segData.mask_png) {
    await new Promise(resolve => {
      fabric.Image.fromURL(segData.mask_png, img => {
        img.set({ left: 0, top: 0, selectable: false, evented: false, _isSegment: true, opacity: 0.65 });
        fc.add(img);
        fc.sendToBack(img);
        resolve();
      });
    });
  }
  const bbColors = ['#89dceb','#a6e3a1','#fab387','#cba6f7','#f9e2af','#89b4fa','#f38ba8','#94e2d5'];
  if (segData.bboxes) {
    segData.bboxes.forEach(([x, y, w, h], i) => {
      fc.add(new fabric.Rect({
        left: x, top: y, width: w, height: h,
        fill: 'transparent',
        stroke: bbColors[i % bbColors.length],
        strokeWidth: 1,
        selectable: false, evented: false,
        _isSegment: true,
      }));
    });
  }
  if (segData.centroids) {
    segData.centroids.forEach(([cx, cy]) => {
      fc.add(new fabric.Circle({
        left: cx - 3, top: cy - 3, radius: 3,
        fill: '#f38ba8', stroke: '#ffffff', strokeWidth: 1,
        selectable: false, evented: false,
        _isSegment: true,
      }));
    });
  }
  fc.requestRenderAll();
}

function _clearSegmentationCanvas() {
  const segs = fc.getObjects().filter(o => o._isSegment);
  segs.forEach(o => fc.remove(o));
  if (segs.length) fc.requestRenderAll();
}

function _clearSegmentation() {
  _clearSegmentationCanvas();
  _lastServerSegKey = null;
  fetch(`${API}/canvas/segmentation`, { method: 'DELETE' }).catch(() => {});
}

// ---- CS4 Pattern Search ----

const CS4_API = 'http://localhost:3002';
let _cs4PatternB64 = null;
let _cs4AtlasB64   = null;

function _setupCs4Panel() {
  const dlg       = document.getElementById('cs4-dialog');
  const btnOpen   = document.getElementById('btn-cs4-open');
  const btnClose  = document.getElementById('btn-cs4-close');
  const btnRun    = document.getElementById('btn-cs4-run');
  const status    = document.getElementById('cs4-status');
  const resultBox = document.getElementById('cs4-result');
  if (!dlg) return;

  btnOpen?.addEventListener('click', () => dlg.showModal());
  btnClose?.addEventListener('click', () => dlg.close());

  // Wire up a generic drop zone to a b64 variable + preview
  function _wireDropZone(dropId, inputId, previewId, hintId, onLoad) {
    const drop    = document.getElementById(dropId);
    const input   = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const hint    = document.getElementById(hintId);
    if (!drop) return;
    drop.addEventListener('click', () => input?.click());
    drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('dragover'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
    drop.addEventListener('drop', e => {
      e.preventDefault();
      drop.classList.remove('dragover');
      const file = e.dataTransfer?.files?.[0];
      if (file) _readFile(file);
    });
    input?.addEventListener('change', () => {
      const file = input.files?.[0];
      if (file) _readFile(file);
      input.value = '';
    });
    function _readFile(file) {
      const reader = new FileReader();
      reader.onload = evt => {
        const dataUrl = evt.target.result;
        if (preview) { preview.src = dataUrl; preview.style.display = 'block'; }
        if (hint)    hint.style.display = 'none';
        onLoad(dataUrl.split(',')[1]);
        _cs4UpdateRunBtn();
      };
      reader.readAsDataURL(file);
    }
  }

  _wireDropZone('cs4-pattern-drop', 'cs4-pattern-input', 'cs4-pattern-preview', 'cs4-pattern-hint',
    b64 => { _cs4PatternB64 = b64; });
  _wireDropZone('cs4-atlas-drop',   'cs4-atlas-input',   'cs4-atlas-preview',   'cs4-atlas-hint',
    b64 => { _cs4AtlasB64 = b64; });

  function _cs4UpdateRunBtn() {
    if (btnRun) btnRun.disabled = !(_cs4PatternB64 && _cs4AtlasB64);
  }

  btnRun?.addEventListener('click', async () => {
    if (!_cs4PatternB64 || !_cs4AtlasB64) return;
    btnRun.disabled = true;
    status.style.display = 'block';
    resultBox.style.display = 'none';
    try {
      const resp = await fetch(`${CS4_API}/find-pattern`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patternBase64: _cs4PatternB64,
          searchBase64:  _cs4AtlasB64,
          searchMode:    'atlas_global_search',
        }),
      });
      status.style.display = 'none';
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        resultBox.innerHTML = `<span style="color:#f38ba8">⚠ Error: ${data.error || resp.statusText}</span>`;
        resultBox.style.display = 'block';
        return;
      }
      _cs4RenderResult(data);
    } catch (err) {
      status.style.display = 'none';
      resultBox.innerHTML = `<span style="color:#f38ba8">⚠ ${err.message}<br><small>Is <code>cs4_api.py</code> running on port 3002?</small></span>`;
      resultBox.style.display = 'block';
    } finally {
      _cs4UpdateRunBtn();
    }
  });
}

function _cs4RenderResult(data) {
  const resultBox = document.getElementById('cs4-result');
  if (!resultBox) return;

  const badge = data.found
    ? '<span class="cs4-badge cs4-badge-found">✓ FOUND</span>'
    : '<span class="cs4-badge cs4-badge-absent">✗ NOT FOUND</span>';
  const conf   = data.confidence != null ? `${Math.round(data.confidence * 100)}%` : '—';
  const tile   = data.tile ? `<b>${data.tile}</b>` : '<i style="color:#6c7086">—</i>';
  const bbox   = (data.found && Array.isArray(data.bbox) && data.bbox.length === 4)
    ? `<div class="cs4-result-row">Bbox (atlas px): <b>[${data.bbox.join(', ')}]</b></div>`
    : '';
  const reason = data.reason
    ? `<div class="cs4-result-reason">${data.reason}</div>`
    : '';
  const model  = data.model
    ? `<div style="color:#585b70;font-size:10px;margin-top:6px">model: ${data.model}</div>`
    : '';

  resultBox.innerHTML = `
    <div class="cs4-result-row">${badge} &nbsp;Confidence: <b>${conf}</b></div>
    <div class="cs4-result-row">Tile: ${tile}</div>
    ${bbox}${reason}${model}
  `;
  resultBox.style.display = 'block';
}

// ---- CS4 Save Pattern to GT ----

function _setupSavePatternPanel() {
  const dlg             = document.getElementById('save-pattern-dialog');
  const btnOpen         = document.getElementById('btn-save-pattern-open');
  const btnClose        = document.getElementById('btn-save-pattern-close');
  const btnSave         = document.getElementById('btn-sp-save');
  const drop            = document.getElementById('sp-image-drop');
  const input           = document.getElementById('sp-image-input');
  const preview         = document.getElementById('sp-image-preview');
  const hint            = document.getElementById('sp-image-hint');
  const idField         = document.getElementById('sp-sample-id');
  const regionSel       = document.getElementById('sp-source-region');
  const fwSel           = document.getElementById('sp-source-fw');
  const tileGrid        = document.getElementById('sp-tile-grid');
  const notesField      = document.getElementById('sp-notes');
  const status          = document.getElementById('sp-status');
  if (!dlg) return;

  let _imageDataUrl = null;
  let _allRegions   = [];   // [{ region, fw, tileCount, label }]
  let _selectedTiles = new Set(); // "y,x" strings

  // ---- Region/fw/tile population ----

  async function _populateRegions() {
    try {
      const s = await fetch(`${API}/camera/regions`).then(r => r.json());
      _allRegions = s?.regions || [];
      if (!_allRegions.length) throw new Error('no regions');
      const uniqueRegions = [...new Set(_allRegions.map(r => r.region))].sort((a, b) => a - b);
      const labelMap = Object.fromEntries(_allRegions.map(r => [r.region, r.label]));
      regionSel.innerHTML =
        '<option value="">— select region —</option>' +
        uniqueRegions.map(r => {
          const lbl = labelMap[r] || `Region${String(r).padStart(3, '0')}`;
          return `<option value="${r}">${lbl}</option>`;
        }).join('');
    } catch {
      regionSel.innerHTML = '<option value="">— regions unavailable —</option>';
    }
    _clearTileGrid();
  }

  function _clearTileGrid() {
    _selectedTiles.clear();
    if (tileGrid) tileGrid.innerHTML = '<span style="color:#6c7086;font-size:11px">Select a region first</span>';
    if (fwSel)    { fwSel.style.display = 'none'; fwSel.innerHTML = ''; }
  }

  async function _onRegionChange() {
    const region = parseInt(regionSel.value, 10);
    if (isNaN(region)) { _clearTileGrid(); return; }

    // Find fw options for this region
    const fwOptions = _allRegions.filter(r => r.region === region).map(r => r.fw).sort((a, b) => a - b);
    if (fwOptions.length === 0) { _clearTileGrid(); return; }

    if (fwOptions.length > 1) {
      fwSel.innerHTML = fwOptions.map(fw => `<option value="${fw}">${fw}um</option>`).join('');
      fwSel.style.display = '';
    } else {
      fwSel.innerHTML = `<option value="${fwOptions[0]}">${fwOptions[0]}um</option>`;
      fwSel.style.display = 'none';
    }
    await _loadTiles(region, fwOptions[0]);
  }

  async function _loadTiles(region, fw) {
    _selectedTiles.clear();
    if (tileGrid) tileGrid.innerHTML = '<span style="color:#6c7086;font-size:11px">Loading…</span>';
    try {
      const d = await fetch(`${API}/camera/tiles?region=${region}&fw=${fw}`).then(r => r.json());
      if (!d.tiles?.length) { tileGrid.innerHTML = '<span style="color:#6c7086;font-size:11px">No tiles found</span>'; return; }

      const maxX = Math.max(...d.tiles.map(t => t.x));
      const maxY = Math.max(...d.tiles.map(t => t.y));
      const cols = maxX + 1;
      const rows = maxY + 1;

      // Index existing tiles for quick lookup
      const tileSet = new Set(d.tiles.map(t => `${t.y},${t.x}`));

      tileGrid.innerHTML = '';
      tileGrid.style.display = 'grid';
      tileGrid.style.gridTemplateColumns = `repeat(${cols}, auto)`;
      tileGrid.style.gap = '3px';
      tileGrid.style.width = 'fit-content';

      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const key = `${y},${x}`;
          const cell = document.createElement('div');
          cell.style.gridColumn = x + 1;
          cell.style.gridRow    = y + 1;

          if (tileSet.has(key)) {
            const btn = document.createElement('button');
            btn.className = 'sp-tile-btn';
            btn.textContent = `(${x},${y})`;
            btn.dataset.key = key;
            btn.addEventListener('click', () => {
              if (_selectedTiles.has(key)) { _selectedTiles.delete(key); btn.classList.remove('active'); }
              else                         { _selectedTiles.add(key);    btn.classList.add('active'); }
            });
            cell.appendChild(btn);
          } else {
            // Empty cell — invisible placeholder keeps grid aligned
            cell.style.visibility = 'hidden';
            const placeholder = document.createElement('button');
            placeholder.className = 'sp-tile-btn';
            placeholder.textContent = `(${x},${y})`;
            placeholder.disabled = true;
            cell.appendChild(placeholder);
          }
          tileGrid.appendChild(cell);
        }
      }
    } catch {
      if (tileGrid) tileGrid.innerHTML = '<span style="color:#f38ba8;font-size:11px">Failed to load tiles</span>';
    }
  }

  regionSel?.addEventListener('change', _onRegionChange);
  fwSel?.addEventListener('change', () => {
    const region = parseInt(regionSel.value, 10);
    const fw     = parseInt(fwSel.value, 10);
    if (!isNaN(region) && !isNaN(fw)) _loadTiles(region, fw);
  });

  btnOpen?.addEventListener('click', () => { dlg.showModal(); _populateRegions(); });
  btnClose?.addEventListener('click', () => dlg.close());

  // ---- Drop zone ----
  drop?.addEventListener('click', () => input?.click());
  drop?.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('dragover'); });
  drop?.addEventListener('dragleave', () => drop.classList.remove('dragover'));
  drop?.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('dragover');
    const file = e.dataTransfer?.files?.[0];
    if (file) _loadFile(file);
  });
  input?.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file) _loadFile(file);
    input.value = '';
  });

  function _loadFile(file) {
    const reader = new FileReader();
    reader.onload = evt => {
      _imageDataUrl = evt.target.result;
      if (preview) { preview.src = _imageDataUrl; preview.style.display = 'block'; }
      if (hint)    hint.style.display = 'none';
      if (idField && !idField.value) idField.value = file.name.replace(/\.[^.]+$/, '');
      _updateSaveBtn();
    };
    reader.readAsDataURL(file);
  }

  function _updateSaveBtn() {
    if (btnSave) btnSave.disabled = !_imageDataUrl;
  }

  // ---- Save ----
  btnSave?.addEventListener('click', async () => {
    if (!_imageDataUrl) return;
    btnSave.disabled = true;
    if (status) { status.textContent = '⏳ Saving…'; status.style.display = 'block'; }

    // Build tile strings from selected toggle buttons: "(x,y)"
    const tilesStr = [..._selectedTiles].map(k => { const [y, x] = k.split(','); return `(${x},${y})`; }).join(', ');

    // Resolve human-readable region label
    const regionNum = parseInt(regionSel?.value, 10);
    const regionEntry = _allRegions.find(r => r.region === regionNum);
    const sourceRegionLabel = isNaN(regionNum) ? '' :
      (regionEntry?.label || `Region${String(regionNum).padStart(3, '0')}`);
    try {
      const res = await fetch(`${API}/cs4/save-pattern`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample_id:      idField?.value?.trim() || '',
          image_data_url: _imageDataUrl,
          source_region:  sourceRegionLabel,
          tiles:          tilesStr,
          notes:          notesField?.value?.trim() || '',
        }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        if (status) { status.textContent = '⚠ ' + (data.error || res.statusText); status.style.color = '#f38ba8'; }
        return;
      }
      const savedTiles = data.gt_tiles?.length ? data.gt_tiles.join(', ') : 'none';
      if (status) {
        status.textContent = `✓ Saved "${data.sample_id}" — tiles: [${savedTiles}]`;
        status.style.color = '#a6e3a1';
      }
      // Reset
      _imageDataUrl = null;
      if (preview) { preview.style.display = 'none'; preview.src = ''; }
      if (hint)    hint.style.display = '';
      if (idField)     idField.value = '';
      if (notesField)  notesField.value = '';
      regionSel.selectedIndex = 0;
      _clearTileGrid();
    } catch (err) {
      if (status) { status.textContent = '⚠ ' + err.message; status.style.color = '#f38ba8'; }
    } finally {
      _updateSaveBtn();
    }
  });
}
