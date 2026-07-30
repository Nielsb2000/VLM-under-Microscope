// server/renderer.js - server-side PNG rendering with node-canvas
const { createCanvas, loadImage } = require('canvas');
const path = require('path');
const fs = require('fs');
const sharp = require('sharp');
const tileGrid = require('./tileGrid');

const UPLOADS_DIR = path.join(__dirname, '..', 'uploads');
const TILE_DATASETS_DIR   = '/app/tile-datasets';
const GRID_PAPER_DIR      = '/app/grid-paper-tiles';
const LABELED_DATASET_DIR = '/app/dataset-labeled';
const UNLABELED_DIR       = '/app/dataset-unlabeled';

function _resolveBackgroundPath(bg) {
  if (!bg) return null;
  if (bg.startsWith('/tile-assets/')) {
    return path.join(TILE_DATASETS_DIR, path.basename(bg));
  }
  if (bg.startsWith('/grid-paper-assets/')) {
    return path.join(GRID_PAPER_DIR, path.basename(bg));
  }
  if (bg.startsWith('/dataset-assets/')) {
    return path.join(LABELED_DATASET_DIR, bg.slice('/dataset-assets/'.length));
  }
  if (bg.startsWith('/unlabeled-assets/')) {
    return path.join(UNLABELED_DIR, bg.slice('/unlabeled-assets/'.length));
  }
  return path.join(UPLOADS_DIR, path.basename(bg));
}

async function _loadBackgroundImage(bg) {
  const imgPath = _resolveBackgroundPath(bg);
  if (!imgPath || !fs.existsSync(imgPath)) return null;
  const ext = path.extname(imgPath).toLowerCase();
  if (ext === '.tif' || ext === '.tiff') {
    // node-canvas can't decode LZW TIFF - convert via sharp first
    const pngBuf = await sharp(imgPath).png().toBuffer();
    return loadImage(pngBuf);
  }
  return loadImage(imgPath);
}

async function renderToPng(state) {
  const { canvas: meta, objects, filters, uiMode, atlas, viewport, segmentation, atlasOverlay } = state;

  // Atlas mode: composite tiles at the current viewport
  if (uiMode === 'atlas' && atlas) {
    return renderAtlasToPng(meta, objects, filters, atlas, viewport, atlasOverlay || {});
  }

  const width = meta.width || 1200;
  const height = meta.height || 800;

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  // White background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  // Background image
  if (meta.backgroundImage) {
    try {
      const img = await _loadBackgroundImage(meta.backgroundImage);
      if (img) ctx.drawImage(img, 0, 0, width, height);
    } catch (e) {
      console.error('Background image load failed:', e.message);
    }
  }

  // Apply brightness / contrast / saturation filters to the background pixels
  // before drawing annotations, so annotations remain unaffected.
  const brightness  = (filters?.brightness  ?? 100) / 100;
  const contrast    = (filters?.contrast    ?? 100) / 100;
  const needsFilter = brightness !== 1 || contrast !== 1;
  if (needsFilter) {
    _applyPixelFilters(ctx, width, height, brightness, contrast);
  }

  // Segmentation overlay - drawn above background/filters, below annotation objects
  if (segmentation) {
    const seg = segmentation;
    const bbColors = ['#89dceb','#a6e3a1','#fab387','#cba6f7','#f9e2af','#89b4fa','#f38ba8','#94e2d5'];

    if (seg.mask_png) {
      try {
        const buf = Buffer.from(seg.mask_png.split(',')[1], 'base64');
        const maskImg = await loadImage(buf);
        ctx.globalAlpha = 0.65;
        ctx.drawImage(maskImg, 0, 0, width, height);
        ctx.globalAlpha = 1;
      } catch (e) { console.error('Seg mask render failed:', e.message); }
    }

    if (seg.bboxes && seg.bboxes.length) {
      ctx.lineWidth = 1;
      seg.bboxes.forEach(([x, y, w, h], i) => {
        ctx.strokeStyle = bbColors[i % bbColors.length];
        ctx.strokeRect(x, y, w, h);
      });
    }

    if (seg.centroids && seg.centroids.length) {
      ctx.fillStyle = '#f38ba8';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      seg.centroids.forEach(([cx, cy]) => {
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    }
  }

  // Draw objects in insertion order
  for (const obj of objects) {
    await drawOne(ctx, obj);
  }

  return canvas.toBuffer('image/png');
}

async function drawOne(ctx, obj) {
  ctx.save();

  const stroke = obj.stroke || '#ff0000';
  const fill = obj.fill && obj.fill !== 'transparent' ? obj.fill : null;
  const lw = obj.strokeWidth || 2;

  switch (obj.type) {
    case 'rect': {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.rect(obj.x, obj.y, obj.width, obj.height);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      ctx.stroke();
      break;
    }

    case 'ellipse': {
      const cx = obj.cx != null ? obj.cx : (obj.x || 0) + (obj.rx || 50);
      const cy = obj.cy != null ? obj.cy : (obj.y || 0) + (obj.ry || 30);
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.ellipse(cx, cy, obj.rx || 50, obj.ry || 30, 0, 0, Math.PI * 2);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      ctx.stroke();
      break;
    }

    case 'arrow': {
      drawArrow(ctx, obj.x1, obj.y1, obj.x2, obj.y2, stroke, lw);
      break;
    }

    case 'dot': {
      const r = obj.radius || 6;
      ctx.fillStyle = obj.fill || stroke;
      ctx.beginPath();
      ctx.arc(obj.cx, obj.cy, r, 0, Math.PI * 2);
      ctx.fill();
      break;
    }

    case 'line': {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.moveTo(obj.x1, obj.y1);
      ctx.lineTo(obj.x2, obj.y2);
      ctx.stroke();
      break;
    }

    case 'text': {
      const fs = obj.fontSize || 18;
      ctx.font = `${fs}px ${obj.fontFamily || 'sans-serif'}`;
      ctx.fillStyle = obj.fill || '#000000';
      ctx.fillText(obj.text || '', obj.x, obj.y + fs);
      break;
    }

    case 'freehand': {
      if (obj.path) {
        ctx.save();

        // Important:
        // Historical Fabric freehand paths are stored in export-ready coordinates.
        // Only apply left/top for explicitly local paths.
        if (obj.coordMode === 'local') {
          ctx.translate(obj.left || 0, obj.top || 0);
        }

        ctx.strokeStyle = stroke;
        ctx.lineWidth = lw;
        ctx.fillStyle = 'transparent';
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        drawSvgPath(ctx, obj.path);
        ctx.stroke();
        ctx.restore();
      }
      break;
    }}

  // Optional label
  if (obj.label) {
    const lx = obj.x ?? obj.cx ?? obj.x1 ?? 0;
    const ly = (obj.y ?? obj.cy ?? obj.y1 ?? 0) - 4;
    ctx.font = '12px sans-serif';
    ctx.fillStyle = stroke;
    ctx.fillText(obj.label, lx, Math.max(ly, 12));
  }

  ctx.restore();
}

function drawArrow(ctx, x1, y1, x2, y2, color, lw) {
  const headLen = Math.max(12, lw * 5);
  const angle = Math.atan2(y2 - y1, x2 - x1);

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lw;

  // Shaft
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  // Arrowhead
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - headLen * Math.cos(angle - Math.PI / 6),
    y2 - headLen * Math.sin(angle - Math.PI / 6),
  );
  ctx.lineTo(
    x2 - headLen * Math.cos(angle + Math.PI / 6),
    y2 - headLen * Math.sin(angle + Math.PI / 6),
  );
  ctx.closePath();
  ctx.fill();
}

// Minimal SVG path parser (M, L, Q, C, Z - the subset Fabric.js generates)
function drawSvgPath(ctx, pathStr) {
  // Fabric.js joinPath output: "M x y Q cx cy x y ..." or array-like repr
  // Tokenise on spaces and commas
  const tok = pathStr.trim().split(/[\s,]+/);
  let i = 0;

  function num() { return parseFloat(tok[i++]); }

  while (i < tok.length) {
    const cmd = tok[i++];
    if (!cmd) continue;
    switch (cmd.toUpperCase()) {
      case 'M': ctx.moveTo(num(), num()); break;
      case 'L': ctx.lineTo(num(), num()); break;
      case 'H': ctx.lineTo(num(), /* keep y */ 0 /* approximate */); break;
      case 'V': ctx.lineTo(0, num()); break;
      case 'Q': ctx.quadraticCurveTo(num(), num(), num(), num()); break;
      case 'C': ctx.bezierCurveTo(num(), num(), num(), num(), num(), num()); break;
      case 'Z': case 'z': ctx.closePath(); break;
      default:
        // Unknown command - skip a number pair to stay in sync
        break;
    }
  }
}

module.exports = { renderToPng };

// ---- Atlas viewport rendering ----
// Composites visible tiles using sharp, then draws annotations via node-canvas.
// Annotations are scaled inversely with zoom so they remain readable at any zoom level.
// atlasOverlay: { grid: bool, labels: bool } - draw tile-boundary grid / tile-ID labels.
async function renderAtlasToPng(meta, objects, filters, atlas, viewport, atlasOverlay = {}) {
  const { tileWidth, tileHeight, cols, rows, region, fw } = atlas;

  // Export the full atlas at its real stitched resolution.
  // Do NOT use the HTML/display canvas dimensions here.
  const atlasW = Math.round(tileWidth * cols);
  const atlasH = Math.round(tileHeight * rows);

  // Build sharp composites for every tile at native atlas resolution.
  const composites = [];

  for (let ty = 0; ty < rows; ty++) {
    for (let tx = 0; tx < cols; tx++) {
      const tile = tileGrid.getTile(region, fw, ty, tx);
      if (!tile) continue;

      try {
        const tileBuf = await sharp(tile.absolutePath)
          .resize(tileWidth, tileHeight)
          .png()
          .toBuffer();

        composites.push({
          input: tileBuf,
          left: Math.round(tx * tileWidth),
          top: Math.round(ty * tileHeight),
        });
      } catch (e) {
        console.warn(`[atlas export] skipping tile ${tile.filename}:`, e.message);
      }
    }
  }

  // Composite all tiles onto the full atlas canvas.
  // Use white instead of dark so any genuinely missing tiles do not become black borders.
  let bgBuf;
  try {
    bgBuf = await sharp({
      create: {
        width: atlasW,
        height: atlasH,
        channels: 3,
        background: { r: 255, g: 255, b: 255 },
      },
    })
      .composite(composites)
      .png()
      .toBuffer();
  } catch (e) {
    bgBuf = await sharp({
      create: {
        width: atlasW,
        height: atlasH,
        channels: 3,
        background: { r: 255, g: 255, b: 255 },
      },
    })
      .png()
      .toBuffer();
  }

  // Draw annotations on top at atlas/native coordinates.
  const canvas = createCanvas(atlasW, atlasH);
  const ctx = canvas.getContext('2d');

  const bgImg = await loadImage(bgBuf);
  ctx.drawImage(bgImg, 0, 0);

  // Apply brightness/contrast to the full atlas background.
  const brightness = (filters?.brightness ?? 100) / 100;
  const contrast = (filters?.contrast ?? 100) / 100;

  if (brightness !== 1 || contrast !== 1) {
    _applyPixelFilters(ctx, atlasW, atlasH, brightness, contrast);
  }

  // Optional overlays now use atlas-native coordinates.
  if (atlasOverlay.grid) {
    _drawAtlasGrid(ctx, atlas, 1, 0, 0, atlasW, atlasH, atlasOverlay);
  }

  if (atlasOverlay.labels) {
    _drawAtlasLabels(ctx, atlas, 1, 0, 0, atlasW, atlasH, atlasOverlay);
  }

  // No viewport transform here.
  // Objects should already be stored in atlas coordinates.
  for (const obj of objects) {
    await drawOne(ctx, obj);
  }

  return canvas.toBuffer('image/png');
}

// Draw analysis-grid boundary lines in display space.
// gridLevel controls virtual atlas subdivision:
//   L0 => subdivision 1: original SEM acquisition-tile grid
//   L1 => subdivision 2: each original tile split into 2 x 2 cells
//   L2 => subdivision 4: each original tile split into 4 x 4 cells
function _drawAtlasGrid(ctx, atlas, zoom, panX, panY, dispW, dispH, atlasOverlay = {}) {
  const { tileWidth, tileHeight, cols, rows } = atlas;

  const gridLevel = Math.max(0, Math.floor(Number(atlasOverlay.gridLevel ?? 0) || 0));
  const subdivision = Math.max(
    1,
    Math.floor(Number(atlasOverlay.subdivision ?? Math.pow(2, gridLevel)) || 1),
  );

  const cellWidth = tileWidth / subdivision;
  const cellHeight = tileHeight / subdivision;
  const effectiveCols = cols * subdivision;
  const effectiveRows = rows * subdivision;

  const gridLineWidth = Math.max(
    1,
    Number(atlasOverlay.gridLineWidth ?? Math.max(1, zoom * 0.5)) || 10,
  );
  const gridAlpha = Math.max(
    0,
    Math.min(1, Number(atlasOverlay.gridLineAlpha ?? 0.85)),
  );

  ctx.save();
  ctx.strokeStyle = `rgba(255, 255, 100, ${gridAlpha})`;  // yellow, semi-transparent
  ctx.lineWidth   = gridLineWidth;
  ctx.lineCap     = 'butt';
  ctx.lineJoin    = 'miter';

  // Vertical lines at atlas x = gx * cellWidth
  for (let gx = 0; gx <= effectiveCols; gx++) {
    const dispX = Math.round(gx * cellWidth * zoom - panX);
    if (dispX < -gridLineWidth || dispX > dispW + gridLineWidth) continue;
    ctx.beginPath();
    ctx.moveTo(dispX, 0);
    ctx.lineTo(dispX, dispH);
    ctx.stroke();
  }

  // Horizontal lines at atlas y = gy * cellHeight
  for (let gy = 0; gy <= effectiveRows; gy++) {
    const dispY = Math.round(gy * cellHeight * zoom - panY);
    if (dispY < -gridLineWidth || dispY > dispH + gridLineWidth) continue;
    ctx.beginPath();
    ctx.moveTo(0,     dispY);
    ctx.lineTo(dispW, dispY);
    ctx.stroke();
  }

  ctx.restore();
}

// Draw analysis-grid cell labels (e.g. (0,0), (1,2)) centred in each visible cell.
// For L1/L2 these are GLOBAL virtual-cell coordinates across the whole atlas,
// not local coordinates within an original SEM acquisition tile.
function _drawAtlasLabels(ctx, atlas, zoom, panX, panY, dispW, dispH, atlasOverlay = {}) {
  const { tileWidth, tileHeight, cols, rows } = atlas;

  const gridLevel = Math.max(0, Math.floor(Number(atlasOverlay.gridLevel ?? 0) || 0));
  const subdivision = Math.max(
    1,
    Math.floor(Number(atlasOverlay.subdivision ?? Math.pow(2, gridLevel)) || 1),
  );

  const cellWidth = tileWidth / subdivision;
  const cellHeight = tileHeight / subdivision;
  const effectiveCols = cols * subdivision;
  const effectiveRows = rows * subdivision;

  const scaledCellW = cellWidth  * zoom;
  const scaledCellH = cellHeight * zoom;
  const defaultFontSize = Math.max(8, Math.min(scaledCellW, scaledCellH) * 0.22);
  const fontSize = Math.max(
    8,
    Number(atlasOverlay.labelFontSize ?? defaultFontSize) || defaultFontSize,
  );
  const pad = Math.max(
    0,
    Number(atlasOverlay.labelBoxPadding ?? fontSize * 0.3) || fontSize * 0.3,
  );
  const boxAlpha = Math.max(
    0,
    Math.min(1, Number(atlasOverlay.labelBoxAlpha ?? 0.70)),
  );

  ctx.save();
  ctx.font         = `bold ${fontSize}px sans-serif`;
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';

  for (let gy = 0; gy < effectiveRows; gy++) {
    for (let gx = 0; gx < effectiveCols; gx++) {
      // Centre of this analysis-grid cell in display space
      const cx = (gx + 0.5) * cellWidth  * zoom - panX;
      const cy = (gy + 0.5) * cellHeight * zoom - panY;

      // Skip cells whose centre is off-screen (with a generous margin)
      if (cx < -scaledCellW || cx > dispW + scaledCellW) continue;
      if (cy < -scaledCellH || cy > dispH + scaledCellH) continue;

      const label = `(${gx},${gy})`;

      // Dark semi-transparent background pill for readability
      const bw  = ctx.measureText(label).width + pad * 2;
      const bh  = fontSize + pad * 1.5;
      ctx.fillStyle = `rgba(0, 0, 0, ${boxAlpha})`;
      _roundRect(ctx, cx - bw / 2, cy - bh / 2, bw, bh, fontSize * 0.2);
      ctx.fill();

      // Label text
      ctx.fillStyle = 'rgba(255, 255, 100, 0.98)';
      ctx.fillText(label, cx, cy);
    }
  }

  ctx.restore();
}

// Helper: draw a rounded-rectangle path (no stroke/fill call - caller does that).
function _roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y,     x + w, y + r,     r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x,     y + h, x,     y + h - r, r);
  ctx.lineTo(x,     y + r);
  ctx.arcTo(x,     y,     x + r, y,         r);
  ctx.closePath();
}

// Scale size properties of an annotation inversely with zoom so they read consistently
function _scaleAnnotation(obj, factor) {
  const s = { ...obj };
  if (s.strokeWidth != null) s.strokeWidth = Math.max(1, s.strokeWidth * factor);
  if (s.radius      != null) s.radius      = Math.max(3, s.radius      * factor);
  if (s.fontSize    != null) s.fontSize    = Math.max(8, s.fontSize    * factor);
  return s;
}

// ---- Pixel-level filter implementation ----
// Simulates CSS brightness/contrast/saturation on an existing canvas context.
// brightness: 1.0 = normal, <1 darker, >1 brighter
// contrast:   1.0 = normal, 0 = grey, >1 more contrast
// saturation: 1.0 = normal, 0 = greyscale, >1 more vivid
function _applyPixelFilters(ctx, width, height, brightness, contrast) {
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  // Contrast intercept: shift so mid-grey stays mid-grey
  const cIntercept = 127 * (1 - contrast);

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i], g = data[i + 1], b = data[i + 2];

    // Brightness (multiply)
    r *= brightness; g *= brightness; b *= brightness;

    // Contrast (linear scale around mid-grey)
    if (contrast !== 1) {
      r = r * contrast + cIntercept;
      g = g * contrast + cIntercept;
      b = b * contrast + cIntercept;
    }

    data[i]     = Math.max(0, Math.min(255, r));
    data[i + 1] = Math.max(0, Math.min(255, g));
    data[i + 2] = Math.max(0, Math.min(255, b));
    // alpha unchanged
  }
  ctx.putImageData(imageData, 0, 0);
}
