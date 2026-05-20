// server/renderer.js — server-side PNG rendering with node-canvas
const { createCanvas, loadImage } = require('canvas');
const path = require('path');
const fs = require('fs');
const sharp = require('sharp');
const tileGrid = require('./tileGrid');

const UPLOADS_DIR = path.join(__dirname, '..', 'uploads');
const TILE_DATASETS_DIR   = '/app/tile-datasets';
const LABELED_DATASET_DIR = '/app/dataset-labeled';
const UNLABELED_DIR       = '/app/dataset-unlabeled';

function _resolveBackgroundPath(bg) {
  if (!bg) return null;
  if (bg.startsWith('/tile-assets/')) {
    return path.join(TILE_DATASETS_DIR, path.basename(bg));
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
    // node-canvas can't decode LZW TIFF — convert via sharp first
    const pngBuf = await sharp(imgPath).png().toBuffer();
    return loadImage(pngBuf);
  }
  return loadImage(imgPath);
}

async function renderToPng(state) {
  const { canvas: meta, objects, filters, uiMode, atlas, viewport, segmentation } = state;

  // Atlas mode: composite tiles at the current viewport
  if (uiMode === 'atlas' && atlas) {
    return renderAtlasToPng(meta, objects, filters, atlas, viewport);
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

  // Segmentation overlay — drawn above background/filters, below annotation objects
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
        ctx.translate(obj.left || 0, obj.top || 0);
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
    }

    default:
      break;
  }

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

// Minimal SVG path parser (M, L, Q, C, Z — the subset Fabric.js generates)
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
        // Unknown command — skip a number pair to stay in sync
        break;
    }
  }
}

module.exports = { renderToPng };

// ---- Atlas viewport rendering ----
// Composites visible tiles using sharp, then draws annotations via node-canvas.
// Annotations are scaled inversely with zoom so they remain readable at any zoom level.
async function renderAtlasToPng(meta, objects, filters, atlas, viewport) {
  const { tileWidth, tileHeight, cols, rows, region, fw } = atlas;
  const zoom = (viewport?.zoom) || 0.1;
  const panX = (viewport?.panX) || 0;
  const panY = (viewport?.panY) || 0;

  // Display canvas dimensions (the HTML canvas element size)
  const dispW = meta.width  || 1920;
  const dispH = meta.height || 1200;

  // Atlas pixel coords at the top-left of the display canvas
  const atlasLeft   = panX / zoom;
  const atlasTop    = panY / zoom;
  const atlasRight  = atlasLeft  + dispW / zoom;
  const atlasBottom = atlasTop   + dispH / zoom;

  // Which tile columns/rows are (partially) visible?
  const tileX0 = Math.max(0, Math.floor(atlasLeft   / tileWidth));
  const tileX1 = Math.min(cols - 1, Math.ceil(atlasRight  / tileWidth)  - 1);
  const tileY0 = Math.max(0, Math.floor(atlasTop    / tileHeight));
  const tileY1 = Math.min(rows - 1, Math.ceil(atlasBottom / tileHeight) - 1);

  // Each tile scaled to display zoom
  const scaledW = Math.max(1, Math.round(tileWidth  * zoom));
  const scaledH = Math.max(1, Math.round(tileHeight * zoom));

  // Build sharp composites for each visible tile
  const composites = [];
  for (let ty = tileY0; ty <= tileY1; ty++) {
    for (let tx = tileX0; tx <= tileX1; tx++) {
      const tile = tileGrid.getTile(region, fw, ty, tx);
      if (!tile) continue;
      try {
        const tileBuf = await sharp(tile.absolutePath)
          .resize(scaledW, scaledH)
          .png()
          .toBuffer();
        composites.push({
          input: tileBuf,
          left: Math.round(tx * tileWidth  * zoom - panX),
          top:  Math.round(ty * tileHeight * zoom - panY),
        });
      } catch (e) {
        console.warn(`[atlas render] skipping tile ${tile.filename}:`, e.message);
      }
    }
  }

  // Composite all tiles onto a dark background
  let bgBuf;
  try {
    bgBuf = await sharp({
      create: { width: dispW, height: dispH, channels: 3, background: { r: 30, g: 30, b: 30 } },
    })
      .composite(composites)
      .png()
      .toBuffer();
  } catch (e) {
    // If no composites, just return a dark canvas
    bgBuf = await sharp({
      create: { width: dispW, height: dispH, channels: 3, background: { r: 30, g: 30, b: 30 } },
    }).png().toBuffer();
  }

  // Draw annotations on top using node-canvas with viewport transform
  const canvas = createCanvas(dispW, dispH);
  const ctx = canvas.getContext('2d');

  const bgImg = await loadImage(bgBuf);
  ctx.drawImage(bgImg, 0, 0);

  // Apply brightness/contrast to tiles (approximate — applied to whole composite)
  const brightness = (filters?.brightness ?? 100) / 100;
  const contrast   = (filters?.contrast   ?? 100) / 100;
  if (brightness !== 1 || contrast !== 1) {
    _applyPixelFilters(ctx, dispW, dispH, brightness, contrast);
  }

  // Draw annotations with viewport transform.
  // Scale strokeWidth/radius/fontSize by 1/zoom so they appear at a consistent visual size.
  ctx.save();
  ctx.setTransform(zoom, 0, 0, zoom, -panX, -panY);
  for (const obj of objects) {
    const scaled = _scaleAnnotation(obj, 1 / zoom);
    await drawOne(ctx, scaled);
  }
  ctx.restore();

  return canvas.toBuffer('image/png');
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
