// server/renderer.js — server-side PNG rendering with node-canvas
const { createCanvas, loadImage } = require('canvas');
const path = require('path');
const fs = require('fs');

const UPLOADS_DIR = path.join(__dirname, '..', 'uploads');

async function renderToPng(state) {
  const { canvas: meta, objects, filters } = state;
  const width = meta.width || 1200;
  const height = meta.height || 800;

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  // White background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  // Background image
  if (meta.backgroundImage) {
    const imgPath = path.join(UPLOADS_DIR, meta.backgroundImage);
    if (fs.existsSync(imgPath)) {
      try {
        const img = await loadImage(imgPath);
        ctx.drawImage(img, 0, 0, width, height);
      } catch (e) {
        console.error('Background image load failed:', e.message);
      }
    }
  }

  // Apply brightness / contrast / saturation filters to the background pixels
  // before drawing annotations, so annotations remain unaffected.
  const brightness  = (filters?.brightness  ?? 100) / 100;
  const contrast    = (filters?.contrast    ?? 100) / 100;
  const saturation  = (filters?.saturation  ?? 100) / 100;
  const needsFilter = brightness !== 1 || contrast !== 1 || saturation !== 1;
  if (needsFilter) {
    _applyPixelFilters(ctx, width, height, brightness, contrast, saturation);
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

// ---- Pixel-level filter implementation ----
// Simulates CSS brightness/contrast/saturation on an existing canvas context.
// brightness: 1.0 = normal, <1 darker, >1 brighter
// contrast:   1.0 = normal, 0 = grey, >1 more contrast
// saturation: 1.0 = normal, 0 = greyscale, >1 more vivid
function _applyPixelFilters(ctx, width, height, brightness, contrast, saturation) {
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  // Contrast intercept: shift so mid-grey stays mid-grey
  const cIntercept = 127 * (1 - contrast);

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i], g = data[i + 1], b = data[i + 2];

    // Brightness (multiply)
    r *= brightness; g *= brightness; b *= brightness;

    // Saturation (interpolate toward luminance)
    if (saturation !== 1) {
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      r = lum + saturation * (r - lum);
      g = lum + saturation * (g - lum);
      b = lum + saturation * (b - lum);
    }

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
