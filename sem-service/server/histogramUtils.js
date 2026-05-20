// server/histogramUtils.js — shared histogram utilities
'use strict';

const path = require('path');
const fs   = require('fs');
const sharp = require('sharp');
const { createCanvas } = require('canvas');

const UPLOADS_DIR     = path.join(__dirname, '..', 'uploads');
const HIST_DIR        = path.join(__dirname, '..', 'histograms');
const HIST_REF_DIR    = path.join(HIST_DIR, 'reference');
const REF_HIST_PATH   = path.join(HIST_REF_DIR, 'ref_hist.json');
const REF_HIST_PNG    = path.join(HIST_REF_DIR, 'ref_hist.png');
const HIST_RAND_DIR   = path.join(HIST_DIR, 'randomized');
const RAND_HIST_PATH  = path.join(HIST_RAND_DIR, 'rand_hist.json');
const RAND_HIST_PNG   = path.join(HIST_RAND_DIR, 'rand_hist.png');
const RAND_PREVIEW_PATH = path.join(HIST_RAND_DIR, 'rand_preview.png'); // rendered image at randomize time

/**
 * Compute a 256-bin grayscale histogram from a raw RGBA/RGB pixel buffer.
 * Uses ITU-R BT.601 luminance coefficients.
 *
 * @param {Buffer}  rawBuf  - raw pixel data (output of sharp().raw())
 * @param {object}  info    - { width, height, channels } from sharp
 * @returns {number[]}      - array of 256 bin counts
 */
function computeHistogramFromBuffer(rawBuf, { width, height, channels }) {
  const bins  = new Array(256).fill(0);
  const total = width * height;
  for (let i = 0; i < total; i++) {
    const o = i * channels;
    const lum = Math.round(0.299 * rawBuf[o] + 0.587 * rawBuf[o + 1] + 0.114 * rawBuf[o + 2]);
    bins[Math.min(lum, 255)]++;
  }
  return bins;
}

/**
 * Compute a histogram from an image file on disk (no filters applied).
 *
 * @param {string} imgPath - absolute path to the image file
 * @returns {Promise<number[]>}
 */
async function computeHistogramFromFile(imgPath) {
  const { data, info } = await sharp(imgPath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return computeHistogramFromBuffer(data, info);
}

/**
 * Compute a histogram from an already-encoded PNG buffer (e.g. from renderToPng).
 *
 * @param {Buffer} pngBuf
 * @returns {Promise<number[]>}
 */
async function computeHistogramFromPngBuffer(pngBuf) {
  const { data, info } = await sharp(pngBuf)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return computeHistogramFromBuffer(data, info);
}

/**
 * Resolve the host filesystem path for a canvas background image filename.
 *
 * @param {string|null} bg - backgroundImage value from canvas state
 * @returns {string|null}
 */
function resolveBackgroundPath(bg) {
  if (!bg) return null;
  if (bg.startsWith('/tile-assets/')) {
    return path.join('/app/tile-datasets', path.basename(bg));
  }
  if (bg.startsWith('/dataset-assets/')) {
    return path.join('/app/dataset-labeled', bg.slice('/dataset-assets/'.length));
  }
  if (bg.startsWith('/unlabeled-assets/')) {
    return path.join('/app/dataset-unlabeled', bg.slice('/unlabeled-assets/'.length));
  }
  return path.join(UPLOADS_DIR, path.basename(bg));
}

/**
 * Render a 256-bin histogram as a PNG bar chart using node-canvas.
 *
 * @param {number[]} bins    - raw or normalised bin counts
 * @param {string}   title   - chart title drawn at the top
 * @returns {Buffer}         - PNG buffer
 */
function renderHistogramPng(bins, title = 'Brightness Histogram') {
  const W = 800, H = 320;
  const PAD = { top: 40, right: 20, bottom: 40, left: 72 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top  - PAD.bottom;

  const canvas = createCanvas(W, H);
  const ctx    = canvas.getContext('2d');

  // Background
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, W, H);

  // Normalise bins to probability
  const total = bins.reduce((s, b) => s + b, 0) || 1;
  const norm  = bins.map(b => b / total);
  const maxVal = Math.max(...norm, 1e-9);

  const barW = plotW / bins.length;

  // Draw bars — colour gradient dark→light matching brightness
  for (let i = 0; i < bins.length; i++) {
    const barH = (norm[i] / maxVal) * plotH;
    const x    = PAD.left + i * barW;
    const y    = PAD.top  + plotH - barH;
    const grey = Math.round((i / (bins.length - 1)) * 220 + 35);
    ctx.fillStyle = `rgb(${grey},${grey},${grey})`;
    ctx.fillRect(x, y, Math.max(barW - 0.5, 1), barH);
  }

  // X-axis line
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD.left, PAD.top + plotH);
  ctx.lineTo(PAD.left + plotW, PAD.top + plotH);
  ctx.stroke();

  // Y-axis line
  ctx.beginPath();
  ctx.moveTo(PAD.left, PAD.top);
  ctx.lineTo(PAD.left, PAD.top + plotH);
  ctx.stroke();

  // Y-axis ticks and labels (5 evenly-spaced probability values)
  ctx.fillStyle = '#aaa';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'right';
  const yTicks = 5;
  for (let t = 0; t <= yTicks; t++) {
    const fraction = t / yTicks;
    const probVal  = maxVal * fraction;
    const yPos     = PAD.top + plotH - fraction * plotH;
    // Tick mark
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(PAD.left - 4, yPos);
    ctx.lineTo(PAD.left,     yPos);
    ctx.stroke();
    // Label — format as e.g. "0.0050"
    ctx.fillText(probVal.toFixed(4), PAD.left - 7, yPos + 4);
  }

  // Y-axis title (rotated)
  ctx.save();
  ctx.fillStyle = '#ccc';
  ctx.font = '13px sans-serif';
  ctx.textAlign = 'center';
  ctx.translate(13, PAD.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Probability', 0, 0);
  ctx.restore();

  // X-axis labels: 0, 64, 128, 192, 255
  ctx.fillStyle = '#aaa';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  [0, 64, 128, 192, 255].forEach(v => {
    const x = PAD.left + (v / (bins.length - 1)) * plotW;
    ctx.fillText(String(v), x, PAD.top + plotH + 18);
  });

  // X-axis label
  ctx.fillStyle = '#ccc';
  ctx.font = '13px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Brightness', PAD.left + plotW / 2, H - 4);

  // Title
  ctx.fillStyle = '#e0e0e0';
  ctx.font = 'bold 14px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(title, W / 2, 24);

  return canvas.toBuffer('image/png');
}

/**
 * Return a timestamp string in YYYYMMDD_HHMM format (day + hour + minute).
 * @returns {string}
 */
function _timestamp() {
  const d = new Date();
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}`;
}

/**
 * Persist the reference histogram (JSON + PNG) to histograms/reference/.
 * Files are written with a timestamp suffix so previous runs are preserved.
 * The un-suffixed "latest" files (ref_hist.json / ref_hist.png) are also
 * updated so that GET /api/histogram/reference always returns the most recent.
 *
 * @param {number[]} bins
 * @param {string}   filename - original image filename (for provenance)
 * @returns {{ bins, filename, capturedAt }}
 */
function saveReferenceHistogram(bins, filename) {
  fs.mkdirSync(HIST_REF_DIR, { recursive: true });
  const ts      = _timestamp();
  const payload = { bins, filename, capturedAt: new Date().toISOString() };
  const json    = JSON.stringify(payload);
  const png     = renderHistogramPng(bins, `Reference Histogram (before randomise)  [${ts}]`);

  // Timestamped copies — never overwritten
  fs.writeFileSync(path.join(HIST_REF_DIR, `ref_hist_${ts}.json`), json);
  fs.writeFileSync(path.join(HIST_REF_DIR, `ref_hist_${ts}.png`),  png);

  // "Latest" copies — always the most recent run, used by the API
  fs.writeFileSync(REF_HIST_PATH, json);
  fs.writeFileSync(REF_HIST_PNG,  png);

  return payload;
}

/**
 * Persist the randomized histogram (JSON + PNG) to histograms/randomized/.
 * Records the random filter values alongside the bins for comparison.
 *
 * @param {number[]} bins
 * @param {string}   filename - original image filename
 * @param {{ brightness, contrast, saturation }} filters - the applied random filters
 * @returns {{ bins, filename, capturedAt, randomFilters }}
 */
function saveRandomizedHistogram(bins, filename, filters) {
  fs.mkdirSync(HIST_RAND_DIR, { recursive: true });
  const ts      = _timestamp();
  const payload = { bins, filename, capturedAt: new Date().toISOString(), randomFilters: filters };
  const json    = JSON.stringify(payload);
  const png     = renderHistogramPng(bins, `Randomized Histogram  [${ts}]`);

  // Timestamped copies — never overwritten
  fs.writeFileSync(path.join(HIST_RAND_DIR, `rand_hist_${ts}.json`), json);
  fs.writeFileSync(path.join(HIST_RAND_DIR, `rand_hist_${ts}.png`),  png);

  // "Latest" copies — always the most recent run
  fs.writeFileSync(RAND_HIST_PATH, json);
  fs.writeFileSync(RAND_HIST_PNG,  png);

  return payload;
}

/**
 * Compute the SEM histogram score between an image histogram and a reference.
 * Matches the Python sem_histogram_error() metric exactly.
 *
 * score = wasserstein + clipping_weight * |clipping_fraction_image − clipping_fraction_ref|
 *
 * The clipping term uses the absolute difference so a perfect match (image == reference)
 * always scores 0, even when the reference itself has clipped pixels.
 *
 * @param {number[]} imageBins
 * @param {number[]} refBins
 * @param {{ edgeBins?: number, clippingWeight?: number }} opts
 * @returns {{ score, wasserstein, clipping_fraction, clipping_penalty }}
 */
function computeScore(imageBins, refBins, { edgeBins = 5, clippingWeight = 5.0 } = {}) {
  const n = imageBins.length;
  const totalImg = imageBins.reduce((s, b) => s + b, 0) || 1;
  const totalRef = refBins.reduce((s, b) => s + b, 0) || 1;
  const pImg = imageBins.map(b => b / totalImg);
  const pRef = refBins.map(b => b / totalRef);

  // Wasserstein-1 via CDF difference, normalised by (n-1)
  let cdfImg = 0, cdfRef = 0, wasserstein = 0;
  for (let i = 0; i < n; i++) {
    cdfImg += pImg[i];
    cdfRef += pRef[i];
    wasserstein += Math.abs(cdfImg - cdfRef);
  }
  wasserstein /= (n - 1);

  // Clipping penalty — |excess clipping in image vs reference|
  // Absolute difference so a perfect match scores 0 even if the reference
  // itself has clipped pixels.
  const clippingFractionImg =
    pImg.slice(0, edgeBins).reduce((s, v) => s + v, 0) +
    pImg.slice(Math.max(0, n - edgeBins)).reduce((s, v) => s + v, 0);
  const clippingFractionRef =
    pRef.slice(0, edgeBins).reduce((s, v) => s + v, 0) +
    pRef.slice(Math.max(0, n - edgeBins)).reduce((s, v) => s + v, 0);
  const clippingFraction = Math.abs(clippingFractionImg - clippingFractionRef);
  const clippingPenalty = clippingWeight * clippingFraction;

  return {
    score:              Math.round((wasserstein + clippingPenalty) * 1e6) / 1e6,
    wasserstein:        Math.round(wasserstein    * 1e6) / 1e6,
    clipping_fraction:  Math.round(clippingFraction * 1e6) / 1e6,
    clipping_penalty:   Math.round(clippingPenalty  * 1e6) / 1e6,
  };
}

/**
 * Load the persisted reference histogram, or null if none exists.
 *
 * @returns {{ bins, filename, capturedAt }|null}
 */
function loadReferenceHistogram() {
  if (!fs.existsSync(REF_HIST_PATH)) return null;
  try {
    return JSON.parse(fs.readFileSync(REF_HIST_PATH, 'utf8'));
  } catch {
    return null;
  }
}

module.exports = {
  computeHistogramFromFile,
  computeHistogramFromPngBuffer,
  resolveBackgroundPath,
  renderHistogramPng,
  saveReferenceHistogram,
  saveRandomizedHistogram,
  loadReferenceHistogram,
  computeScore,
  HIST_REF_DIR,
  REF_HIST_PATH,
  REF_HIST_PNG,
  HIST_RAND_DIR,
  RAND_HIST_PATH,
  RAND_HIST_PNG,
  RAND_PREVIEW_PATH,
  UPLOADS_DIR,
};
