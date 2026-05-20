// server/routes/histogram.js — brightness histogram endpoints
'use strict';

const express = require('express');
const router  = express.Router();
const fs      = require('fs');
const state   = require('../state');
const { renderToPng } = require('../renderer');
const {
  computeHistogramFromFile,
  computeHistogramFromPngBuffer,
  resolveBackgroundPath,
  saveReferenceHistogram,
  loadReferenceHistogram,
  computeScore,
  REF_HIST_PATH,
  RAND_HIST_PATH,
} = require('../histogramUtils');

// POST /api/histogram/capture-reference
// Compute the grayscale histogram of the raw background image (no CSS filters),
// persist it to disk, and return it.
router.post('/capture-reference', async (req, res) => {
  const s  = state.getState();
  const bg = s.canvas.backgroundImage;
  if (!bg) return res.status(400).json({ error: 'No background image is loaded on the canvas.' });

  const imgPath = resolveBackgroundPath(bg);
  if (!imgPath || !fs.existsSync(imgPath)) {
    return res.status(404).json({ error: `Background image file not found: ${imgPath}` });
  }

  try {
    const bins    = await computeHistogramFromFile(imgPath);
    const payload = saveReferenceHistogram(bins, bg);
    res.json({ ok: true, bins: payload.bins, filename: payload.filename, capturedAt: payload.capturedAt });
  } catch (e) {
    console.error('[histogram/capture-reference]', e);
    res.status(500).json({ error: e.message });
  }
});

// GET /api/histogram/reference
// Return the previously saved reference histogram.
router.get('/reference', (req, res) => {
  const data = loadReferenceHistogram();
  if (!data) {
    return res.status(404).json({
      error: 'No reference histogram saved. Call POST /api/histogram/capture-reference first.',
    });
  }
  res.json(data);
});

// GET /api/histogram/current
// Render the canvas with the current filter state, then compute and return its histogram.
// Note: this endpoint exists for the external evaluation script; the agent never calls it.
router.get('/current', async (req, res) => {
  try {
    const s    = state.getState();
    // Strip segmentation overlay and drawn objects — the histogram must reflect
    // only the raw image pixels after CSS filters, nothing drawn on top.
    const { segmentation: _seg, ...histState } = s;
    histState.objects = [];
    const png  = await renderToPng(histState);
    const bins = await computeHistogramFromPngBuffer(png);

    // Score vs neutral (brightness=100, contrast=100) — skip gracefully if render fails
    let metrics = {};
    try {
      const neutralState = { canvas: s.canvas, objects: [], filters: { brightness: 100, contrast: 100 } };
      const neutralPng   = await renderToPng(neutralState);
      const neutralBins  = await computeHistogramFromPngBuffer(neutralPng);
      metrics = computeScore(bins, neutralBins);
    } catch (_) { /* neutral render failed — skip score */ }

    res.json({ ok: true, bins, capturedAt: new Date().toISOString(), ...metrics });
  } catch (e) {
    console.error('[histogram/current]', e);
    res.status(500).json({ error: e.message });
  }
});

// GET /api/histogram/neutral
// Render the canvas with filters at 100/100 (no distortion) and return the histogram.
// Used by the live histogram as the always-visible reference line.
router.get('/neutral', async (req, res) => {
  try {
    const s = state.getState();
    if (!s.canvas.backgroundImage) return res.status(404).json({ error: 'No image loaded.' });
    // Exclude segmentation and drawn objects — neutral baseline is image-only.
    const neutralState = { canvas: s.canvas, objects: [], filters: { brightness: 100, contrast: 100 } };
    const png  = await renderToPng(neutralState);
    const bins = await computeHistogramFromPngBuffer(png);
    res.json({ ok: true, bins });
  } catch (e) {
    console.error('[histogram/neutral]', e);
    res.status(500).json({ error: e.message });
  }
});

// GET /api/histogram/randomized
// Return the histogram captured immediately after randomize was called (with filters applied).
router.get('/randomized', (req, res) => {
  if (!fs.existsSync(RAND_HIST_PATH)) {
    return res.status(404).json({
      error: 'No randomized histogram saved. Call POST /api/randomize first.',
    });
  }
  try {
    const data = JSON.parse(fs.readFileSync(RAND_HIST_PATH, 'utf8'));
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/histogram/result-image
// Render the current canvas (image + active filters, no objects, no segmentation) as a PNG.
// Represents the "final result" image the agent produced.
router.get('/result-image', async (req, res) => {
  try {
    const s = state.getState();
    if (!s.canvas.backgroundImage) return res.status(404).json({ error: 'No image loaded.' });
    const { segmentation: _seg, ...renderState } = s;
    renderState.objects = [];
    const png = await renderToPng(renderState);
    res.set('Content-Type', 'image/png');
    res.send(png);
  } catch (e) {
    console.error('[histogram/result-image]', e);
    res.status(500).json({ error: e.message });
  }
});

// GET /api/histogram/reference-image
// Render the canvas background at neutral filters (brightness=100, contrast=100) as a PNG.
// Represents the "ground truth" image before randomisation.
router.get('/reference-image', async (req, res) => {
  try {
    const s = state.getState();
    if (!s.canvas.backgroundImage) return res.status(404).json({ error: 'No image loaded.' });
    const renderState = { canvas: s.canvas, objects: [], filters: { brightness: 100, contrast: 100 } };
    const png = await renderToPng(renderState);
    res.set('Content-Type', 'image/png');
    res.send(png);
  } catch (e) {
    console.error('[histogram/reference-image]', e);
    res.status(500).json({ error: e.message });
  }
});

// GET /api/histogram/randomized-image
// Serve the rendered image captured at the moment randomize was called.
router.get('/randomized-image', (req, res) => {
  const { RAND_PREVIEW_PATH } = require('../histogramUtils');
  if (!fs.existsSync(RAND_PREVIEW_PATH)) {
    return res.status(404).json({ error: 'No randomized image saved. Call POST /api/randomize first.' });
  }
  res.set('Content-Type', 'image/png');
  res.sendFile(RAND_PREVIEW_PATH);
});

module.exports = router;
