// server/routes/randomize.js - randomizer endpoint for the image-quality case study
'use strict';

const express = require('express');
const router  = express.Router();
const fs      = require('fs');
const state   = require('../state');

function _seededRandom(seed) {
  // Mulberry32 PRNG. Stable across Node versions for reproducible filter randomization.
  let t = Number(seed) >>> 0;
  return function () {
    t += 0x6D2B79F5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function _rngFromSeed(seed) {
  if (seed === undefined || seed === null || seed === '') return Math.random;
  const n = Number(seed);
  if (!Number.isFinite(n)) throw new Error('seed must be numeric');
  return _seededRandom(n);
}
const { renderToPng } = require('../renderer');
const {
  computeHistogramFromFile,
  computeHistogramFromPngBuffer,
  resolveBackgroundPath,
  saveReferenceHistogram,
  saveRandomizedHistogram,
} = require('../histogramUtils');

// POST /api/randomize
// Performs three actions in order:
//   1. Reset filters to 100/100/100 so the canvas shows the true image before randomisation.
//   2. Capture the brightness histogram of the raw background image (no filters)
//      and persist it as the reference histogram for later evaluation.
//   3. Apply a random combination of brightness / contrast / saturation filters.
//
// The client calls this when the human presses the "🎲 Randomize" button.
// After this call the agent is expected to iteratively refine the image quality
// using only visual feedback (VLM) - it must NOT read the reference histogram.
router.post('/', async (req, res) => {
  const { seed } = req.body || {};
  let rng;
  try {
    rng = _rngFromSeed(seed);
  } catch (e) {
    return res.status(400).json({ error: e.message });
  }

  const s  = state.getState();
  const bg = s.canvas.backgroundImage;
  if (!bg) return res.status(400).json({ error: 'No background image is loaded on the canvas.' });

  const imgPath = resolveBackgroundPath(bg);
  if (!imgPath || !fs.existsSync(imgPath)) {
    return res.status(404).json({ error: `Background image not found: ${imgPath}` });
  }

  // --- 1. Reset filters to neutral so the canvas is at the true reference state ---
  state.setFilters({ brightness: 100, contrast: 100, saturation: 100 });

  // --- 2. Capture reference histogram (raw image, no filters) ---
  try {
    const bins = await computeHistogramFromFile(imgPath);
    saveReferenceHistogram(bins, bg);
  } catch (e) {
    console.error('[randomize] histogram capture failed:', e);
    return res.status(500).json({ error: `Histogram capture failed: ${e.message}` });
  }

  // --- 3. Randomize filter values ---
  // Ranges chosen to make recovery non-trivial but possible for a VLM agent:
  //   brightness: 0–300 (allows very dark or very bright states)
  //   contrast:   0–300 (allows washed-out or over-contrasted states)
  const randInt = (min, max) => Math.floor(rng() * (max - min + 1)) + min;
  const brightness = randInt(0, 300);
  const contrast   = randInt(0, 300);

  const filters = state.setFilters({ brightness, contrast });

  // --- 4. Capture randomized histogram + preview image (with filters applied) ---
  try {
    // Strip segmentation and drawn objects - histogram reflects image+filters only.
    const { segmentation: _seg, ...histState } = state.getState();
    histState.objects = [];
    const png  = await renderToPng(histState);
    const bins = await computeHistogramFromPngBuffer(png);
    saveRandomizedHistogram(bins, bg, { brightness, contrast });
    // Also persist the rendered image so the evaluation script can export it.
    const { RAND_PREVIEW_PATH } = require('../histogramUtils');
    require('fs').writeFileSync(RAND_PREVIEW_PATH, png);
  } catch (e) {
    console.error('[randomize] randomized histogram capture failed:', e);
    // non-fatal - continue
  }

  // Reset session counters AFTER applying the random filters so the
  // randomize call itself is not counted as an agent adjustment.
  state.resetSession();

  // Persist the random filter values into the reference JSON so each run
  // records the exact starting difficulty.
  try {
    const {
      loadReferenceHistogram, REF_HIST_PATH, HIST_REF_DIR,
    } = require('../histogramUtils');
    const fs   = require('fs');
    const path = require('path');
    const ref  = loadReferenceHistogram();
    if (ref) {
      ref.randomFilters = { brightness, contrast };
      ref.randomizationSeed = seed ?? null;
      const jsonStr = JSON.stringify(ref);
      fs.writeFileSync(REF_HIST_PATH, jsonStr);
      // Also patch the matching timestamped copy if it exists
      const tsFmt = ref.capturedAt.slice(0, 16).replace(/-/g, '').replace('T', '_').replace(':', '');
      const tsFile = path.join(HIST_REF_DIR, `ref_hist_${tsFmt}.json`);
      if (fs.existsSync(tsFile)) fs.writeFileSync(tsFile, jsonStr);
    }
  } catch (e) {
    console.error('[randomize] failed to patch reference JSON with randomFilters:', e);
  }

  res.json({
    ok: true,
    seed: seed ?? null,
    filters,
    reference_histogram_saved: true,
    randomized_histogram_saved: true,
  });
});

module.exports = router;
