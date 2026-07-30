// server/routes/dataset.js - browse and load images from labeled and unlabeled SEM datasets
//
// GET  /api/dataset/list          list categories from both datasets
// POST /api/dataset/load          { source, category, filename } - set canvas background
//   source: "labeled" (default) | "unlabeled"
//   For unlabeled: category is "{split}/{subfolder}" (e.g. "train/Acquisitions_first_run")

'use strict';

const express = require('express');
const router  = express.Router();
const path    = require('path');
const fs      = require('fs');
const sharp   = require('sharp');
const state   = require('../state');

const VALIDATION_ANDREA_DIR = '/app/validation-andrea-dataset';
const VALIDATION_UNC_LUCA_DIR = '/app/validation-unc-luca-dataset';

function _seededRandom(seed) {
  // Mulberry32 PRNG. Stable across Node versions for reproducible case-study sampling.
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

const LABELED_DIR   = '/app/dataset-labeled';
const UNLABELED_DIR = '/app/dataset-unlabeled';   // sem_segmentation_ssl/{train,val}/{subfolder}
const LABELED_EXTS  = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.webp']);
const TIFF_EXTS     = new Set(['.tiff', '.tif']);
const SEM_EXTS      = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif']);

function _safeRelPath(...parts) {
  // Join and ensure the result stays within the base (no traversal)
  const joined = parts.map(p => path.basename(p)).join(path.sep);
  return joined;
}



function _listImagesRecursive(baseDir, urlPrefix, allowedExts = SEM_EXTS) {
  const imagesByCategory = [];

  function walk(currentDir, relParts) {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });

    const files = entries
      .filter(e => e.isFile() && allowedExts.has(path.extname(e.name).toLowerCase()))
      .map(e => e.name)
      .sort();

    if (files.length) {
      const category = relParts.length ? relParts.join('/') : '.';

      imagesByCategory.push({
        name: category,
        images: files.map(f => ({
          filename: f,
          url: `${urlPrefix}/${relParts.map(encodeURIComponent).join('/')}${relParts.length ? '/' : ''}${encodeURIComponent(f)}`,
        })),
      });
    }

    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (!entry.isDirectory()) continue;
      walk(path.join(currentDir, entry.name), [...relParts, entry.name]);
    }
  }

  if (fs.existsSync(baseDir)) {
    walk(baseDir, []);
  }

  return imagesByCategory;
}



// GET /api/dataset/list
router.get('/list', (req, res) => {
  try {
    const result = { labeled: [], unlabeled: [], validation_andrea: [], validation_unc_luca: [] };

    // ---- Labeled ----
    if (fs.existsSync(LABELED_DIR)) {
      for (const entry of fs.readdirSync(LABELED_DIR, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue;
        const catDir = path.join(LABELED_DIR, entry.name);
        const images = fs.readdirSync(catDir)
          .filter(f => LABELED_EXTS.has(path.extname(f).toLowerCase()))
          .sort()
          .map(f => ({ filename: f, url: `/dataset-assets/${entry.name}/${f}` }));
        if (images.length) result.labeled.push({ name: entry.name, images });
      }
    }

    // ---- Unlabeled ----
    if (fs.existsSync(UNLABELED_DIR)) {
      for (const split of ['train', 'val']) {
        const splitDir = path.join(UNLABELED_DIR, split);
        if (!fs.existsSync(splitDir)) continue;
        for (const sub of fs.readdirSync(splitDir, { withFileTypes: true })) {
          if (!sub.isDirectory()) continue;
          const subDir = path.join(splitDir, sub.name);
          const images = fs.readdirSync(subDir)
            .filter(f => TIFF_EXTS.has(path.extname(f).toLowerCase()))
            .sort()
            .map(f => ({ filename: f, url: `/unlabeled-assets/${split}/${sub.name}/${f}` }));
          if (images.length) {
            result.unlabeled.push({ name: `${split}/${sub.name}`, images });
          }
        }
      }
    }

    // ---- Validation Andrea CS2 ----
    result.validation_andrea = _listImagesRecursive(
      VALIDATION_ANDREA_DIR,
      '/validation-andrea-assets'
    );

    // ---- Validation UNC Luca CS2 ----
    result.validation_unc_luca = _listImagesRecursive(
      VALIDATION_UNC_LUCA_DIR,
      '/validation-unc-luca-assets',
      TIFF_EXTS
    );

    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// POST /api/dataset/load  { source?, category, filename }
router.post('/load', async (req, res) => {
  const { source = 'labeled', category, filename } = req.body || {};

  if (!category || !filename) {
    return res.status(400).json({ error: 'category and filename required' });
  }

  const safeFilename = path.basename(filename);
  let imgPath, bgPath;

  if (source === 'unlabeled') {
    // category is "split/subfolder" - validate each segment
    const parts = category.split('/').map(p => path.basename(p));

    if (parts.length !== 2) {
      return res.status(400).json({
        error: 'unlabeled category must be "split/subfolder"',
      });
    }

    imgPath = path.join(UNLABELED_DIR, ...parts, safeFilename);
    bgPath  = `/unlabeled-assets/${parts[0]}/${parts[1]}/${safeFilename}`;
  } else if (source === 'validation_andrea') {
    const relParts = category === '.'
      ? []
      : category.split('/').map(p => path.basename(p));

    imgPath = path.join(VALIDATION_ANDREA_DIR, ...relParts, safeFilename);

    const relUrl = relParts.length ? `${relParts.join('/')}/` : '';
    bgPath = `/validation-andrea-assets/${relUrl}${safeFilename}`;
  } 
  else if (source === 'labeled') {
    const safeCategory = path.basename(category);

    imgPath = path.join(LABELED_DIR, safeCategory, safeFilename);
    bgPath  = `/dataset-assets/${safeCategory}/${safeFilename}`;
  } 
  else if (source === 'validation_unc_luca') {
    const relParts = category === '.'
      ? []
      : category.split('/').map(p => path.basename(p));

    imgPath = path.join(VALIDATION_UNC_LUCA_DIR, ...relParts, safeFilename);

    const relUrl = relParts.length ? `${relParts.join('/')}/` : '';
    bgPath = `/validation-unc-luca-assets/${relUrl}${safeFilename}`;
  } else if (source === 'labeled') {
    const safeCategory = path.basename(category);

    imgPath = path.join(LABELED_DIR, safeCategory, safeFilename);
    bgPath  = `/dataset-assets/${safeCategory}/${safeFilename}`;
  }
   
  else {
    return res.status(400).json({
      error: 'source must be "unlabeled", "labeled", "validation_andrea", or "validation_unc_luca"',
    });
  }

  if (!fs.existsSync(imgPath)) {
    return res.status(404).json({ error: 'Image not found' });
  }

  let imgWidth  = state.getState().canvas.width;
  let imgHeight = state.getState().canvas.height;

  try {
    const meta = await sharp(imgPath).metadata();
    imgWidth  = meta.width  || imgWidth;
    imgHeight = meta.height || imgHeight;
  } catch (_) {}

  state.setBackground(bgPath, imgWidth, imgHeight, {
    switchToImageMode: true,
  });

  res.json({
    ok: true,
    source,
    category,
    filename: safeFilename,
    width: imgWidth,
    height: imgHeight,
  });
});

// POST /api/dataset/sample
// Load a uniformly random image from the requested dataset.
// Optional body: { source: 'unlabeled' | 'labeled', seed?: number }
// With the same mounted dataset and seed, this returns the same image.
router.post('/sample', async (req, res) => {
  const { source = 'unlabeled', seed } = req.body || {};

  try {
    const rng = _rngFromSeed(seed);
    const candidates = []; // { imgPath, bgPath, category, filename }

    if (source === 'unlabeled') {
      if (!fs.existsSync(UNLABELED_DIR)) {
        return res.status(404).json({
          error: 'Unlabeled dataset not mounted at ' + UNLABELED_DIR,
        });
      }

      for (const split of ['train', 'val']) {
        const splitDir = path.join(UNLABELED_DIR, split);
        if (!fs.existsSync(splitDir)) continue;

        for (const sub of fs.readdirSync(splitDir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
          if (!sub.isDirectory()) continue;

          const subDir = path.join(splitDir, sub.name);

          for (const f of fs.readdirSync(subDir).sort()) {
            if (!TIFF_EXTS.has(path.extname(f).toLowerCase())) continue;

            candidates.push({
              imgPath: path.join(subDir, f),
              bgPath: `/unlabeled-assets/${split}/${sub.name}/${f}`,
              category: `${split}/${sub.name}`,
              filename: f,
            });
          }
        }
      }
    } else if (source === 'labeled') {
      if (!fs.existsSync(LABELED_DIR)) {
        return res.status(404).json({
          error: 'Labeled dataset not mounted at ' + LABELED_DIR,
        });
      }

      for (const entry of fs.readdirSync(LABELED_DIR, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
        if (!entry.isDirectory()) continue;
        const catDir = path.join(LABELED_DIR, entry.name);
        for (const f of fs.readdirSync(catDir).sort()) {
          if (!LABELED_EXTS.has(path.extname(f).toLowerCase())) continue;
          candidates.push({
            imgPath: path.join(catDir, f),
            bgPath: `/dataset-assets/${entry.name}/${f}`,
            category: entry.name,
            filename: f,
          });
        }
      }
    } else if (source === 'validation_andrea') {
      if (!fs.existsSync(VALIDATION_ANDREA_DIR)) {
        return res.status(404).json({
          error: 'Validation Andrea dataset not mounted at ' + VALIDATION_ANDREA_DIR,
        });
      }

      function walk(currentDir, relParts) {
        const entries = fs.readdirSync(currentDir, { withFileTypes: true })
          .sort((a, b) => a.name.localeCompare(b.name));

        for (const entry of entries) {
          const fullPath = path.join(currentDir, entry.name);

          if (entry.isDirectory()) {
            walk(fullPath, [...relParts, entry.name]);
            continue;
          }

          if (!entry.isFile()) continue;
          if (!SEM_EXTS.has(path.extname(entry.name).toLowerCase())) continue;

          const category = relParts.length ? relParts.join('/') : '.';
          const relUrl = relParts.length ? `${relParts.join('/')}/` : '';

          candidates.push({
            imgPath: fullPath,
            bgPath: `/validation-andrea-assets/${relUrl}${entry.name}`,
            category,
            filename: entry.name,
          });
        }
      }

      walk(VALIDATION_ANDREA_DIR, []);
    } else {
      return res.status(400).json({ error: 'source must be "unlabeled", "labeled", or "validation_andrea"' });
        }

    if (candidates.length === 0) {
      return res.status(404).json({
        error: `No images found in ${source} dataset`,
      });
    }

    candidates.sort((a, b) => `${a.category}/${a.filename}`.localeCompare(`${b.category}/${b.filename}`));
    const pick_index = Math.floor(rng() * candidates.length);
    const pick = candidates[pick_index];

    let imgWidth = state.getState().canvas.width;
    let imgHeight = state.getState().canvas.height;

    try {
      const meta = await sharp(pick.imgPath).metadata();
      imgWidth = meta.width || imgWidth;
      imgHeight = meta.height || imgHeight;
    } catch (_) {}

    state.setBackground(pick.bgPath, imgWidth, imgHeight, {
      switchToImageMode: true,
    });

    state.resetViewport();

    res.json({
      ok: true,
      source,
      seed: seed ?? null,
      pick_index,
      category: pick.category,
      filename: pick.filename,
      backgroundImage: pick.bgPath,
      width: imgWidth,
      height: imgHeight,
      total_candidates: candidates.length,
    });
  } catch (e) {
    console.error('[dataset/sample]', e);
    res.status(500).json({ error: e.message });
  }
});

module.exports = router;
