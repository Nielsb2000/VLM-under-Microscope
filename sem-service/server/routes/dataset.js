// server/routes/dataset.js — browse and load images from labeled and unlabeled SEM datasets
//
// GET  /api/dataset/list          list categories from both datasets
// POST /api/dataset/load          { source, category, filename } — set canvas background
//   source: "labeled" (default) | "unlabeled"
//   For unlabeled: category is "{split}/{subfolder}" (e.g. "train/Acquisitions_first_run")

'use strict';

const express = require('express');
const router  = express.Router();
const path    = require('path');
const fs      = require('fs');
const sharp   = require('sharp');
const state   = require('../state');

const LABELED_DIR   = '/app/dataset-labeled';
const UNLABELED_DIR = '/app/dataset-unlabeled';   // sem_segmentation_ssl/{train,val}/{subfolder}
const LABELED_EXTS  = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.webp']);
const TIFF_EXTS     = new Set(['.tiff', '.tif']);

function _safeRelPath(...parts) {
  // Join and ensure the result stays within the base (no traversal)
  const joined = parts.map(p => path.basename(p)).join(path.sep);
  return joined;
}

// GET /api/dataset/list
router.get('/list', (req, res) => {
  try {
    const result = { labeled: [], unlabeled: [] };

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

    // ---- Unlabeled ---- (two-level: split / acquisition)
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
    // category is "split/subfolder" — validate each segment
    const parts = category.split('/').map(p => path.basename(p));
    if (parts.length !== 2) return res.status(400).json({ error: 'unlabeled category must be "split/subfolder"' });
    imgPath = path.join(UNLABELED_DIR, ...parts, safeFilename);
    bgPath  = `/unlabeled-assets/${parts[0]}/${parts[1]}/${safeFilename}`;
  } else {
    const safeCategory = path.basename(category);
    imgPath = path.join(LABELED_DIR, safeCategory, safeFilename);
    bgPath  = `/dataset-assets/${safeCategory}/${safeFilename}`;
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

  state.setBackground(bgPath, imgWidth, imgHeight, { switchToImageMode: true });

  res.json({ ok: true, source, category, filename: safeFilename, width: imgWidth, height: imgHeight });
});

module.exports = router;
