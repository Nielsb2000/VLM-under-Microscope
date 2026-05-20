// server/routes/images.js — list and load images from the uploads directory
const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs');
const { loadImage } = require('canvas');
const state = require('../state');

const UPLOADS_DIR = path.join(__dirname, '..', '..', 'uploads');
const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']);

// GET /api/images  — list all uploaded images
router.get('/', (req, res) => {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
  const files = fs.readdirSync(UPLOADS_DIR)
    .filter(f => IMAGE_EXTS.has(path.extname(f).toLowerCase()))
    .map(f => ({
      filename: f,
      url: `/uploads/${f}`,
      size: fs.statSync(path.join(UPLOADS_DIR, f)).size,
    }));
  res.json({ images: files });
});

// POST /api/images/load  { filename }
// Load a previously uploaded image as the canvas background by filename.
router.post('/load', async (req, res) => {
  const { filename } = req.body || {};
  if (!filename) return res.status(400).json({ error: 'filename required' });

  // Prevent path traversal
  const safe = path.basename(filename);
  const imgPath = path.join(UPLOADS_DIR, safe);
  if (!fs.existsSync(imgPath)) return res.status(404).json({ error: 'Image not found' });

  let imgWidth = state.getState().canvas.width;
  let imgHeight = state.getState().canvas.height;
  try {
    const img = await loadImage(imgPath);
    imgWidth = img.width;
    imgHeight = img.height;
  } catch (_) {}

  state.setBackground(safe, imgWidth, imgHeight);
  state.resetViewport();
  res.json({ ok: true, filename: safe, width: imgWidth, height: imgHeight });
});

module.exports = router;
