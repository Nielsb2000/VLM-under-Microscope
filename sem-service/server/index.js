// server/index.js — Express entry point
const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// ---- Middleware ----
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Serve uploaded images
const UPLOADS_DIR = path.join(__dirname, '..', 'uploads');
fs.mkdirSync(UPLOADS_DIR, { recursive: true });
app.use('/uploads', express.static(UPLOADS_DIR));

// Serve tile-grid dataset images — convert TIFF→PNG on the fly (browsers can't render TIFF)
app.get('/tile-assets/:filename', async (req, res) => {
  const sharp = require('sharp');
  const tilePath = path.join('/app/tile-datasets', path.basename(req.params.filename));
  if (!fs.existsSync(tilePath)) return res.status(404).json({ error: 'Tile not found' });
  try {
    res.setHeader('Content-Type', 'image/png');
    sharp(tilePath).png().pipe(res);
  } catch (err) {
    console.error('[tile-assets]', err);
    res.status(500).json({ error: err.message });
  }
});

// Serve labeled dataset images
app.use('/dataset-assets', express.static('/app/dataset-labeled'));

// Serve unlabeled dataset images — convert TIFF→PNG on the fly
app.get('/unlabeled-assets/*', async (req, res) => {
  const sharp = require('sharp');
  // req.params[0] is everything after /unlabeled-assets/
  const relPath = req.params[0];
  const imgPath = path.join('/app/dataset-unlabeled', relPath);
  if (!fs.existsSync(imgPath)) return res.status(404).json({ error: 'Image not found' });
  try {
    res.setHeader('Content-Type', 'image/png');
    sharp(imgPath).png().pipe(res);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Serve static frontend
app.use(express.static(path.join(__dirname, '..', 'client')));

// ---- API Routes ----
app.use('/api/canvas', require('./routes/canvas'));
app.use('/api/draw', require('./routes/draw'));
app.use('/api/objects', require('./routes/objects'));
app.use('/api/export', require('./routes/export'));
app.use('/api/viewport', require('./routes/viewport'));
app.use('/api/images', require('./routes/images'));
app.use('/api/camera', require('./routes/camera'));
app.use('/api/histogram', require('./routes/histogram'));
app.use('/api/randomize', require('./routes/randomize'));
app.use('/api/session',   require('./routes/session'));
app.use('/api/dataset',   require('./routes/dataset'));
app.use('/api/atlas',     require('./routes/atlas'));

// ---- Fallback to SPA ----
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'client', 'index.html'));
});

// ---- Start ----
app.listen(PORT, () => {
  console.log(`sem-service running at http://localhost:${PORT}`);
});
