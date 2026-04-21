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

// Serve static frontend
app.use(express.static(path.join(__dirname, '..', 'client')));

// ---- API Routes ----
app.use('/api/canvas', require('./routes/canvas'));
app.use('/api/draw', require('./routes/draw'));
app.use('/api/objects', require('./routes/objects'));
app.use('/api/export', require('./routes/export'));
app.use('/api/viewport', require('./routes/viewport'));
app.use('/api/images', require('./routes/images'));

// ---- Fallback to SPA ----
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'client', 'index.html'));
});

// ---- Start ----
app.listen(PORT, () => {
  console.log(`paint-service running at http://localhost:${PORT}`);
});
