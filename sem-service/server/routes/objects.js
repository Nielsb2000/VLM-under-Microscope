// server/routes/objects.js
const express = require('express');
const router = express.Router();
const state = require('../state');

// GET /api/objects
router.get('/', (req, res) => {
  res.json(state.getState().objects);
});

// GET /api/objects/:id
router.get('/:id', (req, res) => {
  const obj = state.getObject(req.params.id);
  if (!obj) return res.status(404).json({ error: 'Object not found' });
  res.json(obj);
});

// PATCH /api/objects/:id
router.patch('/:id', (req, res) => {
  const updated = state.updateObject(req.params.id, req.body || {});
  if (!updated) return res.status(404).json({ error: 'Object not found' });
  res.json(updated);
});

// DELETE /api/objects/:id
router.delete('/:id', (req, res) => {
  const ok = state.deleteObject(req.params.id);
  if (!ok) return res.status(404).json({ error: 'Object not found' });
  res.json({ ok: true });
});

module.exports = router;
