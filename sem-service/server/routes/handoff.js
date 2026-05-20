// server/routes/handoff.js — bidirectional human ↔ model message channel
// The browser POSTs a question; the agent GETs it, annotates, then POSTs a reply.

const express = require('express');
const router = express.Router();

let pending = null;   // { message, pngBase64, annotations, submittedAt }
let response = null;  // { text, submittedAt }

// ---- Browser → Agent ----

// POST /api/handoff  { message }
// Captures current canvas export + user question. Replaces any prior pending request.
router.post('/', async (req, res) => {
  const { message = '' } = req.body || {};
  const state = require('../state').getState();
  const { renderToPng } = require('../renderer');

  let pngBase64 = null;
  try {
    const buf = await renderToPng(state);
    pngBase64 = buf.toString('base64');
  } catch (e) {
    console.error('Handoff PNG render failed:', e.message);
  }

  pending = {
    message,
    pngBase64,
    annotations: state,
    submittedAt: new Date().toISOString(),
  };
  response = null; // clear previous reply

  res.json({ ok: true, submittedAt: pending.submittedAt });
});

// GET /api/handoff  — agent polls for a pending request
router.get('/', (req, res) => {
  if (!pending) return res.json({ pending: false });
  res.json({ pending: true, ...pending });
});

// DELETE /api/handoff  — agent marks request as consumed
router.delete('/', (req, res) => {
  pending = null;
  res.json({ ok: true });
});

// ---- Agent → Browser ----

// POST /api/handoff/response  { text }
router.post('/response', (req, res) => {
  const { text = '' } = req.body || {};
  response = { text, submittedAt: new Date().toISOString() };
  pending = null; // consumed
  res.json({ ok: true });
});

// GET /api/handoff/response  — browser polls for agent reply
router.get('/response', (req, res) => {
  if (!response) return res.json({ ready: false });
  res.json({ ready: true, ...response });
});

module.exports = router;
