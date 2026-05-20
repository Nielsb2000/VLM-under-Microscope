# aig-msd-visual-reasoning: Visual Spatial Reasoning for SEM/TEM Microscopy

## Introduction

**aig-msd-visual-reasoning** is a research-driven project focused on advancing visual spatial reasoning in AI models for Scanning Electron Microscopy (SEM) and Transmission Electron Microscopy (TEM). The project combines cloud VLMs (GPT-4o, GPT-5.x) and local models (LLaVA, Bunny) with a sandboxed DeepAgent framework to build, evaluate, and benchmark spatial reasoning capabilities for scientific imaging.

---

## Project Structure

This repository contains **three self-contained parts**. All share the same `.env` config and `uv` virtual environment.

| Part | Folder | Purpose |
|------|--------|---------|
| 1 — SpatialEval | `spatial_eval/` | Benchmark VLMs on spatial reasoning tasks (Maze-Nav, Spatial-Grid, etc.) |
| 2 — MS Paint Reasoning Evaluation | `MS_Paint_Reasoning_Evaluation/` | Evaluate visual reasoning on MS Paint-style images with blur/greyscale/inversion |
| 3 — DeepAgent + SEM Service | `sem-service/`, root | Interactive annotation agent with live SEM canvas, SAM2 segmentation, histogram evaluation |

---

## Quick Start

### Prerequisites

- Python `>=3.11,<3.12` (use `pyenv` or similar)
- [`uv`](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose

### Install dependencies

```bash
uv sync
```

### Environment

Copy `.env.example` to `.env` and fill in your API keys:

```env
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=https://<azure-endpoint>/openai/v1/
MODEL_NAME=gpt-5.2
MODEL_REASONING_EFFORT=medium   # low | medium | high (GPT-5.x only)
OPIK_ENABLED=false
HUGGINGFACEHUB_API_TOKEN=<token>
```

---

## Part 1 — SpatialEval

Benchmarks spatial reasoning across 4 tasks (Maze-Nav, Spatial-Grid, Spatial-Map, Spatial-Real) using cloud and local VLMs. Three question modalities: VQA, VTQA, TQA.

```bash
# Download dataset (~1.4 GB)
uv run python spatial_eval/download_spatial_eval.py

# Quick smoke test (10 samples)
cd spatial_eval && bash scripts/smoke_test.sh

# Single inference run
uv run python inference_vlm.py --model_path gpt-4o --mode vqa --task mazenav --first_k 10
```

See [`spatial_eval/TESTING_GUIDE.md`](spatial_eval/TESTING_GUIDE.md) and [`spatial_eval/CONVENTIONS.md`](spatial_eval/CONVENTIONS.md) for full usage.

---

## Part 2 — MS Paint Reasoning Evaluation

Evaluates visual reasoning on MS Paint-style images with blur, greyscale, and inversion transformations across multiple models and reasoning effort levels.

```bash
# Smoke test
uv run python MS_Paint_Reasoning_Evaluation/Eval_script.py   --image-types color --blur-levels no_blur   --reasoning-effort low --skills yes --models gpt-5.2 --smoke

# Launch interactive dashboard
cd MS_Paint_Reasoning_Evaluation && docker-compose up -d
# → http://localhost:8050
```

See [`MS_Paint_Reasoning_Evaluation/TESTING_GUIDE.md`](MS_Paint_Reasoning_Evaluation/TESTING_GUIDE.md) and [`MS_Paint_Reasoning_Evaluation/CONVENTIONS.md`](MS_Paint_Reasoning_Evaluation/CONVENTIONS.md) for full usage.

---

## Part 3 — DeepAgent + SEM Service

An interactive LangGraph agent that controls a live SEM image annotation canvas. Supports tile-grid navigation, atlas mode, SAM2 segmentation, and histogram-based image quality evaluation.

### Services

| Service | Port | Description |
|---------|------|-------------|
| `sem-service` | 3000 | Fabric.js annotation canvas (Node.js/Express) |
| `agent-api` | 3001 | FastAPI DeepAgent wrapper with SAM2 segmentation endpoint |

### Start

```bash
docker-compose up -d        # starts sem-service + agent-api
python main.py              # interactive agent loop
```

Open the canvas at **http://localhost:3000**.

### Key agent tools

| Tool | Description |
|------|-------------|
| `paint_canvas` | Full canvas control: annotate, navigate tiles, atlas mode, filters, export |
| `segment_viewport` | SAM2 automatic segmentation overlaid on the current view |
| `get_sem_status` | Wait for canvas to settle then verify state — call after every action |

### Histogram evaluation (Case Study 1)

```bash
# Run from inside the sandbox after agent finishes
python /workspace/skills/master-skill/sem-histogram-eval/sem_histogram_error.py   --paint-url http://host.docker.internal:3000
```

See [`sem-service/README.md`](sem-service/README.md) for full API reference.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Browser (Human + Agent UI)   http://localhost:3000      │
│    Fabric.js canvas  ←── SSE ──────────────────────────┐ │
└──────────────────┬─────────────────────────────────────┘ │
                   │ REST API                               │
                   ▼                                        │
┌──────────────────────────────────────────────────────────┤
│  sem-service  (Node.js/Express, port 3000)               │
│   ├─ /api/canvas      state, load-image, crop, SSE       │
│   ├─ /api/draw        shape annotations                  │
│   ├─ /api/histogram   reference / randomized / result    │
│   ├─ /api/atlas       tile atlas mode                    │
│   ├─ /api/camera      tile grid navigation               │
│   └─ /api/export      PNG / JSON export                  │
└──────────────────────────────────────────────────────────┘
                   ▲
┌──────────────────┴─────────────────────────────────────┐
│  agent-api  (FastAPI, port 3001)                        │
│   ├─ POST /chat       LangGraph DeepAgent loop          │
│   ├─ POST /segment    SAM2 automatic segmentation       │
│   └─ POST /stop|reset agent lifecycle                   │
└──────────────────────────────────────────────────────────┘
                   ▲
┌──────────────────┴─────────────────────────────────────┐
│  main.py / llm_client.py                                │
│  agent_tools.py + agent_tools_vision.py                 │
│  skills/master-skill/                                   │
└──────────────────────────────────────────────────────────┘
```

---

## Documentation

| File | Contents |
|------|----------|
| [`sem-service/README.md`](sem-service/README.md) | SEM Service API reference and quick start |
| [`SANDBOX_SETUP_REVIEW.md`](SANDBOX_SETUP_REVIEW.md) | AIO Sandbox setup and agent capabilities |
| [`spatial_eval/TESTING_GUIDE.md`](spatial_eval/TESTING_GUIDE.md) | SpatialEval CLI reference |
| [`spatial_eval/CONVENTIONS.md`](spatial_eval/CONVENTIONS.md) | SpatialEval naming conventions |
| [`MS_Paint_Reasoning_Evaluation/TESTING_GUIDE.md`](MS_Paint_Reasoning_Evaluation/TESTING_GUIDE.md) | MS Paint eval CLI reference |
| [`MS_Paint_Reasoning_Evaluation/CONVENTIONS.md`](MS_Paint_Reasoning_Evaluation/CONVENTIONS.md) | MS Paint eval conventions |
| [`PLOTTING_GUIDE.md`](PLOTTING_GUIDE.md) | Visual standards for all plots |
| [`.github/copilot-instructions.md`](.github/copilot-instructions.md) | Full project technical reference |

---

## Grid Dataset

The tile-grid SEM dataset (`grid_dataset/`, ~13 GB, 5355 TIFF files) is **not tracked in git**. Contact the team for access or place tiles matching the pattern `Region{NNN}_y{YY}_x{XX}_fw{FW}um[_1].tiff` in `grid_dataset/`.

---

## References

- [AIO Sandbox Documentation](https://sandbox.agent-infra.com/)
- [SAM2 — facebook/sam2.1-hiera-small](https://github.com/facebookresearch/segment-anything-2)
- [Docker's Python guide](https://docs.docker.com/language/python/)
