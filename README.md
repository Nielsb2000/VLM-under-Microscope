# VLM-under-Microscope: AI-Assisted Visual Spatial Reasoning for SEM/TEM

## Introduction
Hi! This repository contains the work from my internship research project. The goal of this project was to advance **visual spatial reasoning** in AI models (VLMs) when applied to scientific imaging, specifically Scanning Electron Microscopy (SEM) and Transmission Electron Microscopy (TEM).

The project is divided into three core components, ranging from standardized benchmarking to an interactive AI-driven annotation agent.

---

## 🚀 Getting Started

### Prerequisites
- **Python**: \`>=3.11,<3.12\`
- **Package Manager**: [\`uv\`](https://github.com/astral-sh/uv)
- **Containerization**: Docker & Docker Compose

### Fast Installation
\`\`\`bash
# Clone and sync dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env 
# Edit .env with your OPENAI_API_KEY, MODEL_NAME, etc.
\`\`\`

---

## 🔬 Core Components

### 1. SpatialEval (Spatial Reasoning Benchmark)
A research-grade benchmark evaluating models on 4 tasks: \`mazenav\`, \`spatialgrid\`, \`spatialmap\`, and \`spatialreal\`. It supports baseline VLM calls and **DeepAgent** spatial skills.

**Key Commands:**
\`\`\`bash
# 1. Download the dataset (~1.4 GB)
uv run python spatial_eval/download_spatial_eval.py

# 2. Run a smoke test (Quick sanity check)
cd spatial_eval && bash scripts/smoke_test.sh

# 3. Main experiment (Baseline vs Skills)
cd spatial_eval && bash scripts/run_experiment.sh

# 4. Generate results visualization
uv run python spatial_eval/eval_summary/compute_and_plot.py
\`\`\`

### 2. MS Paint Reasoning Evaluation
Evaluates how image degradation (blur, greyscale, inversion) and reasoning effort affect VLM performance on geometric layouts.

**Key Commands:**
\`\`\`bash
# 1. Run a smoke test
uv run python MS_Paint_Reasoning_Evaluation/Eval_script.py \
  --image-types color --blur-levels no_blur \
  --reasoning-effort low --skills yes --models gpt-5.2 --smoke

# 2. Launch results dashboard (Interactive Plotly/Dash)
cd MS_Paint_Reasoning_Evaluation && docker-compose up -d
# View at http://localhost:8050
\`\`\`

### 3. Case Studies & DeepAgent Framework
This is the "active" part of the project where an AI agent interacts with a simulated SEM environment.

#### **AIO Sandbox & SEM Service**
The agent operates within a Dockerized sandbox and communicates with a custom \`sem-service\` canvas.

**Setup Services:**
\`\`\`bash
# Start the SEM Service, Agent API, and AIO Sandbox
docker-compose up -d
\`\`\`

**Running Case Studies:**
During the internship, four primary case studies were evaluated. You can reproduce the agent executions using these scripts:
\`\`\`bash
# Case Study 1: Contact Sheet Generation
uv run python run_case_study_1.py

# Case Study 2: Tile-Grid Navigation & Multi-Mag Analysis
uv run python run_case_study_2.py

# Case Study 3: Atlas Exploration
uv run python run_case_study_3.py

# Case Study 4: Automated Reporting
uv run python run_case_study_4.py
\`\`\`

---

## 🛠️ Specialized Installation Details

### AIO Sandbox (DeepAgent Execution)
The \`AIO Sandbox\` is a high-security environment for code execution. It is managed via \`docker-compose.yml\` in the root. Verify connectivity:
\`\`\`bash
uv run python test_sandbox.py
\`\`\`

### SEM Service
A Node.js/Express service providing a Fabric.js canvas for the agent to "see" and annotate.
- **Canvas UI**: \`http://localhost:3000\`
- **Agent API**: \`http://localhost:3001\`

---

## 📈 Visualizations
Visualization code for the Case Studies is located in \`outputs/\`.
- [PLOTTING_GUIDE.md](PLOTTING_GUIDE.md) contains details on how to regenerate figures for the report.

---
*Created by Niels Broekhuizen during the 2026 AI Internship.*
