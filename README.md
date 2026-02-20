
# aig-msd-visual-reasoning: Visual Spatial Reasoning for SEM/TEM Microscopy

---

## Introduction

**aig-msd-visual-reasoning** is a research-driven Python project focused on advancing visual spatial reasoning in AI models, with a special emphasis on applications in Scanning Electron Microscopy (SEM) and Transmission Electron Microscopy (TEM). The project leverages both proprietary cloud models and local vision-language models to build, evaluate, and benchmark spatial reasoning capabilities for scientific imaging.

Our goal is to develop a model architecture and skill suite that enables robust spatial reasoning and visual understanding of SEM/TEM images, supporting downstream tasks in scientific analysis, automation, and agent-based workflows.

---

## Project Goals & Vision

- **Spatial Reasoning for Microscopy:** Enable AI models to interpret, reason, and act on SEM/TEM images, supporting scientific discovery and automation.
- **Proprietary & Local Models:** Integrate cloud-based proprietary models (e.g., GPT-4o, Azure) and local VLMs (e.g., LLaVA, Bunny) for flexible experimentation.
- **Skill-Based Architecture:** Build a collection of modular skills for spatial reasoning, image processing, and agent execution.
- **Benchmarking & Evaluation:** Use curated test cases (MS Paint images, SpatialEval) to rigorously evaluate spatial reasoning performance.
- **Agent-Driven Workflows:** Support reproducible, extensible, and sandboxed agent workflows for scientific and industrial use.

---

## Key Features & Components

- **MS_Paint_Reasoning_Evaluation/**: Evaluate visual reasoning using MS Paint-style images. Includes blurring, evaluation, plotting, and Excel export tools.
- **spatial_eval/**: Evaluate spatial reasoning (VQA/VTQA) in vision-language models. Supports batch inference, evaluation, and result visualization.
- **skills/**: Modular skills for agent execution in the sandbox.
- **pizza_not_pizza/**: Example dataset for image classification tasks.
- **SANDBOX_SETUP.md**: Guide for running the agent in a secure Docker sandbox.
- **README.Docker.md**: Instructions for Docker-based development and deployment.

---

## Architecture Diagram (ASCII)

```
[User/Agent]
    |
    v
[main.py / llm_client.py]
    |
    v
[Evaluation Modules]
  |         |
  v         v
MS_Paint    spatial_eval
  |             |
  v             v
[Results/Plots/Exports]
    |
    v
[Reproducible Outputs]
```

---

## Quick Start

### 1. Setup Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
uv sync
```

### 2. Run in Sandbox (Recommended)
```bash
docker-compose up -d
python main.py
```

### 3. Run Evaluation Scripts
- See MS_Paint_Reasoning_Evaluation/MS_paint_review.md and spatial_eval/spatial_eval_review.md for detailed usage.

---

## Documentation

- [MS_Paint_Reasoning_Evaluation/MS_paint_review.md](MS_Paint_Reasoning_Evaluation/MS_paint_review.md): Technical review and usage for MS Paint evaluation.
- [spatial_eval/spatial_eval_review.md](spatial_eval/spatial_eval_review.md): Technical review and usage for spatial evaluation.
- [SANDBOX_SETUP_REVIEW.md](SANDBOX_SETUP_REVIEW.md): Sandbox setup and agent capabilities.
- [README.Docker.REVIEW.md](README.Docker.REVIEW.md): Docker build, run, and deployment guide.
- [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md): Symbol and structure reference.
- [CODEBASE_TECHNICAL_REVIEW.md](CODEBASE_TECHNICAL_REVIEW.md): In-depth technical review.

---

## Extensibility & Best Practices

- **Modular Design:** Add new models, datasets, or skills with minimal changes.
- **CLI-Driven:** All major scripts support command-line arguments for automation.
- **Reproducibility:** Results are organized by run/model/config for easy comparison.
- **Agent-Ready:** Designed for LLM agent workflows and sandboxed execution.

---

## References
- [AIO Sandbox Documentation](https://sandbox.agent-infra.com/)
- [Docker's Python guide](https://docs.docker.com/language/python/)

---

## Key Components

- **MS_Paint_Reasoning_Evaluation/**: Evaluate visual reasoning using MS Paint-style images. Includes blurring, evaluation, plotting, and Excel export tools.
- **spatial_eval/**: Evaluate spatial reasoning (VQA/VTQA) in vision-language models. Supports batch inference, evaluation, and result visualization.
- **skills/**: Modular skills for agent execution in the sandbox.
- **pizza_not_pizza/**: Example dataset for image classification tasks.
- **SANDBOX_SETUP.md**: Guide for running the agent in a secure Docker sandbox.
- **README.Docker.md**: Instructions for Docker-based development and deployment.

---

## Architecture Diagram (ASCII)

```
[User/Agent]
    |
    v
[main.py / llm_client.py]
    |
    v
[Evaluation Modules]
  |         |
  v         v
MS_Paint    spatial_eval
  |             |
  v             v
[Results/Plots/Exports]
    |
    v
[Reproducible Outputs]
```

---

## Quick Start

### 1. Setup Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
uv sync
```

### 2. Run in Sandbox (Recommended)
```bash
docker-compose up -d
python main.py
```

### 3. Run Evaluation Scripts
- See MS_Paint_Reasoning_Evaluation/MS_paint_review.md and spatial_eval/spatial_eval_review.md for detailed usage.

---

## Documentation

- [MS_Paint_Reasoning_Evaluation/MS_paint_review.md](MS_Paint_Reasoning_Evaluation/MS_paint_review.md): Technical review and usage for MS Paint evaluation.
- [spatial_eval/spatial_eval_review.md](spatial_eval/spatial_eval_review.md): Technical review and usage for spatial evaluation.
- [SANDBOX_SETUP_REVIEW.md](SANDBOX_SETUP_REVIEW.md): Sandbox setup and agent capabilities.
- [README.Docker.REVIEW.md](README.Docker.REVIEW.md): Docker build, run, and deployment guide.
- [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md): Symbol and structure reference.
- [CODEBASE_TECHNICAL_REVIEW.md](CODEBASE_TECHNICAL_REVIEW.md): In-depth technical review.

---

## Extensibility & Best Practices

- **Modular Design:** Add new models, datasets, or skills with minimal changes.
- **CLI-Driven:** All major scripts support command-line arguments for automation.
- **Reproducibility:** Results are organized by run/model/config for easy comparison.
- **Agent-Ready:** Designed for LLM agent workflows and sandboxed execution.

---

## References
- [AIO Sandbox Documentation](https://sandbox.agent-infra.com/)
- [Docker's Python guide](https://docs.docker.com/language/python/)

---

