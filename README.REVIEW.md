# aig-msd-visual-reasoning: Project Overview & Guide

---

## Overview

**aig-msd-visual-reasoning** is a modular Python project for evaluating and analyzing visual reasoning in AI models. It provides tools for image processing, spatial reasoning evaluation, sandboxed agent execution, and reproducible experiment management. The codebase is organized for extensibility, automation, and agent-driven workflows.

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

*This document is auto-generated for agent understanding. Please verify details with the codebase as needed.*
