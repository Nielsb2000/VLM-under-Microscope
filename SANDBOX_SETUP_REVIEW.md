# SANDBOX_SETUP.md: Technical Review & Implementation Guide

---

## Overview

**SANDBOX_SETUP.md** documents the setup and operation of the AIO Sandbox for isolated, agent-driven code execution. The sandbox enables the LLM agent to safely run Python code, shell commands, and manage files in a Dockerized environment, with clear separation between host and sandboxed resources.

---

## Architecture & Data Flow

```
┌─────────────────────────────┐
│  Host (VS Code + Agent)     │
│ ┌─────────────────────────┐ │
│ │ main.py (agent)         │ │
│ │  ↓                      │ │
│ │ llm_client.py           │ │
│ │  ↓                      │ │
│ │ sandbox_tools (API) ────┼─┼─────┐
│ └─────────────────────────┘ │     │
└─────────────────────────────┘     │ HTTP API
                                    ▼
┌───────────────────────────────────┐
│ Docker: AIO Sandbox Container     │
│ ┌───────────────────────────────┐ │
│ │ Sandbox API (port 8080)      │ │
│ │  • Python exec, shell, files │ │
│ │  • Skill mgmt, info          │ │
│ └───────────────────────────────┘ │
│  /workspace/skills/ (rw)         │
│  /workspace/pizza_not_pizza/ (ro)│
└───────────────────────────────────┘
```

---

## Key Capabilities

- **run_python_in_sandbox**: Execute Python code in the sandbox
- **run_shell_in_sandbox**: Run shell commands in the sandbox
- **read_file_from_sandbox** / **write_file_to_sandbox**: File I/O
- **list_sandbox_files**: Directory listing
- **create_skill**: Add new skills to `/workspace/skills/`
- **get_sandbox_info**: Query sandbox environment

---

## Setup & Usage

### 1. Start the Sandbox
```bash
docker-compose up -d
```

### 2. Verify Sandbox
```bash
curl http://localhost:8080/v1/sandbox
```

### 3. Install Python Dependencies
```bash
source .venv/bin/activate
uv sync
```

### 4. Run the Agent
```bash
python main.py
```

---

## Mounted Directories

| Local Path           | Sandbox Path                  | Access     |
|----------------------|------------------------------|------------|
| `./skills/`          | `/workspace/skills/`         | Read/Write |
| `./pizza_not_pizza/` | `/workspace/pizza_not_pizza/`| Read-only  |

---

## Example Agent Workflows

- **Create and run a Python script:**
  1. `write_file_to_sandbox` → `/workspace/hello.py`
  2. `run_python_in_sandbox` → `/workspace/hello.py`
- **List pizza images:**
  1. `list_sandbox_files` → `/workspace/pizza_not_pizza/pizza/`

---

## Management & Troubleshooting

- **View logs:** `docker-compose logs -f aio-sandbox`
- **Stop:** `docker-compose down`
- **Restart:** `docker-compose restart`
- **Shell:** `docker exec -it aio-sandbox bash`

---

## Environment Variables

Set in `.env`:
- `OPENAI_API_KEY`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, `MODEL_NAME`
- `SANDBOX_BASE_URL` (default: `http://localhost:8080`)

---

## References
- [AIO Sandbox Docs](https://sandbox.agent-infra.com/)
- [AIO Sandbox GitHub](https://github.com/agent-infra/sandbox)
- [Python SDK](https://github.com/agent-infra/sandbox/tree/main/sdk/python)

---


