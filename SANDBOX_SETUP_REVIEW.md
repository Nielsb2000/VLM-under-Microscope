# AIO Sandbox Setup

This project uses the AIO Sandbox for isolated code execution. The LLM agent runs locally in VS Code but executes all code and commands inside the sandbox container.

## Quick Start

### 1. Start the Sandbox

```bash
docker-compose up -d
```

This starts the AIO Sandbox container with:
- **Port 8080**: Sandbox API and web interface
- **Mounted volumes**: 
  - `./skills` → `/workspace/skills` (read/write)
  - `./pizza_not_pizza` → `/workspace/pizza_not_pizza` (read-only images)

### 2. Verify Sandbox is Running

```bash
curl http://localhost:8080/v1/sandbox
```

Or visit in browser:
- **API Docs**: http://localhost:8080/v1/docs
- **VNC Browser**: http://localhost:8080/vnc/index.html?autoconnect=true
- **VSCode Server**: http://localhost:8080/code-server/

### 3. Install Python Dependencies

```bash
# Make sure you're in your virtual environment
source .venv/bin/activate

# Install dependencies including agent-sandbox SDK
uv sync
```

### 4. Run the Agent

```bash
python main.py
```

## How It Works

### Architecture

```
┌─────────────────────────────────────┐
│   Your Computer (Host)               │
│                                      │
│  ┌──────────────────────────────┐   │
│  │  main.py (runs locally)      │   │
│  │    ↓                          │   │
│  │  llm_client.py               │   │
│  │    ↓                          │   │
│  │  Agent with sandbox_tools ───┼───┼──┐
│  └──────────────────────────────┘   │  │
└──────────────────────────────────────┘  │
                                          │ HTTP API
┌─────────────────────────────────────────┼─┐
│   Docker: AIO Sandbox Container         │ │
│                                         ▼ │
│  ┌──────────────────────────────────────┤
│  │  Sandbox API (port 8080)            │
│  │    • Execute Python code            │
│  │    • Run shell commands             │
│  │    • File operations                │
│  │    • Create skills                  │
│  └─────────────────────────────────────┤
│                                         │
│  Filesystem:                            │
│    /workspace/skills/                   │
│    /workspace/pizza_not_pizza/          │
│    /home/gem/                           │
└─────────────────────────────────────────┘
```

### Agent Capabilities

The agent has access to these sandbox tools:

1. **run_python_in_sandbox**: Execute Python code in sandbox
2. **run_shell_in_sandbox**: Execute shell commands in sandbox
3. **read_file_from_sandbox**: Read files from sandbox filesystem
4. **write_file_to_sandbox**: Write files to sandbox filesystem
5. **list_sandbox_files**: List files in sandbox directories
6. **create_skill**: Create new skills in `/workspace/skills/`
7. **get_sandbox_info**: Get sandbox environment information

### Example Usage

**User**: "Create a hello world Python script in the sandbox and run it"

**Agent will**:
1. Use `write_file_to_sandbox` to create `/workspace/hello.py`
2. Use `run_python_in_sandbox` to execute the script
3. Output appears in sandbox terminal

**User**: "List all pizza images"

**Agent will**:
1. Use `list_sandbox_files` with directory `/workspace/pizza_not_pizza/pizza/`
2. Return list of pizza image files

## Mounted Directories

| Local Path | Sandbox Path | Access |
|------------|--------------|--------|
| `./skills/` | `/workspace/skills/` | Read/Write |
| `./pizza_not_pizza/` | `/workspace/pizza_not_pizza/` | Read-only |

## Sandbox Management

### View Logs
```bash
docker-compose logs -f aio-sandbox
```

### Stop Sandbox
```bash
docker-compose down
```

### Restart Sandbox
```bash
docker-compose restart
```

### Access Sandbox Shell
```bash
docker exec -it aio-sandbox bash
```

## Environment Variables

Set these in your `.env` file:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_key_here
OPENAI_ORG_ID=your_org_id
OPENAI_PROJECT_ID=your_project_id
MODEL_NAME=gpt-4

# Sandbox Configuration (optional)
SANDBOX_BASE_URL=http://localhost:8080
```

## Troubleshooting

### Sandbox won't start
```bash
# Check if port 8080 is already in use
sudo lsof -i :8080

# Check Docker logs
docker-compose logs aio-sandbox
```

### Agent can't connect to sandbox
```bash
# Verify sandbox is running
curl http://localhost:8080/v1/sandbox

# Check environment variable
echo $SANDBOX_BASE_URL
```

### Skills not visible in sandbox
```bash
# Verify volume mount
docker exec -it aio-sandbox ls -la /workspace/skills/

# Check if local skills directory exists
ls -la ./skills/
```

## References

- [AIO Sandbox Documentation](https://sandbox.agent-infra.com/)
- [AIO Sandbox GitHub](https://github.com/agent-infra/sandbox)
- [Python SDK Documentation](https://github.com/agent-infra/sandbox/tree/main/sdk/python)
