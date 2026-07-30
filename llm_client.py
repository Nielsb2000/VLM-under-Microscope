from deepagents import create_deep_agent
from sandbox_backend import get_aio_sandbox_backend
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from agent_tools import (
    create_skill,
    get_sandbox_info,
    call_mcp_tool_in_sandbox,
    run_browser_steps,
    paint_canvas,
    get_sem_status,
)
from agent_tools_vision import (
    make_analyze_sandbox_image_tool,
    make_screenshot_and_ask_tool,
    make_move_and_verify_tool,
    make_segment_viewport_tool,
)


def get_default_llm(model_name: str | None = None):
    """Get deep agent with AIOSandboxBackend so all tools execute in the sandbox.

    Important:
    - Do not pass reasoning_effort here. gpt-5.5 rejects reasoning_effort when
      function tools are used through /v1/chat/completions.
    - Do not pass image_detail through ChatOpenAI model_kwargs. For chat
      completions, image detail belongs inside each multimodal image_url payload,
      which is handled in agent_tools_vision.py.
    """
    if not model_name:
        model_name = MODEL_NAME

    checkpointer = MemorySaver()
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=model_name,
        temperature=0.0,
    )

    # Expose model name for trace/manifest extraction in agent_api.py/package_run.py.
    try:
        setattr(llm, "model_name", model_name)
    except Exception:
        pass

    # Custom system prompt with sandbox capabilities
    system_prompt = """You are a helpful assistant with access to an AIO Sandbox environment and a set of modular skill files.

**Workflow:**
1. Read /workspace/skills/master-skill/SKILL.md to identify the relevant skill.
2. Read the specific sub-skill file for exact commands and patterns.
3. Execute using built-in tools: glob, grep, read_file, write_file, execute.

**Sandbox filesystem:**
- Skills: /workspace/skills/
- Images: /workspace/MS_paint_images/
- Home: /home/gem/
- Workspace: /workspace/

**Available skills:** bash-scripting, python-programming, web-network, file-management, sandbox-browser, sandbox-filesystem, mcp-tools.

**sem-service rule — always verify before acting:**
After every paint_canvas, segment_viewport, or camera navigation call, you MUST call get_sem_status (default settle_seconds=1.5) before your next action. Check that the returned state matches what you intended (correct background image, filter values, tile position, etc.). Only proceed once the state is confirmed. This ensures the browser has had time to apply and render the change."""

    analyze_sandbox_image = make_analyze_sandbox_image_tool(llm)
    screenshot_and_ask = make_screenshot_and_ask_tool(llm)
    move_and_verify = make_move_and_verify_tool(llm)
    segment_viewport = make_segment_viewport_tool()

    # Essential tools for unique functionality (basic ops handled by built-in tools)
    essential_tools = [
        create_skill,
        get_sandbox_info,
        call_mcp_tool_in_sandbox,
        analyze_sandbox_image,
        screenshot_and_ask,
        move_and_verify,
        run_browser_steps,
        paint_canvas,
        segment_viewport,
        get_sem_status,
    ]

    agent = create_deep_agent(
        model=llm,
        checkpointer=checkpointer,
        system_prompt=system_prompt,
        skills=["/workspace/skills"],
        backend=get_aio_sandbox_backend(),
        tools=essential_tools,
        debug=True,
    )

    return agent
