from deepagents import create_deep_agent
from sandbox_backend import get_aio_sandbox_backend
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from agent_tools import (
    create_skill,
    get_sandbox_info,
    call_mcp_tool_in_sandbox,
)


def get_default_llm(model_name: str | None = None):
    """Get deep agent with AIOSandboxBackend so all tools execute in the sandbox."""
    if not model_name:
        model_name = MODEL_NAME
    
    checkpointer = MemorySaver()
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=model_name,
    )
    
    # Custom system prompt with sandbox capabilities
    system_prompt = """You are a helpful assistant with access to specialized skill files and an AIO Sandbox environment.

**CRITICAL RULE: YOU MUST ALWAYS CONSULT SKILLS BEFORE ACTING**

**MANDATORY WORKFLOW - FOLLOW EVERY TIME:**
1. **FIRST**: Read /workspace/skills/master-skill/SKILL.md to identify relevant skill
2. **SECOND**: Read the specific skill file to learn the exact commands/syntax
3. **THIRD**: Execute using built-in tools (glob, grep, read_file, write_file, execute)
4. **ALWAYS**: Cite which skill you used in your response

**IF YOU ACT WITHOUT READING THE RELEVANT SKILL FILE FIRST, YOU ARE DOING IT WRONG.**

**Built-in Tools Available:**
- glob: Find files matching patterns
- grep: Search file contents
- read_file: Read file content
- write_file: Write file content
- execute: Run shell commands or Python code

**Sandbox Filesystem:**
- Skills directory: /workspace/skills/
- Images directory: /workspace/pizza_not_pizza/
- Home directory: /home/gem/
- Working directory: /workspace/

**SKILL FILES ARE YOUR REFERENCE MANUAL:**
- **bash-scripting** → File operations, text processing, command examples
- **python-programming** → Python syntax, file I/O, data structures, HTTP
- **web-network** → curl, wget, API calls, web scraping examples
- **file-management** → Reading, writing, organizing files
- **pizza-making** → Pizza recipes and cooking methods

**CORRECT Example: "Write a Python script that prints hello world and execute it"**
```
Step 1: Read /workspace/skills/master-skill/SKILL.md (identifies bash-scripting and python-programming)
Step 2: Read /workspace/skills/master-skill/bash-scripting/SKILL.md (learn file creation syntax)
Step 3: Read /workspace/skills/master-skill/python-programming/SKILL.md (learn Python syntax)
Step 4: Use write_file to create /home/gem/hello.py with content: print("hello world")
Step 5: Use execute to run: python /home/gem/hello.py
Step 6: Respond: "Based on the bash-scripting skill for file creation and python-programming skill for syntax, I created and executed the script. Output: hello world"
```

**INCORRECT Example:**
❌ Directly executing commands without reading skills first
❌ Not citing which skill you referenced
❌ Acting from built-in knowledge instead of consulting the skill files

**REMEMBER: Skills contain the specific commands and patterns you should use. Always look them up first!**"""
    
    # Essential tools for unique functionality (basic ops handled by built-in tools)
    essential_tools = [
        create_skill,
        get_sandbox_info,
        call_mcp_tool_in_sandbox,
    ]
    
    agent = create_deep_agent(
        model=llm,
        checkpointer=checkpointer,
        system_prompt=system_prompt,
        skills=["skills"],
        backend=get_aio_sandbox_backend(),
        tools=essential_tools,
        debug=True,
    )

    return agent