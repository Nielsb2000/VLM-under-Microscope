import logging
from llm_client import get_default_llm
from pathlib import Path
from skills_utils import discover_skills, get_skills_summary

LOG_FILE = "agent_session.log"


def _setup_logging() -> logging.Logger:
    """Configure root logger to write to both console and a fresh log file."""
    log_path = Path(LOG_FILE)
    log_path.unlink(missing_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Attach to root so deepagents/langgraph debug output is also captured
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    logger = logging.getLogger("agent")
    logger.info("Session started — log: %s", log_path.resolve())
    return logger


def main():
    logger = _setup_logging()
    print("Ask questions to the LLM (type 'quit' to exit)")
    print("💡 The agent can execute code and commands in the AIO Sandbox")
    print("   - Python code: runs in sandbox, output appears in sandbox terminal")
    print("   - Shell commands: executed in sandbox environment")
    print("   - Files: read/write in sandbox at /workspace/\n")
    
    # Discover and display available skills
    skills = discover_skills(Path("./skills"))
    if skills:
        print(get_skills_summary(skills))
        print()
    
    agent = get_default_llm()
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not question:
            continue
        
        try:
            logger.info("USER: %s", question)
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                },
                config={"configurable": {"thread_id": "default"}},
            )

            print("\n--- INTERMEDIATE STEPS ---")
            for key, value in result.items():
                if key != "messages":
                    print(f"{key}: {value}")
                    logger.debug("STEP %s: %s", key, value)
            print("--- END STEPS ---\n")

            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                print(f"Assistant: {last_message.content}\n")
                logger.info("ASSISTANT: %s", last_message.content)
        except Exception as e:
            import traceback
            print(f"Error: {type(e).__name__}: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            logger.exception("ERROR during agent invocation")
 

if __name__ == "__main__":
    main()
