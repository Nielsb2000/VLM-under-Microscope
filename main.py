import argparse
import logging
import sys
from datetime import datetime
from llm_client import get_default_llm
from pathlib import Path
from skills_utils import discover_skills, get_skills_summary

LOG_FILE = "agent_session.log"
LOGS_DIR = Path("logs")


class _Tee:
    """Write to both a real stream and a file simultaneously."""
    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _setup_logging(enable_log_file: bool) -> logging.Logger:
    """Configure root logger; optionally tee stdout/stderr to a timestamped .txt in logs/."""
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Always keep the lightweight agent_session.log (DEBUG, overwritten each run)
    log_path = Path(LOG_FILE)
    log_path.unlink(missing_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if enable_log_file:
        LOGS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_log = LOGS_DIR / f"session_{ts}.txt"
        # File handler that also captures everything at DEBUG level
        full_handler = logging.FileHandler(full_log, mode="w", encoding="utf-8")
        full_handler.setLevel(logging.DEBUG)
        full_handler.setFormatter(fmt)
        root.addHandler(full_handler)
        # Tee stdout and stderr so every print() also lands in the file
        f = open(full_log, "a", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, f)
        sys.stderr = _Tee(sys.__stderr__, f)
        print(f"[logging] Session log: {full_log.resolve()}")

    logger = logging.getLogger("agent")
    logger.info("Session started")
    return logger


def _parse_args():
    parser = argparse.ArgumentParser(description="DeepAgent interactive loop")
    parser.add_argument(
        "--log",
        type=lambda v: v.lower() in ("1", "true", "yes"),
        default=False,
        metavar="BOOL",
        help="Write full session log (all output + tool calls) to logs/session_<ts>.txt",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    logger = _setup_logging(args.log)
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
