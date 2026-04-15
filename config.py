from dotenv import load_dotenv
import os

load_dotenv()

# Access variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.2")
MODEL_REASONING_EFFORT = os.getenv("MODEL_REASONING_EFFORT", "medium")
OPIK_ENABLED = os.getenv("OPIK_ENABLED", "false").lower() == "true"
