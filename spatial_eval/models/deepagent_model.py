import os
import base64
import uuid
from io import BytesIO
from PIL import Image
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend


class DeepAgentGPT:
    """GPT Vision model augmented with spatial reasoning skills via deepagents.

    Drop-in replacement for GPT4Vision: exposes the same generate() interface
    so it can be used transparently in the inference_vlm.py main loop.

    The agent reads task-specific skill files (mazenav, spatialgrid, spatialmap)
    from spatial_eval/models/skills/ before answering each question.
    """

    SYSTEM_PROMPT = (
        "You are a spatial reasoning assistant with access to specialized skills.\n"
        "\n"
        "CRITICAL REQUIREMENT: Follow this EXACT workflow BEFORE answering:\n"
        "\n"
        "STEP 1: READ MASTER SKILL\n"
        "- MANDATORY: Call read_file to read skills/master-skill/SKILL.md\n"
        "- Identify which task skill you need based on the routing table\n"
        "\n"
        "STEP 2: READ TASK SKILL\n"
        "- Call read_file on the task-specific skill file listed in the routing table\n"
        "  (e.g., skills/mazenav/SKILL.md, skills/spatialgrid/SKILL.md, or skills/spatialmap/SKILL.md)\n"
        "- Study the solving strategy carefully\n"
        "\n"
        "STEP 3 (VQA mode only): READ EXAMPLE IMAGES\n"
        "- If the question contains NO textual representation (VQA mode), call read_file on each\n"
        "  example image listed in the task skill's 'Worked Examples' section\n"
        "- Study how the solved examples were approached and answered\n"
        "- Skip this step if you are in VTQA mode (textual data is already in the question)\n"
        "\n"
        "STEP 4: ANSWER THE QUESTION\n"
        "- Analyze the provided image inline in this message\n"
        "- For VTQA tasks, also use the textual representation embedded in the question text\n"
        "- Apply the solving strategy from the task skill\n"
        "- Give your final answer as: [Option Letter]. [Answer]\n"
        "  Examples: 'C. 2', 'A. Northeast', 'B. elephant'\n"
        "\n"
        "IMPORTANT:\n"
        "- Use read_file for skill files AND for example images listed in the skill — NOT for the actual question image\n"
        "- The actual question image is provided inline in this message (do not try to read_file it)\n"
        "- Always include the option letter (A/B/C/D) in your final answer\n"
        "- You WILL be penalized for skipping the skill reading steps\n"
        "- In your response, state which skills and example images you read and why before giving the final answer"
    )

    def __init__(self, model_name: str = "gpt-5.2", max_tokens: int = 1024):
        self.model_name = model_name
        self.max_tokens = max_tokens
        # root_dir = the directory containing this file (spatial_eval/models/)
        # skill paths are relative to this root, e.g. skills/master-skill/SKILL.md
        self._root_dir = os.path.dirname(os.path.abspath(__file__))

        self._llm = ChatOpenAI(model=model_name, max_tokens=max_tokens)
        backend = FilesystemBackend(root_dir=self._root_dir, virtual_mode=False)
        self._agent = create_deep_agent(
            model=self._llm,
            system_prompt=self.SYSTEM_PROMPT,
            skills=["skills/master-skill"],
            backend=backend,
            debug=False,
        )

    def _encode_image(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate(self, query_text: str, query_images, temperature: float = 0.2, max_new_tokens: int = None):
        """Generate an answer using the deepagents skill pipeline.

        Args:
            query_text: The question text (may include ASCII/textual map for VTQA).
            query_images: PIL Image object, or None for text-only (tqa) mode.
            temperature: Sampling temperature (not directly passed to agent; model uses default).
            max_new_tokens: Override max tokens for this call.

        Returns:
            Tuple of (prompt, answer_text) matching the GPT4Vision interface.
        """
        if max_new_tokens is not None:
            self._llm.max_tokens = max_new_tokens

        if query_images is not None:
            image_base64 = self._encode_image(query_images)
            content = [
                {"type": "text", "text": query_text},
                {"type": "image", "base64": image_base64, "mime_type": "image/png"},
            ]
        else:
            content = query_text

        agent_input = {
            "messages": [{"role": "user", "content": content}]
        }
        # Use a unique thread_id per call to avoid state leakage between questions
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        response = self._agent.invoke(agent_input, config=config)
        messages = response.get("messages", [])
        answer_text = messages[-1].content.strip() if messages else ""

        return query_text, answer_text
