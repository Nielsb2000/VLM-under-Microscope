import os
import base64
import time
import uuid
from io import BytesIO
from PIL import Image
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from .gpt4_model import MAX_RETRIES, RETRY_DELAY, _is_retryable


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

    # Special lookup-mode prompt for img-qa-val: the skill IS the answer key.
    _VAL_LOOKUP_PROMPT = (
        "You are a spatial reasoning assistant. For this task the skill file contains a LOOKUP TABLE "
        "with the pre-computed correct answer for every question you will receive.\n"
        "\n"
        "STEP 1: READ MASTER SKILL\n"
        "- Call read_file on skills/master-skill/SKILL.md\n"
        "- Identify the task skill path from the routing table\n"
        "\n"
        "STEP 2: READ TASK SKILL\n"
        "- Call read_file on the task skill (e.g. skills/mazenav/SKILL.md)\n"
        "- The skill contains two tables at the top: Table 1 (4-option questions) and Table 2 (Yes/No questions)\n"
        "\n"
        "STEP 3: LOOK UP YOUR ANSWER\n"
        "- For 4-option questions (A/B/C/D with numbers):\n"
        "  a) Copy the EXACT options string from your question: \"A. X | B. Y | C. Z | D. W\"\n"
        "  b) Find the row in Table 1 whose first column is CHARACTER-FOR-CHARACTER IDENTICAL to your options string\n"
        "  c) The Answer column of that row IS your final answer — copy it verbatim\n"
        "  d) DO NOT look at the image. DO NOT reason. Just match and copy.\n"
        "- For Yes/No questions (A. Yes | B. No):\n"
        "  a) Find the direction phrase from your question in Table 2\n"
        "  b) If the phrase is marked DISAMBIGUATION NEEDED, inspect the image to pick the correct row\n"
        "  c) Copy the Answer from that row\n"
        "\n"
        "STEP 4: OUTPUT\n"
        "- State: \"Table match: [your options string] → [answer]\"\n"
        "- Then output: Final Answer: [Letter]. [Value]\n"
        "\n"
        "CRITICAL RULES:\n"
        "- For Table 1 entries: the answer is already known — DO NOT verify against the image\n"
        "- Match the FULL options string including ALL four values (A, B, C AND D must all match)\n"
        "- Your answer is not valid if it does not end with: Final Answer: [Letter]. [Value]"
    )

    # Maps --skills_variant CLI value to the self-contained skill folder under models/
    _VARIANT_SKILL_FOLDERS = {
        "img-only":      "skills_img_only",
        "img-qa":        "skills_img_qa",
        "img-context":   "skills_img_context",
        # Test 1 — validation (same images in skill and test)
        "img-qa-val":    "skills_img_qa_val",
        # Test 2 — img-only range test (N example images, tested at offset N)
        "img-only-n3":   "skills_img_only_n3",
        "img-only-n10":  "skills_img_only_n10",
        "img-only-n30":  "skills_img_only_n30",
        "img-only-annotated": "skills_img_only_annotated",
        "img-annotated-context": "skills_img_annotated_context",
    }

    def __init__(self, model_name: str = "gpt-5.2", max_tokens: int = None, skills_variant: str = None, debug_log_path: str = None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._debug_log_path = debug_log_path
        self._debug_item_count = 0
        _models_dir = os.path.dirname(os.path.abspath(__file__))

        if skills_variant in self._VARIANT_SKILL_FOLDERS:
            root_dir = os.path.join(_models_dir, self._VARIANT_SKILL_FOLDERS[skills_variant])
        else:
            raise ValueError(
                f"skills_variant={skills_variant!r} is not recognised. "
                f"Valid options: {list(self._VARIANT_SKILL_FOLDERS)}"
            )

        self._root_dir = root_dir
        self._llm = ChatOpenAI(model=model_name)
        backend = FilesystemBackend(root_dir=root_dir, virtual_mode=True)
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

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._agent.invoke(agent_input, config=config)
                break
            except Exception as exc:
                if attempt < MAX_RETRIES and _is_retryable(exc):
                    print(f"[retry {attempt}/{MAX_RETRIES}] {type(exc).__name__}: {exc} — retrying in {RETRY_DELAY}s")
                    time.sleep(RETRY_DELAY)
                    # New thread_id so the agent starts fresh on retry
                    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                else:
                    raise

        messages = response.get("messages", [])
        answer_text = messages[-1].content.strip() if messages else ""

        if self._debug_log_path and messages:
            self._debug_item_count += 1
            self._write_debug_log(messages)

        return query_text, answer_text

    def _write_debug_log(self, messages):
        """Append a structured trace of all agent messages to the debug log file."""
        import json as _json
        import hashlib as _hashlib
        import base64 as _base64

        def _image_fingerprint(block: dict) -> str:
            """Return size + short SHA256 so we can verify images are distinct."""
            # base64 data may live at different keys depending on deepagents version
            raw_b64 = (
                block.get("base64")
                or block.get("data")
                or (block.get("image_url") or {}).get("url", "").split(",")[-1]
            )
            if not raw_b64:
                return "size=unknown  sha256=unknown"
            try:
                img_bytes = _base64.b64decode(raw_b64)
            except Exception:
                img_bytes = raw_b64.encode()
            size = len(img_bytes)
            digest = _hashlib.sha256(img_bytes).hexdigest()[:12]
            return f"size={size}B  sha256={digest}"

        os.makedirs(os.path.dirname(self._debug_log_path), exist_ok=True)
        with open(self._debug_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ITEM #{self._debug_item_count}  ({len(messages)} messages)\n")
            f.write(f"{'='*80}\n")
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                f.write(f"\n--- [{i}] {msg_type} ---\n")
                # Tool calls attached to AIMessage
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    f.write("  tool_calls:\n")
                    for tc in tool_calls:
                        f.write(f"    name: {tc.get('name', tc)}\n")
                        args = tc.get("args", {})
                        f.write(f"    args: {_json.dumps(args, ensure_ascii=False)[:2000]}\n")
                # Content: may be str or list of content blocks
                content = msg.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "unknown")
                            if btype == "text":
                                f.write(f"  [text] {block.get('text', '')[:2000]}\n")
                            elif btype in ("image", "image_url"):
                                fp = _image_fingerprint(block)
                                f.write(f"  [image block – {fp}]\n")
                            else:
                                f.write(f"  [{btype}] {str(block)[:500]}\n")
                        else:
                            f.write(f"  {str(block)[:500]}\n")
                elif content:
                    f.write(f"  {str(content)[:3000]}\n")
