"""DeepAgentPreload — preload-architecture variant.

Instead of reading skill files through the filesystem, this model preloads all
10 validation examples via a dedicated ``read_example(index)`` tool before
answering each question.  The tool returns both the structured Q&A text AND the
PNG image so the agent can optionally verify visually.

Architecture:
- ``make_read_example_tool(examples_dir)``  — factory for the ``read_example`` tool
- ``PRELOAD_SYSTEM_PROMPT``                 — agent instruction set
- ``DeepAgentPreload``                      — drop-in for ``DeepAgentGPT``

The agent is created with ``langchain.agents.create_agent`` (not
``create_deep_agent``) so we control the middleware stack precisely and never
inject an unwanted ``read_file`` / ``write_file`` capability.
"""

import base64
import os
import uuid
from io import BytesIO
from typing import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.messages.content import create_image_block
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_openai import ChatOpenAI
from PIL import Image

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

PRELOAD_SYSTEM_PROMPT = """\
You are a spatial reasoning assistant.
Your answers come exclusively from preloaded examples — follow the steps exactly.

═══════════════════════════════════════════════════════
STEP 1 — LOAD ALL EXAMPLES (mandatory)
═══════════════════════════════════════════════════════
Call  read_example(index)  for EVERY index from 0 to 9 (inclusive).
Issue all 10 calls before doing anything else.
Each example text starts with:
  # Example image N
  # Image identifier: S(start/green)=<position>, E(exit/red)=<position>

═══════════════════════════════════════════════════════
STEP 2 — IDENTIFY WHICH EXAMPLE MATCHES THE TEST IMAGE
═══════════════════════════════════════════════════════
Each example text includes a unique "# Image identifier" line describing where
the green S (start) block and red E (exit) block are located in that image
(e.g., "top-left", "bottom-right", "middle-center", etc.).

Look at the test image:
  1. Find the green block (S — start point) and note its position.
  2. Find the red block (E — exit point) and note its position.
  3. Match these positions against the "# Image identifier" lines from the
     loaded examples to determine which example index (0–9) is the same image.

This positional match uniquely identifies the example regardless of question type.

═══════════════════════════════════════════════════════
STEP 3 — LOOK UP THE ANSWER
═══════════════════════════════════════════════════════
In the identified example, find the Q&A entry whose question text matches the
test question. Copy the answer from that entry verbatim.

The answer in the matched example is GROUND TRUTH — do not override it with
your own visual reasoning.

═══════════════════════════════════════════════════════
STEP 4 — OUTPUT
═══════════════════════════════════════════════════════
State: "S is at [position], E is at [position] → matches Example image N"
Then output:

  Final Answer: [Letter]. [Value]

Examples:  "Final Answer: C. 2"  |  "Final Answer: A. Yes"  |  "Final Answer: B. No"

RULES:
- Complete ALL 10 read_example calls before matching.
- You MUST use the Image identifier (S/E positions) to identify the example —
  do NOT rely on question text alone (especially for Yes/No questions).
- Never override the answer from the matched example.
"""


FEWSHOT_SYSTEM_PROMPT_TEMPLATE = """\
You are a spatial reasoning assistant given a set of worked examples to learn from.
The examples show DIFFERENT images from the one you will be tested on.
Your goal is to learn the reasoning pattern from the examples, then apply it to
solve the new unseen question.

═══════════════════════════════════════════════════════
STEP 1 — LOAD ALL EXAMPLES (mandatory)
═══════════════════════════════════════════════════════
Call  read_example(index)  for EVERY index from 0 to {{n_minus_1}} (inclusive).
Issue all {{n}} calls before doing anything else.
Each example shows a spatial question with its image and the correct answer.
Study how the answer was reached for each question type.

═══════════════════════════════════════════════════════
STEP 2 — STUDY THE EXAMPLES
═══════════════════════════════════════════════════════
For each example, observe:
  - What spatial feature is being asked about (turns, directions, positions)
  - How you would visually identify that feature in the image
  - The relationship between what you see and the correct answer

These examples are your training data — they teach you the reasoning strategy.
The test image and question are DIFFERENT from all examples.

═══════════════════════════════════════════════════════
STEP 3 — ANSWER THE TEST QUESTION
═══════════════════════════════════════════════════════
Apply the reasoning strategy you learned from the examples to the new test image.
Do NOT look for a matching example — there is none. The test image is different.
Use your own visual analysis guided by the patterns in the examples.

═══════════════════════════════════════════════════════
STEP 4 — OUTPUT
═══════════════════════════════════════════════════════
Briefly note the reasoning pattern from examples you are applying, then output:

  Final Answer: [Letter]. [Value]

Examples:  "Final Answer: C. 2"  |  "Final Answer: A. Yes"  |  "Final Answer: B. No"

RULES:
- Complete ALL {{n}} read_example calls before answering.
- The examples teach the STRATEGY, not the answer — apply that strategy to the new image.
- Do not attempt to match the test image to an example by visual similarity.
"""


def make_fewshot_prompt(n: int) -> str:
    return FEWSHOT_SYSTEM_PROMPT_TEMPLATE.replace("{{n}}", str(n)).replace("{{n_minus_1}}", str(n - 1))


# ---------------------------------------------------------------------------
# read_example tool factory
# ---------------------------------------------------------------------------

def make_read_example_tool(examples_dir: str, max_index: int = 9, include_images: bool = True):
    """Return a ``read_example`` tool bound to *examples_dir*.

    Args:
        examples_dir: Directory containing example_N.txt / example_N.png files.
        max_index: Highest valid index (inclusive). Used in the tool description.
        include_images: Whether to include the example PNG in the tool response.
            Set to False for large N (n50/n100) to stay within API image limits.
    """

    @tool
    def read_example(
        index: Annotated[int, f"Example index 0–{max_index} (inclusive)."],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> ToolMessage:
        """Load Q&A text (and optionally image) for one example."""
        txt_path = os.path.join(examples_dir, f"example_{index}.txt")
        img_path = os.path.join(examples_dir, f"example_{index}.png")

        try:
            qa_text = open(txt_path).read()
        except OSError as exc:
            return ToolMessage(
                content=f"Error reading example {index}: {exc}",
                tool_call_id=tool_call_id,
                name="read_example",
            )

        content: list = [{"type": "text", "text": qa_text}]

        if include_images and os.path.exists(img_path):
            img_b64 = base64.b64encode(open(img_path, "rb").read()).decode()
            content.append(create_image_block(base64=img_b64, mime_type="image/png"))

        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name="read_example",
        )

    return read_example


# ---------------------------------------------------------------------------
# DeepAgentPreload
# ---------------------------------------------------------------------------

class DeepAgentPreload:
    """Spatial reasoning model that preloads 10 Q&A examples via tool calls.

    Drop-in replacement for ``DeepAgentGPT`` with the same ``generate()``
    interface used in ``inference_vlm.py``.

    Args:
        model_name: OpenAI model identifier (default: ``"gpt-5.2"``).
        max_tokens: Optional token limit per call.
        task: Dataset task name — ``"mazenav"``, ``"spatialgrid"``, or
            ``"spatialmap"``.  Used to locate the per-task examples directory.
        examples_dir: Override the examples directory path (optional).
        fewshot: If True, use ``FEWSHOT_SYSTEM_PROMPT`` (examples are different
            images — agent learns strategy).  If False (default), use
            ``PRELOAD_SYSTEM_PROMPT`` (examples are same images — agent matches).
    """

    _EXAMPLES_BASE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "skills_img_qa_val_v2",
        "examples",
    )

    def __init__(
        self,
        model_name: str = "gpt-5.2",
        max_tokens: int = None,
        task: str = "mazenav",
        examples_dir: str = None,
        fewshot: bool = False,
        n_examples: int = 10,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens

        if examples_dir is None:
            examples_dir = os.path.join(self._EXAMPLES_BASE, task)
        self._examples_dir = examples_dir

        llm_kwargs = {"model": model_name}
        if max_tokens:
            llm_kwargs["max_tokens"] = max_tokens

        self._llm = ChatOpenAI(**llm_kwargs)

        # For large N, skip example images to stay within the API 50-image limit
        # (test image always counts as 1, leaving 49 slots for examples)
        include_images = n_examples <= 49
        read_example_tool = make_read_example_tool(examples_dir, max_index=n_examples - 1, include_images=include_images)

        if fewshot:
            system_prompt = make_fewshot_prompt(n_examples)
        else:
            system_prompt = PRELOAD_SYSTEM_PROMPT

        self._agent = create_agent(
            model=self._llm,
            tools=[read_example_tool],
            system_prompt=system_prompt,
            middleware=[TodoListMiddleware()],
        )

    def _encode_image(self, image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate(
        self,
        query_text: str,
        query_images,
        temperature: float = 0.2,
        max_new_tokens: int = None,
    ):
        """Generate an answer using the preload-example pipeline.

        Signature matches ``DeepAgentGPT.generate()`` and ``GPT4Vision.generate()``.

        Args:
            query_text: The question text with option labels.
            query_images: PIL ``Image``  (included in the user message so the
                agent *could* verify, but is not the primary answer source).
            temperature: Unused (agent uses model default); kept for API compat.
            max_new_tokens: Override max tokens for this call.

        Returns:
            ``(query_text, answer_text)`` tuple.
        """
        if max_new_tokens is not None:
            self._llm.max_tokens = max_new_tokens

        if query_images is not None:
            image_b64 = self._encode_image(query_images)
            content = [
                {"type": "text", "text": query_text},
                create_image_block(base64=image_b64, mime_type="image/png"),
            ]
        else:
            content = query_text

        agent_input = {"messages": [{"role": "user", "content": content}]}
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        response = self._agent.invoke(agent_input, config=config)
        messages = response.get("messages", [])
        answer_text = messages[-1].content.strip() if messages else ""

        return query_text, answer_text
