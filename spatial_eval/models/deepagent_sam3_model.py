"""DeepAgentSAM3 — vision agent with on-demand SAM 3 segmentation tool.

The agent receives an image + question and autonomously decides which objects
to segment by calling the ``segment(text_prompt)`` tool one or more times.
No task-specific knowledge is baked into the system prompt — the agent infers
what to look for from the question itself, making the approach task-agnostic.

Drop-in replacement for DeepAgentGPT / DeepAgentPreload: exposes the same
``generate(query_text, query_images, temperature, max_new_tokens)`` interface
used in inference_vlm.py.

Usage (inference_vlm.py):
    uv run python inference_vlm.py \\
        --model_path gpt-5.2 --mode vqa --task mazenav \\
        --first_k 10 --use_skills --skills_variant sam3
"""

import base64
import io
import os
import random
import uuid

import numpy as np
import torch
from PIL import Image

# ── System prompt ─────────────────────────────────────────────────────────────
# Deliberately task-agnostic: no maze vocabulary, no hardcoded object names.
# The agent must infer what to segment from the question it receives.

SYSTEM_PROMPT = """\
You are a visual reasoning assistant with access to a segmentation tool.

When given an image and a question, follow these steps:

STEP 1 — UNDERSTAND THE QUESTION
Read the question carefully. Identify which objects, regions, or visual elements
you need to locate in the image to answer it.

STEP 2 — SEGMENT RELEVANT OBJECTS
Call the `segment_batch` tool ONCE with a list of all the object descriptions
you need (e.g. ["red block", "entrance door", "largest dark region"]).
  - The tool segments all prompts in a single pass and returns one overlay image
    and centroid report per prompt.
  - If an object is not found (0 instances), rephrase that prompt and call
    segment_batch again with just the missing ones.

STEP 3 — REASON AND ANSWER
Use the original image, the segmented overlays, and the reported centroids to
reason about the question. Centroids are in pixel space (x = left→right,
y = top→bottom). Two objects share a column if |Δx| ≤ 10 px; same row if
|Δy| ≤ 10 px.

STEP 4 — OUTPUT YOUR FINAL ANSWER
End your response with your final answer on its own last line as ONLY the option
letter and its text.
Examples:  "C. 3"  |  "A. Yes"  |  "B. No"  |  "D. Northeast"
"""

# ── SAM 3 palette for overlay colours (deterministic per prompt index) ────────
_PALETTE = [
    np.array([220,  50,  50], dtype=float),   # red
    np.array([ 50, 200,  80], dtype=float),   # green
    np.array([ 50, 120, 220], dtype=float),   # blue
    np.array([220, 180,  50], dtype=float),   # yellow
    np.array([180,  50, 220], dtype=float),   # purple
    np.array([ 50, 200, 200], dtype=float),   # cyan
    np.array([220, 120,  50], dtype=float),   # orange
    np.array([200, 200, 200], dtype=float),   # grey
]

SAM3_THRESHOLD      = 0.4
SAM3_MASK_THRESHOLD = 0.4


# ── Tool factory ──────────────────────────────────────────────────────────────

def _make_segment_batch_tool(image: Image.Image, model_sam, processor_sam, device: str):
    """Build a ``segment_batch`` LangChain tool bound to *image*.

    Runs all text prompts in a single SAM 3 forward pass and returns one
    overlay image + centroid report per prompt.
    """
    from typing import Annotated

    from langchain_core.messages import ToolMessage
    from langchain_core.messages.content import create_image_block
    from langchain_core.tools import tool
    from langchain_core.tools.base import InjectedToolCallId

    image_np = np.array(image)

    @tool
    def segment_batch(
        text_prompts: Annotated[
            list,
            "List of plain-text descriptions of the objects to segment, "
            "e.g. ['red block', 'blue path', 'entrance']. "
            "All prompts are processed in a single forward pass.",
        ],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> ToolMessage:
        """Segment multiple object types in a single SAM 3 forward pass.

        Returns one coloured overlay image and pixel centroid report per prompt.
        """
        if not text_prompts:
            return ToolMessage(
                content="No prompts provided.", tool_call_id=tool_call_id, name="segment_batch"
            )

        # SAM 3 processor accepts a list of images and a list of text prompts.
        # Replicate the image once per prompt so shapes match.
        images_batch = [image] * len(text_prompts)
        inputs = processor_sam(
            images=images_batch, text=text_prompts, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model_sam(**inputs)

        # post_process returns one result dict per image in the batch.
        batch_results = processor_sam.post_process_instance_segmentation(
            outputs,
            threshold=SAM3_THRESHOLD,
            mask_threshold=SAM3_MASK_THRESHOLD,
            target_sizes=[[image.height, image.width]] * len(text_prompts),
        )

        content = []
        rng = random.Random(42)

        for idx, (prompt, res) in enumerate(zip(text_prompts, batch_results)):
            masks  = res["masks"]
            scores = res["scores"]

            base_col = _PALETTE[idx % len(_PALETTE)]
            overlay  = image_np.copy().astype(float)

            centroid_lines = []
            for i, (mask, score) in enumerate(zip(masks, scores.tolist())):
                m = mask.cpu().numpy().astype(bool)
                col = np.clip(base_col + rng.randint(-25, 25), 0, 255)
                overlay[m] = overlay[m] * 0.35 + col * 0.65
                ys, xs = np.where(m)
                if ys.size:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    centroid_lines.append(
                        f"  instance_{i}: centroid=({cx},{cy})px  score={score:.3f}"
                    )

            overlay_img = Image.fromarray(overlay.astype(np.uint8))
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            overlay_b64 = base64.b64encode(buf.getvalue()).decode()

            summary = f"[{idx}] '{prompt}': {len(masks)} instance(s) found.\n"
            summary += (
                "Centroids:\n" + "\n".join(centroid_lines)
                if centroid_lines
                else "No instances detected above the confidence threshold."
            )

            content.append({"type": "text", "text": summary})
            content.append(create_image_block(base64=overlay_b64, mime_type="image/png"))

        return ToolMessage(content=content, tool_call_id=tool_call_id, name="segment_batch")

    return segment_batch


# ── Model class ───────────────────────────────────────────────────────────────

class DeepAgentSAM3:
    """GPT vision agent that uses SAM 3 segmentation as a callable tool.

    The agent autonomously decides what to segment based on the question —
    no task-specific knowledge is baked in.

    Args:
        model_name: OpenAI model identifier (default: ``"gpt-5.2"``).
        max_tokens: Optional token cap per LLM call.
    """

    def __init__(self, model_name: str = "gpt-5.2", max_tokens: int = None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._sam3_loaded = False
        self._model_sam = None
        self._processor_sam = None
        self._device = None

    # ── Lazy SAM 3 loading ────────────────────────────────────────────────────

    def _ensure_sam3(self):
        if self._sam3_loaded:
            return
        from transformers import Sam3Model, Sam3Processor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on mismatched
        # driver/library versions — torch falls back to its own conv kernels.
        torch.backends.cudnn.enabled = False

        print("Loading SAM 3 model...")
        self._model_sam = Sam3Model.from_pretrained(
            "facebook/sam3", torch_dtype=dtype
        ).to(self._device)
        self._processor_sam = Sam3Processor.from_pretrained("facebook/sam3")
        self._sam3_loaded = True

    # ── generate() ────────────────────────────────────────────────────────────

    def generate(
        self,
        query_text: str,
        query_images,
        temperature: float = 0.2,
        max_new_tokens: int = None,
    ):
        """Run the SAM3-augmented agent on one question.

        Args:
            query_text:   Question text (with A/B/C/D options).
            query_images: PIL Image, or None for text-only mode.
            temperature:  Unused (kept for interface compatibility).
            max_new_tokens: Override token limit for this call.

        Returns:
            ``(query_text, answer_text)`` — same as GPT4Vision / DeepAgentGPT.
        """
        self._ensure_sam3()

        from langchain.agents import create_agent
        from langchain.agents.middleware import TodoListMiddleware
        from langchain_core.messages.content import create_image_block
        from langchain_openai import ChatOpenAI

        llm_kwargs = {"model": self.model_name}
        if self.max_tokens or max_new_tokens:
            llm_kwargs["max_tokens"] = max_new_tokens or self.max_tokens
        llm = ChatOpenAI(**llm_kwargs)

        # Build a fresh segment_batch tool bound to this image
        segment_tool = _make_segment_batch_tool(
            query_images, self._model_sam, self._processor_sam, self._device
        ) if query_images is not None else None

        tools = [segment_tool] if segment_tool is not None else []

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=[TodoListMiddleware()],
        )

        # Build user message: text + image (if provided)
        if query_images is not None:
            buf = io.BytesIO()
            query_images.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            content = [
                {"type": "text", "text": query_text},
                create_image_block(base64=img_b64, mime_type="image/png"),
            ]
        else:
            content = query_text

        agent_input = {"messages": [{"role": "user", "content": content}]}
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        response = agent.invoke(agent_input, config=config)
        messages = response.get("messages", [])
        answer_text = messages[-1].content.strip() if messages else ""

        return query_text, answer_text
