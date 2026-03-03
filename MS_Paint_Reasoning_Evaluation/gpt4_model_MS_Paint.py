import base64
from io import BytesIO
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class GPT4VisionMSPaint:
    """Wrapper for GPT Vision models (GPT-4o, GPT-5.1, GPT-4 Turbo with vision, etc.) for MS Paint evaluation with skill referencing and standardized answer format."""
    def __init__(self, model_name="gpt-4o", max_tokens=None, reasoning_mode="medium"):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.reasoning_mode = reasoning_mode
        self.skills_dir = "MS_Paint_Reasoning_Evaluation/skills"
        self.system_prompt = (
            "You are an expert at answering MS Paint reasoning questions. "
            "Your answers must always end with a JSON object containing the key 'final_answer', "
            "where the value is your concise answer. Example: {\"final_answer\": \"your answer here\"}. "
            "Do not output anything after the JSON. Make the JSON easy to parse. "
            "If reasoning is required, provide it before the JSON, but always finish with the JSON object. "
            "\n\nCRITICAL RULE: BEFORE DOING ANYTHING, YOU MUST READ THE MASTER SKILL FILE ({self.skills_dir}/SKILL.md). "
            "MANDATORY WORKFLOW: "
            f"1. FIRST: Read {self.skills_dir}/SKILL.md to identify relevant skill. "
            "2. SECOND: Read the specific skill file to learn the exact commands/syntax. "
            "3. THIRD: Use the information from skills to inform your answer. "
            "ALWAYS: Cite which skill(s) you used in your response reasoning. "
            "ALWAYS: Specify which skill(s) you used in your answer JSON, e.g. {\"final_answer\": ..., \"skills_used\": [\"grayscale-images\"]}. "
            "ALWAYS: Report token usage and time usage in your answer JSON, e.g. {\"final_answer\": ..., \"skills_used\": [...], \"token_usage\": ..., \"time_usage\": ...}. "
            "If you act without reading the relevant skill file first, you are doing it wrong. "
            f"Skills directory: {self.skills_dir}/"
        )
        from deepagents import create_deep_agent
        from langgraph.checkpoint.memory import MemorySaver
        self.checkpointer = MemorySaver()
        from langchain_openai import ChatOpenAI
        import os
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=OPENAI_API_KEY
        )
        # DeepAgents expects tools to be a list of tool objects, not directory strings
        # If you have custom tool loading logic, import or define tool objects here
        # For now, pass an empty list to avoid AttributeError
        self.agent = create_deep_agent(
            model=self.llm,
            checkpointer=self.checkpointer,
            system_prompt=self.system_prompt,
            tools=[],
            debug=True,
        )

    def _encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate(self, query_text, query_images=None, temperature=0.2, max_new_tokens=None, reasoning_mode=None):
        """
        Generate response from GPT-4 Vision deep agent with skill integration and multimodal image support.
        Returns: (prompt, answer_text, token_usage_dict)
        """
        # Prepare deep agent message
        content = []
        content.append({"type": "text", "text": query_text})
        if query_images is not None:
            image_base64 = self._encode_image(query_images)
            content.append({"type": "image", "base64": image_base64, "mime_type": "image/png"})
        agent_input = {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
        config = {"configurable": {"thread_id": "default"}}
        response = self.agent.invoke(agent_input, config=config)
        # Extract answer and token usage
        messages = response.get("messages", [])
        answer_text = messages[-1].content if messages else None
        token_usage = response.get("usage_metadata", None)
        return query_text, answer_text, token_usage
