import base64
from io import BytesIO
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class GPT4VisionMSPaint:
    """Wrapper for GPT Vision models (GPT-4o, GPT-5.1, GPT-4 Turbo with vision, etc.) for MS Paint evaluation with token/time reporting and reasoning mode support."""
    def __init__(self, model_name="gpt-4o", max_tokens=5000, reasoning_mode="medium"):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.reasoning_mode = reasoning_mode
        self.llm = ChatOpenAI(
            model=self.model_name,
            max_tokens=self.max_tokens
        )

    def _encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate(self, query_text, query_images, temperature=0.2, max_new_tokens=None, reasoning_mode=None):
        """
        Generate response from GPT-4 Vision model with reasoning mode as API parameter.
        Returns: (prompt, answer_text, token_usage_dict)
        """
        mode = reasoning_mode if reasoning_mode is not None else self.reasoning_mode
        if max_new_tokens is not None:
            self.llm.max_tokens = max_new_tokens
        # Prepare message
        if query_images is not None:
            image_base64 = self._encode_image(query_images)
            message = HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": query_text}
                ]
            )
        else:
            message = HumanMessage(content=query_text)

        # Pass reasoning as API parameter if supported
        # NOTE: This assumes langchain_openai.ChatOpenAI supports extra keyword arguments for OpenAI API
        # If not, you may need to use openai directly or update your langchain version
        # Enable reasoning summary for supported models
        reasoning_param = {"effort": mode}
        if self.model_name in ("gpt-5.1", "gpt-5.2", "gpt-5-reasoning"):  # Add other supported models as needed
            reasoning_param["summary"] = "auto"
        response = self.llm.invoke(
            [message],
            temperature=temperature,
            reasoning=reasoning_param
        )
        # Handle response.content as string or list
        if isinstance(response.content, list):
            combined_text = '\n'.join([str(x) for x in response.content])
        else:
            combined_text = str(response.content)
        token_usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_usage = dict(response.usage_metadata)
        return query_text, combined_text, token_usage
