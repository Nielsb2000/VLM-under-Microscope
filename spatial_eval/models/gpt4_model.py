import base64
import time
from io import BytesIO
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries

_RETRYABLE_CODES = {
    # OpenAI server-side errors
    500, 502, 503, 504,
    # Rate limit
    429,
}

# Models that reject any explicit temperature parameter (only the API default is accepted).
# Passing temperature=0.2 (or any non-default value) returns a 400 BadRequestError.
_FIXED_TEMPERATURE_MODELS = ("gpt-5.5",)


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is worth retrying."""
    name = type(exc).__name__
    msg = str(exc)
    # BadRequestError with invalid_prompt — transient false-positive flagging
    if "invalid_prompt" in msg:
        return True
    # Any OpenAI error carrying a retryable HTTP status code
    status = getattr(exc, "status_code", None)
    if status in _RETRYABLE_CODES:
        return True
    # APIConnectionError, APITimeoutError (no status_code)
    if name in ("APIConnectionError", "APITimeoutError", "InternalServerError"):
        return True
    return False


class GPT4Vision:
    """Wrapper for GPT Vision models (GPT-4o, GPT-5.1, GPT-4 Turbo with vision, etc.)"""
    
    def __init__(self, model_name="gpt-4o", max_tokens=None):
        """
        Initialize GPT Vision model.
        
        Args:
            model_name: The OpenAI model name (e.g., "gpt-4o", "gpt-5.1", "gpt-4-turbo")
            max_tokens: Maximum tokens for response
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._fixed_temperature = any(
            tag in model_name.lower() for tag in _FIXED_TEMPERATURE_MODELS
        )
        self.llm = ChatOpenAI(
            model=self.model_name
        )
    
    def _encode_image(self, image):
        """
        Encode PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded image string
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate(self, query_text, query_images, temperature=0.2, max_new_tokens=None):
        """
        Generate response from GPT-4 Vision model.
        
        Args:
            query_text: The text prompt/question
            query_images: PIL Image object or None
            temperature: Sampling temperature
            max_new_tokens: Override max tokens if specified
            
        Returns:
            Tuple of (prompt, answer_text)
        """
        # Prepare message content
        if query_images is not None:
            # Encode image to base64
            image_base64 = self._encode_image(query_images)
            
            # Create message with image and text
            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": query_text
                    }
                ]
            )
        else:
            # Text-only message
            message = HumanMessage(content=query_text)
        
        # Get response from GPT-4 (with retry for transient errors).
        # Some models (e.g. gpt-5.5) reject any explicit temperature — omit it.
        invoke_kwargs = {} if self._fixed_temperature else {"temperature": temperature}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.llm.invoke([message], **invoke_kwargs)
                answer_text = response.content.strip()
                return query_text, answer_text
            except Exception as exc:
                if attempt < MAX_RETRIES and _is_retryable(exc):
                    print(f"[retry {attempt}/{MAX_RETRIES}] {type(exc).__name__}: {exc} — retrying in {RETRY_DELAY}s")
                    time.sleep(RETRY_DELAY)
                else:
                    raise
