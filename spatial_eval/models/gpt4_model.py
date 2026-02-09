import base64
from io import BytesIO
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class GPT4Vision:
    """Wrapper for GPT-4 Vision models (GPT-4o, GPT-4 Turbo with vision, etc.)"""
    
    def __init__(self, model_name="gpt-4o", max_tokens=500):
        """
        Initialize GPT-4 Vision model.
        
        Args:
            model_name: The OpenAI model name (e.g., "gpt-4o", "gpt-4-turbo")
            max_tokens: Maximum tokens for response
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.llm = ChatOpenAI(
            model=self.model_name,
            max_tokens=self.max_tokens
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
        # Override max_tokens if specified
        if max_new_tokens is not None:
            self.llm.max_tokens = max_new_tokens
        
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
        
        # Get response from GPT-4
        response = self.llm.invoke([message], temperature=temperature)
        answer_text = response.content.strip()
        
        return query_text, answer_text
