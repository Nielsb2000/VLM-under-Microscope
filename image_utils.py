from langchain_openai import ChatOpenAI
import base64
from pathlib import Path
import re

def caption_with_gpt4_vision(image_path: str, prompt: str = None) -> str:
    """
    Generate detailed caption using GPT-4 Vision.
    
    Args:
        image_path: Path to the image file
        prompt: Custom prompt (default asks for detailed food description)
    
    Returns:
        Detailed caption string
    """
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Default prompt for general image analysis
    if not prompt:
        prompt = "Describe this image in detail, including its main subjects, colors, composition, and any notable features."
    
    llm = ChatOpenAI(model="gpt-4o", max_tokens=300)
    
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            }
        ]
    }
    
    response = llm.invoke([message])
    return response.content