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
    
    # Default prompt for food analysis
    if not prompt:
        prompt = """Describe this food image in detail. Include:
        - Type of food/dish
        - Visible ingredients and toppings
        - Cooking style and presentation
        - Colors and textures
        - Any distinctive features"""
    
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


def extract_image_path(query: str) -> str:
    """
    Extract image path from user query if it contains a reference to a pizza image.
    
    Supports patterns like:
    - pizza_not_pizza/pizza/709947.jpg
    - pizza/709947.jpg
    - 709947.jpg
    
    Returns:
        Full path to image if found, else None
    """
    pizza_not_pizza_dir = Path("pizza_not_pizza")
    
    # Pattern 1: pizza_not_pizza/pizza/ID.jpg or pizza_not_pizza/not_pizza/ID.jpg
    match = re.search(r'pizza_not_pizza/(pizza|not_pizza)/(\d+\.jpg)', query)
    if match:
        return str(pizza_not_pizza_dir / match.group(1) / match.group(2))
    
    # Pattern 2: pizza/ID.jpg or not_pizza/ID.jpg
    match = re.search(r'(pizza|not_pizza)/(\d+\.jpg)', query)
    if match:
        return str(pizza_not_pizza_dir / match.group(1) / match.group(2))
    
    # Pattern 3: Bare ID.jpg
    match = re.search(r'(\d{5,7}\.jpg)', query)
    if match:
        filename = match.group(1)
        # Search both subdirectories
        pizza_path = pizza_not_pizza_dir / "pizza" / filename
        not_pizza_path = pizza_not_pizza_dir / "not_pizza" / filename
        
        if pizza_path.exists():
            return str(pizza_path)
        elif not_pizza_path.exists():
            return str(not_pizza_path)
    
    return None