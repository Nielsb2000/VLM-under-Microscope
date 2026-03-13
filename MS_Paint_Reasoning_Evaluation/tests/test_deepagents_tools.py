#!/usr/bin/env python3
"""
Minimal test to debug why deepagents isn't using tools WITH IMAGE
"""
import os
import base64
from io import BytesIO
from PIL import Image
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_openai import ChatOpenAI

# Set up paths
root_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Root dir: {root_dir}")

# Create backend
backend = FilesystemBackend(root_dir=root_dir, virtual_mode=False)
print(f"Backend created: {backend}")
print(f"Backend has read method: {hasattr(backend, 'read')}")

# Test backend directly
try:
    content = backend.read("skills/master-skill/SKILL.md")
    print(f"Direct backend read works! Content length: {len(content)}")
except Exception as e:
    print(f"Direct backend read failed: {e}")

# Create LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Load and encode image
image_path = os.path.join(root_dir, "MS_paint_images/original_images/img1.png")
print(f"\nLoading image: {image_path}")
img = Image.open(image_path)
buffered = BytesIO()
img.save(buffered, format="PNG")
image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
print(f"Image encoded, length: {len(image_base64)}")

# Create agent with skills
skills = ["skills/master-skill"]
system_prompt = """You are an MS Paint reasoning assistant with access to specialized skills.

CRITICAL REQUIREMENT: Follow this EXACT workflow BEFORE answering:

STEP 1: READ MASTER SKILL
- MANDATORY: Call read_file to read skills/master-skill/SKILL.md
- LIST ALL 4 SKILLS you find in the routing table

STEP 2: ANALYZE INPUT TO DETERMINE SKILL ROUTING
- Look at the PROVIDED IMAGE to determine its type:
  * Does it have color? → Use colored-images skill
  * Only grayscale (no color)? → Use grayscale-images skill
  * Inverted/negative grayscale? → Use inverted-grayscale-images skill
- Read the QUESTION to check if it asks about shape identification:
  * Keywords: 'shape', 'identify', 'what shape', 'circle', 'square', 'triangle', 'star', etc.
  * If YES → ALSO use recognizing-shapes skill

STEP 3: READ RELEVANT SKILL FILE(S)
- Call read_file for the domain skill (colored-images, grayscale-images, or inverted-grayscale-images)
- If question is about shapes, ALSO call read_file for recognizing-shapes/SKILL.md
- Read any example images referenced in the skill files if needed

STEP 4: ANSWER THE QUESTION
- Use the guidance from the skill files you read
- Analyze the provided input image
- Apply the protocols and reasoning methods from the skills

IMPORTANT NOTES:
- The input image is PROVIDED INLINE in the message. Use read_file ONLY for skill files and example images.
- You WILL BE PENALIZED if you skip reading skills or don't follow the routing protocol.
- In your reasoning, EXPLICITLY state which skills you accessed and why."""

print(f"\nCreating agent with skills={skills}")
agent = create_deep_agent(
    model=llm,
    system_prompt=system_prompt,
    skills=skills,
    backend=backend,
    debug=True,
)

# Test agent WITH IMAGE
agent_input = {
    "messages": [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "Question 1: For the 2 shapes in the middle of the image, which shape is in front of the other one?"},
                {"type": "image", "base64": image_base64, "mime_type": "image/png"}
            ]
        }
    ],
    "reasoning": {
        "effort": "low",
        "summary": "auto",
    }
}

print("\nInvoking agent...")
for chunk in agent.stream(agent_input):
    print(f"Chunk: {chunk}")
    if 'model' in chunk and 'messages' in chunk['model']:
        for msg in chunk['model']['messages']:
            if hasattr(msg, 'tool_calls'):
                print(f"TOOL CALLS: {msg.tool_calls}")

print("\nTest complete!")
