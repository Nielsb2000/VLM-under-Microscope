from llm_client import get_default_llm
from pathlib import Path
from image_utils import caption_with_gpt4_vision, extract_image_path
from skills_utils import discover_skills, get_skills_summary


def main():
    print("Ask questions to the LLM (type 'quit' to exit)")
    print("You can reference pizza images by filename (e.g., '709947.jpg')")
    print("💡 The agent can execute code and commands in the AIO Sandbox")
    print("   - Python code: runs in sandbox, output appears in sandbox terminal")
    print("   - Shell commands: executed in sandbox environment")
    print("   - Files: read/write in sandbox at /workspace/\n")
    
    # Discover and display available skills
    skills = discover_skills(Path("./skills"))
    if skills:
        print(get_skills_summary(skills))
        print()
    
    agent = get_default_llm()
    image_caption_cache = {}  # Cache captions to avoid re-generating
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not question:
            continue
        
        # Check if query references an image
        image_path = extract_image_path(question)
        query_to_send = question
        
        if image_path:
            # Generate or retrieve cached caption
            if image_path not in image_caption_cache:
                print("🔍 Analyzing image...", end="", flush=True)
                try:
                    # Custom prompt focused on recipe details
                    recipe_prompt = """Describe this pizza image in detail for recipe generation purposes. Include:
- Pizza style (Neapolitan, New York, Detroit, etc.)
- Crust characteristics (thin/thick, charred spots, texture)
- Cheese type and coverage
- All visible toppings and ingredients
- Sauce type and amount
- Cooking method indicators (oven marks, char patterns)
- Any distinctive features or garnishes"""
                    
                    caption = caption_with_gpt4_vision(image_path, prompt=recipe_prompt)
                    image_caption_cache[image_path] = caption
                    print(" Done! ✓\n")
                except Exception as e:
                    print(f" Failed: {e}\n")
                    continue
            else:
                caption = image_caption_cache[image_path]
                print(f"📸 Using cached analysis for {Path(image_path).name}\n")
            
            # Transform query to include image context
            query_to_send = f"""I want to make a pizza that looks like this:

[Image: {Path(image_path).name}]
{caption}
"""
            
        try:
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query_to_send,
                        }
                    ],
                },
                config={"configurable": {"thread_id": "default"}},
            )

            print("\n--- INTERMEDIATE STEPS ---")
            for key, value in result.items():
                if key != "messages":
                    print(f"{key}: {value}")
            print("--- END STEPS ---\n")

            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                print(f"Assistant: {last_message.content}\n")
        except Exception as e:
            import traceback
            print(f"Error: {type(e).__name__}: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            print()
 

if __name__ == "__main__":
    main()
