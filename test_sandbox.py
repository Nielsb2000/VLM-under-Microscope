"""
Quick test to verify AIO Sandbox integration works correctly.
Run this after starting docker-compose to check connectivity.
"""

import os
from sandbox_tools import get_sandbox_context, execute_python_code, execute_shell_command

def test_sandbox_connection():
    """Test basic sandbox connectivity and operations."""
    print("=" * 60)
    print("AIO Sandbox Integration Test")
    print("=" * 60)
    
    # Test 1: Get sandbox context
    print("\n1️⃣  Testing sandbox connectivity...")
    result = get_sandbox_context()
    if result["success"]:
        print(f"   ✓ Connected to sandbox!")
        print(f"   Home: {result.get('home_dir')}")
        print(f"   User: {result.get('user')}")
        print(f"   Workspace: {result.get('workspace')}")
    else:
        print(f"   ✗ Failed: {result.get('error')}")
        print("\n   Make sure to start the sandbox first:")
        print("   docker-compose up -d")
        return False
    
    # Test 2: Execute Python code
    print("\n2️⃣  Testing Python code execution...")
    result = execute_python_code("print('Hello from AIO Sandbox!')")
    if result["success"]:
        print(f"   ✓ Python execution works!")
        print(f"   Output: {result.get('output')}")
    else:
        print(f"   ✗ Failed: {result.get('error')}")
        return False
    
    # Test 3: Execute shell command
    print("\n3️⃣  Testing shell command execution...")
    result = execute_shell_command("echo 'Shell command works!' && pwd")
    if result["success"]:
        print(f"   ✓ Shell execution works!")
        print(f"   Output: {result.get('output')}")
    else:
        print(f"   ✗ Failed: {result.get('error')}")
        return False
    
    # Test 4: Check mounted volumes
    print("\n4️⃣  Testing volume mounts...")
    result = execute_shell_command("ls -la /workspace/")
    if result["success"]:
        output = result.get('output', '')
        has_skills = 'skills' in output
        has_pizza = 'pizza_not_pizza' in output
        
        if has_skills and has_pizza:
            print(f"   ✓ Both volumes are mounted!")
            print(f"     - /workspace/skills/")
            print(f"     - /workspace/pizza_not_pizza/")
        else:
            print(f"   ⚠ Partial mount:")
            print(f"     - skills: {'✓' if has_skills else '✗'}")
            print(f"     - pizza_not_pizza: {'✓' if has_pizza else '✗'}")
    else:
        print(f"   ✗ Failed: {result.get('error')}")
        return False
    
    # Test 5: List skills
    print("\n5️⃣  Testing skills directory access...")
    result = execute_shell_command("ls -la /workspace/skills/")
    if result["success"]:
        output = result.get('output', '')
        skill_count = output.count('SKILL.md')
        print(f"   ✓ Can access skills directory!")
        print(f"   Found {skill_count} skill(s)")
    else:
        print(f"   ✗ Failed: {result.get('error')}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Sandbox integration is working.")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python main.py")
    print("\nOr visit the sandbox web interface:")
    print("  http://localhost:8080/vnc/index.html?autoconnect=true")
    return True


if __name__ == "__main__":
    test_sandbox_connection()
