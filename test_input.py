#!/usr/bin/env python3
"""Test different input methods"""

print("Test 1: Standard input()")
try:
    result = input("Enter something: ")
    print(f"✓ Standard input works: {result}")
except Exception as e:
    print(f"✗ Standard input failed: {e}")

print("\nTest 2: Rich Prompt.ask()")
try:
    from rich.prompt import Prompt
    result = Prompt.ask("Enter something")
    print(f"✓ Rich Prompt works: {result}")
except Exception as e:
    print(f"✗ Rich Prompt failed: {e}")

print("\nTest 3: Rich Console.input()")
try:
    from rich.console import Console
    console = Console()
    result = console.input("Enter something: ")
    print(f"✓ Rich Console.input works: {result}")
except Exception as e:
    print(f"✗ Rich Console.input failed: {e}")
