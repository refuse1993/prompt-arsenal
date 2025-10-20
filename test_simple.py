#!/usr/bin/env python3
"""Simple test"""

print("Step 1: Import Console")
from rich.console import Console
console = Console()

print("Step 2: Test prompt")
from rich.prompt import Prompt

console.print("[green]Test started[/green]")
choice = Prompt.ask("Enter something", default="h")
console.print(f"[yellow]You entered: {choice}[/yellow]")

print("Step 3: Done!")
console.print("[green]Success![/green]")
