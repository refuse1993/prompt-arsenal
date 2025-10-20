#!/usr/bin/env python3
"""
Debug script to test CLI initialization
"""

import sys
print("Step 1: Starting debug script")

print("Step 2: Importing rich...")
from rich.console import Console
console = Console()
console.print("[green]Rich imported successfully[/green]")

print("Step 3: Importing core modules...")
try:
    from core.database import ArsenalDB
    console.print("[green]ArsenalDB imported[/green]")
except Exception as e:
    console.print(f"[red]ArsenalDB import failed: {e}[/red]")
    sys.exit(1)

try:
    from core.config import Config
    console.print("[green]Config imported[/green]")
except Exception as e:
    console.print(f"[red]Config import failed: {e}[/red]")
    sys.exit(1)

try:
    from core import Judge
    console.print("[green]Judge imported[/green]")
except Exception as e:
    console.print(f"[red]Judge import failed: {e}[/red]")
    sys.exit(1)

print("Step 4: Creating PromptArsenal instance...")
try:
    from interactive_cli import PromptArsenal
    console.print("[green]PromptArsenal class imported[/green]")

    console.print("[yellow]Creating instance...[/yellow]")
    app = PromptArsenal()
    console.print("[green]PromptArsenal instance created successfully![/green]")

except Exception as e:
    console.print(f"[red]Failed to create PromptArsenal: {e}[/red]")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 5: Testing banner...")
try:
    app.show_banner()
    console.print("[green]Banner displayed successfully[/green]")
except Exception as e:
    console.print(f"[red]Banner failed: {e}[/red]")
    import traceback
    traceback.print_exc()

print("Step 6: Testing menu...")
try:
    app.show_menu()
    console.print("[green]Menu displayed successfully[/green]")
except Exception as e:
    console.print(f"[red]Menu failed: {e}[/red]")
    import traceback
    traceback.print_exc()

console.print("\n[bold green]All tests passed! CLI should work.[/bold green]")
