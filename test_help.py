#!/usr/bin/env python3
"""
Test help menu display
"""
import sys

# Add project to path
sys.path.insert(0, '/Users/brownkim/Downloads/ACDC/prompt_arsenal')

from interactive_cli import PromptArsenal

print("Step 1: Creating PromptArsenal instance...")
app = PromptArsenal()

print("Step 2: Testing show_help()...")
app.show_help()

print("\n✓ Help menu displayed successfully!")
