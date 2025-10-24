"""
API Blueprints Package
Modular API endpoints for Prompt Arsenal Dashboard
"""

from .prompts import prompts_bp
from .multimodal import multimodal_bp
from .multiturn import multiturn_bp
from .ctf import ctf_bp
from .security import security_bp
from .system import system_bp
from .stats import stats_bp

__all__ = [
    'prompts_bp',
    'multimodal_bp',
    'multiturn_bp',
    'ctf_bp',
    'security_bp',
    'system_bp',
    'stats_bp'
]
