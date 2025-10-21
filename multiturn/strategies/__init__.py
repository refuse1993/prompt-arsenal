"""
Multi-Turn Attack Strategies
"""

from .base import AttackStrategy
from .visual_storytelling import VisualStorytellingStrategy, StoryDecomposer
from .crescendo import CrescendoStrategy
from .roleplay import RoleplayStrategy

__all__ = [
    'AttackStrategy',
    'VisualStorytellingStrategy',
    'StoryDecomposer',
    'CrescendoStrategy',
    'RoleplayStrategy'
]
