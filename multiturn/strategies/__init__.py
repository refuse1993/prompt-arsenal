"""
Multi-Turn Attack Strategies
"""

from .base import AttackStrategy
from .visual_storytelling import VisualStorytellingStrategy, StoryDecomposer
from .improved_visual_storytelling import ImprovedVisualStorytellingStrategy
from .crescendo import CrescendoStrategy
from .roleplay import RoleplayStrategy
from .figstep import FigStepStrategy
from .mml_attack import MMLAttackStrategy
from .visual_roleplay import VisualRolePlayStrategy

__all__ = [
    'AttackStrategy',
    'VisualStorytellingStrategy',
    'StoryDecomposer',
    'ImprovedVisualStorytellingStrategy',
    'CrescendoStrategy',
    'RoleplayStrategy',
    'FigStepStrategy',
    'MMLAttackStrategy',
    'VisualRolePlayStrategy'
]
