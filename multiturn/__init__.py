"""
Multi-Turn Attack Framework
"""

from .orchestrator import MultiTurnOrchestrator, TurnResult
from .conversation_manager import ConversationManager
from .scorer import MultiTurnScorer
from .strategies import (
    AttackStrategy,
    VisualStorytellingStrategy,
    StoryDecomposer,
    CrescendoStrategy,
    RoleplayStrategy
)

__all__ = [
    'MultiTurnOrchestrator',
    'TurnResult',
    'ConversationManager',
    'MultiTurnScorer',
    'AttackStrategy',
    'VisualStorytellingStrategy',
    'StoryDecomposer',
    'CrescendoStrategy',
    'RoleplayStrategy'
]
