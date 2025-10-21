"""
Base Attack Strategy
Abstract base class for all multi-turn attack strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class AttackStrategy(ABC):
    """
    Base class for multi-turn attack strategies

    All strategies must implement:
    - generate_next(): Generate the next prompt in the sequence
    - adapt(): Adjust strategy based on target's response
    """

    def __init__(self, name: str):
        """
        Args:
            name: Strategy name (e.g., 'visual_storytelling', 'crescendo')
        """
        self.name = name
        self.state = {}  # Strategy-specific state

    @abstractmethod
    async def generate_next(self, goal: str, conversation: List[Dict], turn: int) -> Dict[str, Any]:
        """
        Generate next prompt in the attack sequence

        Args:
            goal: Attack goal/objective
            conversation: List of previous turns (TurnResult objects)
            turn: Current turn number (0-indexed)

        Returns:
            Prompt dictionary with structure:
            {
                'text': str,
                'images': List[str],  # Image file paths
                'audio': List[str],   # Audio file paths
                'video': List[str]    # Video file paths
            }
        """
        pass

    @abstractmethod
    async def adapt(self, response: str, evaluation: Dict) -> None:
        """
        Adapt strategy based on target's response and evaluation

        Args:
            response: Target LLM's response
            evaluation: Evaluation results from MultiTurnScorer
        """
        pass

    def reset(self):
        """Reset strategy state (for new campaigns)"""
        self.state = {}
